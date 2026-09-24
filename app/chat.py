from __future__ import annotations

import asyncio
import json
import logging
import time
import math
import re

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .chat_schemas import ChatResponse, ChatSource, EvidenceSelection
from .gemini import GeminiFailure
from .schemas import CatalogSearch
from .query_normalizer import classify_intent, unsafe_evidence, normalize_query

LOGGER = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 6000


def evidence_context(results, catalog=None) -> list[dict]:
    context = []
    size = 2  # JSON array brackets
    for hit in results:
        if (not all(math.isfinite(getattr(hit, key)) and 0 <= getattr(hit, key) <= 1
                    for key in ("score", "semantic_score", "lexical_score"))
                or unsafe_evidence(hit.nama) or len(hit.nama) > 120):
            continue
        for evidence_id, text in zip(hit.evidence_ids, hit.evidence):
            if len(context) >= 12 or not text.strip() or len(text) > 500 or unsafe_evidence(text):
                continue
            if any(row["evidence_id"] == evidence_id for row in context):
                continue
            if catalog is not None:
                try:
                    prefix, class_id, field, index = evidence_id.split(":")
                    if field not in {"deskripsi", "deskripsi_lengkap", "filosofi", "ciri_ciri", "kata_kunci"}:
                        continue
                    raw = catalog.profiles[hit.id][field]
                    values = raw if isinstance(raw, list) else [raw]
                    original = values[int(index)]
                    if (prefix != "profile" or int(class_id) != hit.id or int(index) < 0
                            or hit.label != catalog.labels[hit.id] or hit.nama != catalog.items[hit.id].nama
                            or not isinstance(original, str) or unsafe_evidence(original)
                            or original[:500] != text):
                        continue
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
            row = {"evidence_id": evidence_id, "id": hit.id, "label": hit.label,
                   "nama": hit.nama, "text": text}
            row_size = len(json.dumps(row, ensure_ascii=False)) + (2 if context else 0)
            if size + row_size > MAX_CONTEXT_CHARS:
                continue
            size += row_size
            context.append(row)
    return context


def catalog_reply(result: CatalogSearch, status="not_needed") -> ChatResponse:
    # Tidak ada teks Gemini pada mode katalog: nama tidak bisa ditambah/diganti.
    names = ", ".join(item.nama for item in result.catalog)
    if len(names) > 3000 or any(unsafe_evidence(item.nama) for item in result.catalog):
        raise ValueError("Katalog tidak aman")
    return ChatResponse(
        mode=result.mode, answer=f"{result.message}\n\n{names}.",
        sources=[ChatSource()], catalog=result.catalog,
        suggestions=result.suggestions, llm_status=status,
    )


def display_excerpt(text: str) -> str:
    """Render capped evidence without a possibly incomplete final word."""
    if len(text) < 500:
        return text.strip()
    clipped = text[:500]
    endings = list(re.finditer(r'[.!?](?:["\u201d\u2019]*)?(?=\s|$)', clipped))
    if endings:
        return clipped[:endings[-1].end()].strip()
    # At the evidence cap we cannot know whether the last token is complete.
    boundaries = list(re.finditer(r'\s+', clipped))
    return clipped[:boundaries[-1].start()].strip() if boundaries else ""


def rendered_evidence(rows: list[dict]) -> list[tuple[dict, str]]:
    """Keep the longest contained quote per character, with stable tie ordering."""
    candidates = [(index, row, display_excerpt(row["text"])) for index, row in enumerate(rows)]
    kept = []
    for index, row, text in sorted(candidates, key=lambda item: (
            -len(" ".join(item[2].split()).casefold()), item[0])):
        normalized = " ".join(text.split()).casefold()
        if not normalized:
            continue
        if any(row["id"] == other["id"] and re.search(
                r'(?<!\w)' + re.escape(normalized) + r'(?!\w)',
                " ".join(quote.split()).casefold()) for _, other, quote in kept):
            continue
        kept.append((index, row, text))
    return [(row, text) for _, row, text in sorted(kept, key=lambda item: item[0])]


def evidence_reply(rows: list[dict], status: str) -> ChatResponse:
    sources: dict[int, ChatSource] = {}
    paragraphs = []
    for row in rows:
        source = sources.setdefault(row["id"], ChatSource(id=row["id"], label=row["label"]))
        if row["evidence_id"] not in source.evidence_ids:
            source.evidence_ids.append(row["evidence_id"])
    for row, text in rendered_evidence(rows):
        paragraphs.append(f'{row["nama"]}: {text}')
    return ChatResponse(
        mode="answer", answer="Berdasarkan basis pengetahuan Wayang Sasak:\n\n" + "\n\n".join(paragraphs),
        sources=list(sources.values()), llm_used=status == "used", llm_status=status,
        suggestions=[f'Apa ciri-ciri {rows[0]["nama"]}?'],
    )


class ChatService:
    def __init__(self, retrieval, gemini=None, timeout_seconds: float = 15):
        self.retrieval = retrieval
        self.gemini = gemini
        self.timeout_seconds = timeout_seconds

    async def chat(self, message: str) -> ChatResponse:
        decision = classify_intent(message, self.retrieval.catalog)
        if decision.intent in {"unsupported", "malicious"}:
            return catalog_reply(self.retrieval.catalog.response(message, "", True))
        reply = await self._answer(decision.query, decision.comparison_ids)
        reply.scope_notice = decision.scope_notice
        return reply

    async def _answer(self, message: str, comparison_ids=frozenset()) -> ChatResponse:
        # General overview is routed to an explicit cultural retrieval query.
        # Ranking/threshold are unchanged; insufficient evidence still falls back.
        retrieval_query = ("Siapa tokoh utama Wayang Sasak?" if normalize_query(message) in
                           {"jelaskan wayang sasak", "apa itu wayang sasak", "explain wayang sasak"} else message)
        result = await run_in_threadpool(self.retrieval.search, retrieval_query, 3)
        if isinstance(result, CatalogSearch):
            return catalog_reply(result)
        context = evidence_context(
            [hit for hit in result.results if math.isfinite(hit.score) and hit.score >= self.retrieval.threshold],
            self.retrieval.catalog,
        )
        if comparison_ids and not comparison_ids <= {row["id"] for row in context}:
            context = []
        if not context:
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True))
        status = "not_configured"
        selected = context[:3]
        if comparison_ids and not comparison_ids <= {row["id"] for row in selected}:
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True))
        if self.gemini is not None:
            started = time.perf_counter()
            try:
                raw = await asyncio.wait_for(self.gemini.select(message, context), timeout=self.timeout_seconds)
                if not isinstance(raw, str) or len(raw) > 8192:
                    raise ValueError("Respons malformed")
                selection = EvidenceSelection.model_validate_json(raw)
                by_id = {row["evidence_id"]: row for row in context}
                if len(selection.evidence_ids) != len(set(selection.evidence_ids)) or any(
                    evidence_id not in by_id for evidence_id in selection.evidence_ids
                ):
                    raise ValueError("ID evidence tidak valid")
                if not selection.evidence_ids:
                    status = "insufficient_evidence"
                    return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True), "insufficient_evidence")
                selected = [by_id[evidence_id] for evidence_id in selection.evidence_ids]
                if comparison_ids and not comparison_ids <= {row["id"] for row in selected}:
                    return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True), "insufficient_evidence")
                status = "used"
            except asyncio.TimeoutError:
                status = "timeout"
            except GeminiFailure as exc:
                status = exc.category
            except (ValidationError, ValueError):
                status = "invalid_response"
            except Exception:
                status = "internal"
            finally:
                LOGGER.info("Chat Gemini selesai: category=%s duration=%.3f", status, time.perf_counter() - started)
        if not rendered_evidence(selected):
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True),
                                 "not_needed" if self.gemini is None else "insufficient_evidence")
        return evidence_reply(selected, status)
