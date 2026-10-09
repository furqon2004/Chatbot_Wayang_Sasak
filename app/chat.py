from __future__ import annotations

import asyncio
import json
import logging
import time
import math
import re
import random

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .chat_schemas import ChatResponse, ChatSource, EvidenceSelection
from .gemini import GeminiFailure
from .schemas import CatalogSearch
from .query_normalizer import classify_intent, unsafe_evidence, normalize_query, general_knowledge_intent

LOGGER = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 6000
CATALOG_SCOPE = ("Aplikasi ini menyediakan profil rinci dan pengenalan gambar untuk 20 tokoh Wayang Sasak. "
                 "Angka 20 adalah cakupan katalog aplikasi, bukan jumlah seluruh tokoh dalam tradisi Wayang Sasak.")


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
    for _, text in rendered_evidence(rows):
        paragraphs.append(text)
    return ChatResponse(
        mode="answer", answer="\n\n".join(paragraphs),
        sources=list(sources.values()), llm_used=status == "used", llm_status=status,
        suggestions=[f'Apa ciri-ciri {rows[0]["nama"]}?'],
    )


class ChatService:
    def __init__(self, retrieval, gemini=None, timeout_seconds: float = 15, web_timeout_seconds: float = 90, clock=time.monotonic):
        from .rag_cache import RagCache
        self.rag_cache = RagCache(clock)
        self.retrieval = retrieval
        self.gemini = gemini
        self.timeout_seconds = timeout_seconds
        self.web_timeout_seconds = web_timeout_seconds

    async def chat(self, message: str, *, awaiting_more=False, seen_ids=(), character_id=None, explain_more=False) -> ChatResponse:
        normalized = normalize_query(message).strip(" .!?")
        if awaiting_more and normalized in {"ya", "iya", "iya dong", "ya mau", "mau", "boleh", "boleh dong", "oke", "ok", "lanjut", "yuk", "tentu"}:
            candidates = [item for item in self.retrieval.catalog.items if item.id not in seen_ids]
            if len(candidates) < 3:
                candidates += [item for item in self.retrieval.catalog.items if item.id in seen_ids]
            choices = random.SystemRandom().sample(candidates, min(3, len(candidates)))
            return ChatResponse(mode="suggestions", answer="Mari kenali tokoh lainnya. Pilih salah satu dari tiga tokoh berikut:",
                                sources=[], choices=choices)
        if awaiting_more and normalized in {"tidak", "nggak", "enggak", "tidak dulu", "nggak dulu", "cukup", "nanti"}:
            return ChatResponse(mode="conversation", answer="Baik, kapan pun Anda ingin melanjutkan, tulis pertanyaan atau kirim gambar wayang di sini.", sources=[])
        decision = classify_intent(message, self.retrieval.catalog)
        if decision.intent in {"unsupported", "malicious"}:
            return catalog_reply(self.retrieval.catalog.response(message, "", True))
        if character_id is not None:
            if explain_more:
                return await self._answer(message, character_id=character_id)
            return self.character_profile(character_id)
        query = normalize_query(decision.query)
        count_question = bool(re.search(r"\b(?:jumlah|berapa|banyak|total)\b", query) and
                              re.search(r"\b(?:tokoh|karakter|wayang)\b", query))
        if count_question and re.search(r"\b(?:sistem|aplikasi|katalog|database)\b", query) and not re.search(r"\b(?:seluruh|sebenarnya|tradisi|keseluruhan)\b", query):
            return ChatResponse(mode="conversation", answer=CATALOG_SCOPE, sources=[ChatSource()],
                                follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")
        story_question = bool(re.search(r"\b(?:sejarah|cerita|kisah|lakon|asal usul)\b", query))
        named_character = any(re.search(r"(?<!\w)" + re.escape(normalize_query(name)) + r"(?!\w)", query)
                              for item in self.retrieval.catalog.items for name in (item.nama, item.label))
        if not named_character and (general_knowledge_intent(query) or count_question or story_question):
            return await self._web_answer(decision.query)
        reply = await self._answer(decision.query, decision.comparison_ids, original_message=message)
        reply.scope_notice = decision.scope_notice
        return reply

    def character_profile(self, character_id):
        catalog = self.retrieval.catalog
        item = catalog.items[character_id]
        profile = catalog.profiles[character_id]
        description = "deskripsi_lengkap" if profile.get("deskripsi_lengkap") else "deskripsi"
        paragraphs, evidence_ids = [], []
        for field in (description, "filosofi", "ciri_ciri"):
            text = profile.get(field)
            if isinstance(text, str) and text.strip() and not unsafe_evidence(text):
                if text.strip() not in paragraphs:
                    paragraphs.append(text.strip())
                evidence_ids.append(f"profile:{character_id}:{field}:0")
        if not paragraphs:
            return catalog_reply(catalog.response(item.nama, "", True))
        return ChatResponse(mode="answer", answer="\n\n".join(paragraphs), character=item,
            sources=[ChatSource(id=item.id, label=item.label, evidence_ids=evidence_ids)],
            llm_status="not_needed", follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")

    async def _web_answer(self, message):
        status = "not_configured"
        if self.gemini is not None and callable(getattr(self.gemini, "search", None)):
            try:
                answer, sources, suggestions_html = await asyncio.wait_for(self.gemini.search(message), timeout=self.web_timeout_seconds)
                if not sources:
                    raise GeminiFailure("insufficient_evidence")
                return ChatResponse(mode="web_answer", answer=answer, sources=[], web_sources=sources,
                                    search_suggestions_html=suggestions_html, llm_used=True, llm_status="used",
                                    scope_notice=CATALOG_SCOPE,
                                    follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")
            except GeminiFailure as exc:
                status = exc.category
            except asyncio.TimeoutError:
                status = "timeout"
            except Exception:
                status = "internal"
        LOGGER.warning("Web chat failed: category=%s", status)
        return ChatResponse(mode="fallback_catalog",
            answer="Informasi internet belum dapat diverifikasi saat ini. Basis pengetahuan Wayang AI memuat 20 tokoh Wayang Sasak yang dapat Anda jelajahi.",
            sources=[ChatSource()], catalog=self.retrieval.catalog.items,
            llm_status="unavailable", follow_up="Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")

    async def _answer(self, message: str, comparison_ids=frozenset(), *, character_id=None, original_message=None) -> ChatResponse:
        # General overview is routed to an explicit cultural retrieval query.
        # Ranking/threshold are unchanged; insufficient evidence still falls back.
        retrieval_query = ("Siapa tokoh utama Wayang Sasak?" if normalize_query(message) in
                           {"jelaskan wayang sasak", "apa itu wayang sasak", "explain wayang sasak"} else message)
        if character_id is not None:
            retrieval_query = "Jelaskan " + self.retrieval.catalog.items[character_id].nama
        result = await run_in_threadpool(self.retrieval.search, retrieval_query, 3)
        if isinstance(result, CatalogSearch):
            return catalog_reply(result)
        context = evidence_context(
            [hit for hit in result.results if math.isfinite(hit.score) and hit.score >= self.retrieval.threshold],
            self.retrieval.catalog,
        )
        if character_id is not None:
            context = [row for row in context if row["id"] == character_id]
        if comparison_ids and not comparison_ids <= {row["id"] for row in context}:
            context = []
        if not context:
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True))
        cache_key = self.rag_cache.key(original_message or message, message, context,
            character_id, comparison_ids, self.gemini)
        cached = self.rag_cache.get(cache_key)
        if cached is not None:
            return cached
        status = "not_configured"
        follow_up = "Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?"
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
                follow_up = selection.follow_up
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
        reply = evidence_reply(selected, status)
        reply.character = self.retrieval.catalog.items[selected[0]["id"]]
        reply.follow_up = follow_up
        reply.suggestions = []
        if status == "used":
            self.rag_cache.put(cache_key, reply)
        return reply
