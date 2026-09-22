from __future__ import annotations

import asyncio
import json
import logging
import time

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .chat_schemas import ChatResponse, ChatSource, EvidenceSelection
from .gemini import GeminiFailure
from .schemas import CatalogSearch

LOGGER = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 6000


def evidence_context(results) -> list[dict]:
    context = []
    size = 2  # JSON array brackets
    for hit in results:
        for evidence_id, text in zip(hit.evidence_ids, hit.evidence):
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
    return ChatResponse(
        mode=result.mode, answer=f"{result.message}\n\n{names}.",
        sources=[ChatSource()], catalog=result.catalog,
        suggestions=result.suggestions, llm_status=status,
    )


def evidence_reply(rows: list[dict], status: str) -> ChatResponse:
    sources: dict[int, ChatSource] = {}
    paragraphs = []
    for row in rows:
        source = sources.setdefault(row["id"], ChatSource(id=row["id"], label=row["label"]))
        source.evidence_ids.append(row["evidence_id"])
        paragraphs.append(f'{row["nama"]}: {row["text"]}')
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
        result = await run_in_threadpool(self.retrieval.search, message, 3)
        if isinstance(result, CatalogSearch):
            return catalog_reply(result)
        context = evidence_context(result.results)
        if not context:
            return catalog_reply(self.retrieval.catalog.response(message, result.normalized_query, True), "insufficient_evidence")
        status = "not_configured"
        selected = context[:3]
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
        return evidence_reply(selected, status)
