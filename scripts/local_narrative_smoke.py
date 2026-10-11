"""One local, direct-TLS narrative smoke. Run explicitly from Windows PowerShell."""
import asyncio
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class Diagnostics(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        # Only the adapter's fixed, sanitized metadata format is accepted.
        from app.openai_rag import SAFE_CATEGORIES, SAFE_EXCEPTION_TYPES, SAFE_ERROR_CODES
        if record.msg != ("RAG request failed: provider=openai category=%s "
                          "exception_type=%s http_status=%s provider_code=%s"):
            return
        category, kind, status, code = record.args
        self.rows.append({
            "category": category if category in SAFE_CATEGORIES else "internal",
            "exception_type": kind if kind in SAFE_EXCEPTION_TYPES else "OtherError",
            "http_status": status if type(status) is int and 100 <= status <= 599 else None,
            "provider_code": code if type(code) is str and code in SAFE_ERROR_CODES else None,
        })


class OneCall:
    def __init__(self, create):
        self.create = create
        self.count = 0

    async def __call__(self, **kwargs):
        if self.count:
            raise RuntimeError("Smoke SDK budget exhausted")
        self.count += 1
        return await self.create(**kwargs)


async def smoke(settings, retrieval, *, transport=None, query="Apa filosofi Jayangrana?",
                conversation_context=None):
    """Use the real SDK, adapter and ChatService; transport injection is for tests."""
    from dataclasses import replace
    import httpx
    from openai import AsyncOpenAI
    from app.chat import ChatService
    from app.narrative import NARRATIVE_VERSION
    from app.openai_rag import OpenAIRagClient
    if NARRATIVE_VERSION != "local-narrative-v2":
        raise ValueError("Unexpected narrative contract")
    settings = replace(settings, llm_provider="openai", llm_model="gpt-4.1-mini",
                       llm_max_attempts=1, chat_enabled=True)
    if not settings.llm_api_key:
        raise ValueError("Missing credential")
    diagnostics = Diagnostics()
    logger = logging.getLogger("app.openai_rag")
    previous = logger.level, logger.disabled, logger.propagate
    logger.setLevel(logging.WARNING)
    logger.disabled, logger.propagate = False, False
    logger.addHandler(diagnostics)
    try:
        async with httpx.AsyncClient(trust_env=False, verify=True, follow_redirects=False,
                                    timeout=settings.llm_timeout_seconds, transport=transport) as http:
            async with AsyncOpenAI(api_key=settings.llm_api_key, base_url="https://api.openai.com/v1",
                                   max_retries=0, timeout=settings.llm_timeout_seconds,
                                   http_client=http) as sdk:
                budget = OneCall(sdk.responses.create)
                sdk.responses.create = budget
                adapter = OpenAIRagClient(settings, client=sdk)
                service = ChatService(retrieval, adapter, timeout_seconds=settings.llm_timeout_seconds)
                reply = await service.chat(query, conversation_context=conversation_context)
                evidence = []
                for source in reply.sources:
                    for eid in source.evidence_ids:
                        _, identity, field, index = eid.split(":")
                        raw = retrieval.catalog.profiles[int(identity)][field]
                        evidence.append({"evidence_id": eid, "text": raw[int(index)]
                                         if isinstance(raw, list) else raw})
                result = {"mode": reply.mode, "llm_status": reply.llm_status,
                          "cache_hit": reply.cache_hit, "provider_requests": reply.provider_requests,
                          "rendered_narrative": reply.answer,
                          "sources": [source.model_dump() for source in reply.sources],
                          "evidence": evidence,
                          "conversation_context": reply.conversation_context.model_dump()
                          if reply.conversation_context else None}
                if reply.llm_status != "used":
                    result["error_metadata"] = diagnostics.rows or [{"category": reply.llm_status,
                        "exception_type": None, "http_status": None, "provider_code": None}]
                return result
    finally:
        logger.removeHandler(diagnostics)
        logger.setLevel(previous[0])
        logger.disabled, logger.propagate = previous[1:]


def run_local(operation=None):
    # Load environment BEFORE importing embedding libraries and their cache settings.
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(LLM_MAX_ATTEMPTS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", ANONYMIZED_TELEMETRY="False")
    from dataclasses import replace
    from app.config import Settings
    from app.resources import parse_label_mapping, parse_knowledge_base
    from app.catalog import Catalog
    from app.retrieval import RetrievalService
    from app.retrieval_resources import load_retrieval
    settings = replace(Settings.from_env(), chat_enabled=True, embedding_local_only=True,
                       llm_api_key=os.getenv("LLM_API_KEY", "").strip())
    labels = parse_label_mapping(json.loads(
        (settings.model_dir / "label_mapping.json").read_text(encoding="utf-8")))
    catalog = Catalog(labels, parse_knowledge_base(json.loads(
        settings.knowledge_base_path.read_text(encoding="utf-8")), labels))
    resources = load_retrieval(settings, catalog)  # Opens only a temporary Chroma snapshot.
    try:
        return asyncio.run((operation or smoke)(settings, RetrievalService(catalog, resources.semantic,
                                                                          settings.search_threshold)))
    finally:
        resources.close()


def main():
    # Suppress third-party progress/debug output, not sanitized adapter diagnostics.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    root.handlers = [logging.NullHandler()]
    root.setLevel(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                result = run_local()
            except Exception:
                result = {"error_metadata": [{"category": "local_execution_error",
                           "exception_type": "OtherError", "http_status": None, "provider_code": None}]}
    finally:
        root.handlers = handlers
        root.setLevel(level)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("llm_status") == "used" else 1


if __name__ == "__main__":
    raise SystemExit(main())
