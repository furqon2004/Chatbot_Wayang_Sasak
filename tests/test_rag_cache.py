import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from app.chat import ChatService
from app.chat_schemas import ChatRequest
from app.provider import ProviderFailure
from evaluation.mock_search import mock_service


def setup_service():
    retrieval = mock_service()
    retrieval.semantic.scores = lambda *_: {0: .99, 2: .99}
    async def select(query, context):
        return json.dumps({"evidence_ids": [context[0]["evidence_id"]]})
    provider = NS(select=AsyncMock(side_effect=select), settings=NS(llm_model="provider-test"))
    now = [0]
    return ChatService(retrieval, provider, clock=lambda: now[0]), provider, now


def ask(service, query="Jelaskan Jayangrana", character_id=0):
    return asyncio.run(service.chat(query, character_id=character_id, explain_more=True))


def test_rag_hit_expiry_and_copy_isolation():
    service, provider, now = setup_service()
    first = ask(service)
    first.answer = "mutation"
    assert ask(service, "  JELASKAN Jayangrana! ").answer != "mutation"
    assert provider.select.await_count == 1
    now[0] = 900
    ask(service)
    assert provider.select.await_count == 2


def test_cache_separates_evidence_identity_model_prompt_and_context(monkeypatch):
    service, provider, _ = setup_service()
    first = ask(service)
    service.retrieval.catalog.profiles[0]["deskripsi"] = "Deskripsi fixture diperbarui."
    changed = ask(service)
    assert changed.answer != first.answer and provider.select.await_count == 2
    # A different explicit subject must not return the first character.
    other = ask(service, "Jelaskan Maktal", character_id=2)
    assert other.character.id == 2 and provider.select.await_count == 3
    provider.settings.llm_model = "provider-other"
    ask(service)
    assert provider.select.await_count == 4
    monkeypatch.setattr("app.rag_cache.PROMPT_VERSION", "evidence-selection-v2")
    ask(service)
    assert provider.select.await_count == 5
    service.retrieval.catalog.profiles[0]["filosofi"] = "Evidence tambahan sintetis."
    ask(service)
    assert provider.select.await_count == 6


def test_cache_maximum_and_eviction():
    service, provider, _ = setup_service()
    for i in range(33):
        ask(service, f"Jelaskan Jayangrana bagian {i}")
    assert len(service.rag_cache.entries) == 32
    ask(service, "Jelaskan Jayangrana bagian 0")
    assert provider.select.await_count == 34


@pytest.mark.parametrize("query", ["Jelaskan Jayangrana email saya user@example.org",
    "Jelaskan Jayangrana NIK 1234567890123456", "Jelaskan Jayangrana nama saya Budi"])
def test_sensitive_queries_not_cached(query):
    service, provider, _ = setup_service()
    ask(service, query)
    ask(service, query)
    assert not service.rag_cache.entries


@pytest.mark.parametrize("result", ["not-json", '{"evidence_ids":[]}',
    '{"evidence_ids":["profile:19:deskripsi:0"]}', ProviderFailure("unavailable")])
def test_invalid_error_and_fallback_not_cached(result):
    service, provider, _ = setup_service()
    if isinstance(result, Exception):
        provider.select.side_effect = result
    else:
        provider.select.side_effect = None
        provider.select.return_value = result
    ask(service)
    ask(service)
    assert provider.select.await_count == 2 and not service.rag_cache.entries


def test_explanation_requires_identity_and_default_selection_stays_local():
    with pytest.raises(ValueError):
        ChatRequest(message="Jelaskan tokoh", explain_more=True)
    service, provider, _ = setup_service()
    reply = asyncio.run(service.chat("Jelaskan Jayangrana", character_id=0))
    assert reply.character.id == 0
    provider.select.assert_not_called()
