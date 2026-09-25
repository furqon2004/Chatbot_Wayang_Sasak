from app.chat import evidence_context
import asyncio

import pytest

from app.chat import ChatService
from evaluation.mock_search import mock_service
from scripts.colab_chat_smoke_test import _assert_sources


def test_smoke_source_verifier_accepts_mock_grounded_answer():
    service = mock_service()
    payload = asyncio.run(ChatService(service).chat("Jelaskan Jayangrana")).model_dump()
    _assert_sources(payload, evidence_context(service.search("Jelaskan Jayangrana").results, service.catalog))


@pytest.mark.parametrize("evidence_ids", [[], ["profile:999:deskripsi:0"]])
def test_smoke_source_verifier_rejects_empty_or_invented_sources(evidence_ids):
    service = mock_service()
    payload = asyncio.run(ChatService(service).chat("Jelaskan Jayangrana")).model_dump()
    payload["sources"][0]["evidence_ids"] = evidence_ids
    with pytest.raises(AssertionError):
        _assert_sources(payload, evidence_context(service.search("Jelaskan Jayangrana").results, service.catalog))


@pytest.mark.parametrize("status", ["invalid_request", "model_unavailable", "invalid_key", "insufficient_evidence"])
def test_in_scope_smoke_requires_live_gemini_and_reports_safe_category(status):
    from scripts.colab_chat_smoke_test import _assert_live
    with pytest.raises(AssertionError, match=status):
        _assert_live({"llm_used": False, "mode": "fallback_catalog", "llm_status": status})


def test_smoke_never_echoes_unknown_provider_status():
    from scripts.colab_chat_smoke_test import safe_status
    assert safe_status({"llm_status": "private provider detail"}) == "internal"


def test_smoke_checks_only_evidence_actually_sent():
    service = mock_service()
    payload = asyncio.run(ChatService(service).chat("Jelaskan Jayangrana")).model_dump()
    with pytest.raises(AssertionError, match="Source tidak dikenal"):
        _assert_sources(payload, [])


def test_smoke_records_context_without_printing(capsys):
    from scripts.colab_chat_smoke_test import RecordingGemini
    class Client:
        async def select(self, query, evidence):
            return '{"evidence_ids":[]}'
    wrapper = RecordingGemini(Client())
    evidence = [{"evidence_id": "profile:0:deskripsi:0", "text": "private fixture"}]
    assert asyncio.run(wrapper.select("private query", evidence)) == '{"evidence_ids":[]}'
    assert wrapper.evidence == evidence
    assert capsys.readouterr().out == ""
