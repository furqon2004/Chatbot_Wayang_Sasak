import asyncio

import pytest

from app.chat import ChatService
from evaluation.mock_search import mock_service
from scripts.colab_chat_smoke_test import _assert_sources


def test_smoke_source_verifier_accepts_mock_grounded_answer():
    service = mock_service()
    payload = asyncio.run(ChatService(service).chat("Jelaskan Jayangrana")).model_dump()
    _assert_sources(payload, service.search("Jelaskan Jayangrana"))


@pytest.mark.parametrize("evidence_ids", [[], ["profile:999:deskripsi:0"]])
def test_smoke_source_verifier_rejects_empty_or_invented_sources(evidence_ids):
    service = mock_service()
    payload = asyncio.run(ChatService(service).chat("Jelaskan Jayangrana")).model_dump()
    payload["sources"][0]["evidence_ids"] = evidence_ids
    with pytest.raises(AssertionError):
        _assert_sources(payload, service.search("Jelaskan Jayangrana"))
