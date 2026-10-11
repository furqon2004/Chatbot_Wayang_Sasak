import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest

from evaluation.mock_search import mock_service
from scripts.local_narrative_smoke import OneCall, main, smoke
from test_openai_rag import response


@pytest.mark.parametrize("outcome", ["success", "connection", "server", "invalid", "insufficient"])
def test_smoke_one_sdk_call_and_safe_output(settings, monkeypatch, outcome):
    sentinel = "private-smoke-sentinel"
    retrieval = mock_service()
    retrieval.semantic.scores = lambda *_: {0: .99}
    retrieval.catalog.profiles[0]["filosofi"] = "Jayangrana merupakan perlambangan raga manusia."
    calls = []
    def handle(request):
        calls.append(request)
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-4.1-mini" and payload["tools"] == []
        assert payload["text"]["format"]["name"] == "evidence_narrative"
        if outcome == "connection":
            raise httpx.ConnectError(sentinel, request=request)
        if outcome == "server":
            return httpx.Response(503, json={"error": {"message": sentinel, "code": "server_error"}})
        raw = json.dumps({"status": "supported", "sections": [{
            "text": "Jayangrana melambangkan raga manusia.", "kind": "explanation",
            "subject_ids": [0], "evidence_ids": ["profile:0:filosofi:0"]}]})
        if outcome == "invalid":
            raw = sentinel
        elif outcome == "insufficient":
            raw = '{"status":"insufficient_evidence","sections":[]}'
        return httpx.Response(200, json=response(raw))
    original = httpx.AsyncClient
    options = []
    class Client(original):
        def __init__(self, **kwargs):
            options.append(kwargs)
            super().__init__(**kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setenv("HTTPS_PROXY", "http://invalid.invalid:9")
    result = asyncio.run(smoke(replace(settings, llm_api_key=sentinel), retrieval,
                               transport=httpx.MockTransport(handle)))
    assert len(calls) == 1 and result["provider_requests"] == 1 and not result["cache_hit"]
    assert options[0]["trust_env"] is False and options[0]["verify"] is True
    assert sentinel not in json.dumps(result)
    if outcome == "success":
        assert result["rendered_narrative"] == "Jayangrana melambangkan raga manusia. [1]"
        assert result["evidence"][0]["text"] == retrieval.catalog.profiles[0]["filosofi"]
        assert result["sources"][0]["document"] == "wayang_db.json"
    else:
        assert result["mode"] == "fallback_catalog" and result["error_metadata"]
    if outcome == "server":
        assert result["error_metadata"][0]["http_status"] == 503
        assert result["error_metadata"][0]["provider_code"] == "server_error"


def test_hard_sdk_budget():
    create = AsyncMock(return_value=None)
    budget = OneCall(create)
    async def run():
        await budget()
        with pytest.raises(RuntimeError):
            await budget()
    asyncio.run(run())
    create.assert_awaited_once()


def test_cli_does_not_print_raw_exceptions_or_library_output(monkeypatch, capsys):
    def fail():
        print("private-output")
        raise RuntimeError("private-exception")
    monkeypatch.setattr("scripts.local_narrative_smoke.run_local", fail)
    assert main() == 1
    captured = capsys.readouterr()
    assert "private" not in captured.out + captured.err
    assert json.loads(captured.out)["error_metadata"][0]["category"] == "local_execution_error"
