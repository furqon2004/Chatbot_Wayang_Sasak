import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from app.chat import ChatService
from app.config import Settings
from app.provider import ProviderClient, ProviderFailure, REQUEST_METER
from app.llm import LLMRouter, load_llm
from app.openai_rag import OpenAIRagClient
from evaluation.mock_search import mock_service

SECRET = "synthetic-credential-sentinel"
RAW = "synthetic-private-response"
EVIDENCE = [{"evidence_id": "profile:0:deskripsi:0", "id": 0, "label": "jayangrana",
             "nama": "Jayangrana", "text": "Deskripsi fixture."}]


def config(settings):
    return replace(settings, chat_enabled=True, llm_provider="openai", llm_api_key=SECRET)


def response(raw=None, status="completed"):
    return {"id": "resp_synthetic", "object": "response", "created_at": 0,
        "model": "gpt-4.1-mini", "status": status,
        "output": [{"type": "message", "id": "msg_synthetic", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": raw or json.dumps({
                        "evidence_ids": ["profile:0:deskripsi:0"],
                        "follow_up": "Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?"}), "annotations": []}]}]}


def adapter(settings, handler):
    sdk = AsyncOpenAI(api_key=SECRET, max_retries=0,
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return OpenAIRagClient(config(settings), client=sdk, sleep=AsyncMock())


def test_default_openai_clients_ignore_system_proxy(settings):
    async def run():
        clients = [OpenAIRagClient(config(settings)), ProviderClient(config(settings))]
        try:
            assert all(client.client._client._trust_env is False for client in clients)
        finally:
            await asyncio.gather(*(client.close() for client in clients))
    asyncio.run(run())


def test_sdk_contract_and_success(settings):
    calls = []
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.headers["authorization"] == "Bearer " + SECRET
        return httpx.Response(200, json=response())
    async def run():
        client = adapter(settings, handle)
        try:
            result = await client.select("Jelaskan tokoh", EVIDENCE)
            assert json.loads(result)["evidence_ids"] == [EVIDENCE[0]["evidence_id"]]
        finally:
            await client.close()
    asyncio.run(run())
    assert len(calls) == 1
    body = calls[0]
    assert body["model"] == "gpt-4.1-mini" and body["tools"] == [] and body["store"] is False
    schema = body["text"]["format"]["schema"]
    assert body["text"]["format"]["strict"] and not schema["additionalProperties"]
    assert set(schema["required"]) == set(schema["properties"])


def test_narrative_sdk_contract_has_strict_nested_schema(settings):
    captured = []
    raw = json.dumps({"status": "supported", "sections": [{
        "text": "Deskripsi fixture.", "kind": "explanation", "subject_ids": [0],
        "evidence_ids": [EVIDENCE[0]["evidence_id"]]}]})
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response(raw))
    async def run():
        client = adapter(settings, handle)
        try:
            assert await client.narrate("Jelaskan lebih sederhana", EVIDENCE,
                {"style": "simplify", "subject_ids": [0]}) == raw
        finally:
            await client.close()
    asyncio.run(run())
    body = captured[0]
    assert body["model"] == "gpt-4.1-mini" and body["tools"] == [] and not body["store"]
    contract = body["text"]["format"]
    assert contract["strict"] and contract["name"] == "evidence_narrative"
    schema = contract["schema"]
    for node in [schema, *schema["$defs"].values()]:
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
    assert json.loads(body["input"])["context"]["style"] == "simplify"


@pytest.mark.parametrize("code,expected,category", [(429, 1, "rate_limit"), (503, 2, "service_unavailable"),
    (504, 2, "service_unavailable"), (401, 1, "invalid_key")])
def test_errors_single_retry_layer_meter_and_safe_logs(settings, code, expected, category, caplog):
    calls = []
    def handle(request):
        calls.append(1)
        return httpx.Response(code, json={"error": {"message": RAW + SECRET, "type": "synthetic", "code": "synthetic"}})
    async def run():
        client = adapter(settings, handle)
        meter = {"count": 0}
        token = REQUEST_METER.set(meter)
        try:
            with pytest.raises(ProviderFailure, match=category):
                await client.select("Jelaskan tokoh", EVIDENCE)
            if code == 429:
                assert not client.breaker.available()
                with pytest.raises(ProviderFailure, match="unavailable"):
                    await client.select("Jelaskan tokoh", EVIDENCE)
            assert meter["count"] == expected
        finally:
            REQUEST_METER.reset(token)
            await client.close()
    asyncio.run(run())
    assert len(calls) == expected
    assert SECRET not in caplog.text and RAW not in caplog.text


@pytest.mark.parametrize("raw,status", [(RAW, "completed"),
    ('{"evidence_ids":["profile:19:deskripsi:0"]}', "completed"),
    ('{"evidence_ids":["profile:0:deskripsi:0","profile:0:deskripsi:0"]}', "completed"),
    ('{"evidence_ids":[]}', "incomplete")])
def test_invalid_response_rejected_without_logging_body(settings, raw, status, caplog):
    async def run():
        client = adapter(settings, lambda _: httpx.Response(200, json=response(raw, status)))
        try:
            with pytest.raises(ProviderFailure, match="invalid_response"):
                await client.select("Jelaskan tokoh", EVIDENCE)
        finally:
            await client.close()
    asyncio.run(run())
    assert RAW not in caplog.text and SECRET not in caplog.text


def test_timeout_once(settings):
    calls = []
    def handle(request):
        calls.append(1)
        raise httpx.ReadTimeout(RAW, request=request)
    async def run():
        client = adapter(settings, handle)
        try:
            with pytest.raises(ProviderFailure, match="timeout"):
                await client.select("Jelaskan tokoh", EVIDENCE)
        finally:
            await client.close()
    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("provider_code,expected", [
    ("insufficient_quota", "insufficient_quota"),
    (SECRET + "\nproject-private", "None"),
    ({"secret": SECRET}, "None"),
    (None, "None")])
def test_error_metadata_is_allowlisted_and_private(settings, caplog, provider_code, expected):
    def handle(request):
        return httpx.Response(429, headers={"x-request-id": "private-request-id"}, json={
            "error": {"message": RAW + SECRET, "code": provider_code,
                      "type": "project-private", "param": "private-prompt"}})
    async def run():
        client = adapter(settings, handle)
        try:
            with pytest.raises(ProviderFailure, match="rate_limit"):
                await client.narrate("private-prompt", EVIDENCE, {})
        finally:
            await client.close()
    asyncio.run(run())
    records = [r for r in caplog.records if r.name == "app.openai_rag"]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() == ("RAG request failed: provider=openai category=rate_limit "
        f"exception_type=RateLimitError http_status=429 provider_code={expected}")
    assert record.exc_info is None and record.stack_info is None
    for private in (SECRET, RAW, "project-private", "private-prompt", "private-request-id"):
        assert private not in caplog.text


def test_connection_error_metadata_does_not_invent_http_status(settings, caplog):
    def handle(request):
        raise httpx.ConnectError(SECRET + RAW, request=request)
    async def run():
        client = adapter(replace(settings, llm_max_attempts=1), handle)
        try:
            with pytest.raises(ProviderFailure, match="unavailable"):
                await client.narrate("private-prompt", EVIDENCE, {})
        finally:
            await client.close()
    asyncio.run(run())
    assert "exception_type=APIConnectionError http_status=None provider_code=None" in caplog.text
    assert SECRET not in caplog.text and RAW not in caplog.text


def test_chat_cache_context_and_provider_separation(settings):
    calls = []
    def handle(request):
        calls.append(1)
        body = json.loads(request.content)
        payload = json.loads(body["input"])
        rows = payload.get("evidence", payload)["untrusted_evidence"]
        raw = ({"status": "supported", "sections": [{"text": rows[0]["text"], "kind": "explanation",
            "subject_ids": [rows[0]["id"]], "evidence_ids": [rows[0]["evidence_id"]]}]}
            if body["text"]["format"]["name"] == "evidence_narrative" else {"evidence_ids": [rows[0]["evidence_id"]]})
        return httpx.Response(200, json=response(json.dumps(raw)))
    async def run():
        client = adapter(settings, handle)
        web = NS(search=AsyncMock(side_effect=ProviderFailure("rate_limit")), close=AsyncMock())
        router = LLMRouter(config(settings), client, web)
        retrieval = mock_service()
        retrieval.semantic.scores = lambda *_: {0: .99}
        retrieval.catalog.profiles[0]["filosofi"] = "Filosofi fixture."
        service = ChatService(retrieval, router)
        try:
            first = await service.chat("Jelaskan Jayangrana")
            assert first.mode == "answer" and first.provider_requests == 1
            cached = await service.chat("Jelaskan Jayangrana")
            assert cached.cache_hit and cached.provider_requests == 0 and len(calls) == 1
            follow = await service.chat("Apa filosofinya?", conversation_context=first.conversation_context)
            assert follow.sources[0].evidence_ids == ["profile:0:filosofi:0"]
            assert follow.provider_requests == 1
            router.rag_provider = "provider"
            await service.chat("Jelaskan Jayangrana")
            assert len(calls) == 3
            router.rag_model = "another-model"
            await service.chat("Jelaskan Jayangrana")
            assert len(calls) == 4
            web.search.assert_not_called()
            await service.chat("Sejarah Wayang Sasak")
            web.search.assert_awaited_once()
            assert len(calls) == 4
        finally:
            await router.close()
    asyncio.run(run())


def test_loader_uses_single_credential_and_disables_sdk_retry(settings, monkeypatch):
    import openai
    captured = []
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: captured.append(kw) or object())
    configured = config(settings)
    routed = load_llm(configured)
    assert routed.rag_provider == "openai" and routed.rag_model == "gpt-4.1-mini"
    assert len(captured) == 2
    assert all(c["api_key"] == SECRET and c["max_retries"] == 0 for c in captured)
    assert SECRET not in repr(configured)


def test_missing_key_disables_rag_and_search(settings):
    router = load_llm(replace(config(settings), llm_api_key=""))
    async def run():
        with pytest.raises(ProviderFailure, match="not_configured"):
            await router.select("query", EVIDENCE)
        with pytest.raises(ProviderFailure, match="not_configured"):
            await router.search("query")
    asyncio.run(run())


def test_config_reads_only_shared_credential(monkeypatch):
    monkeypatch.setenv("WAYANG_ENABLE_CHAT", "true")
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    configured = Settings.from_env()
    assert configured.llm_api_key == SECRET
    assert configured.llm_provider == "openai" and configured.llm_model == "gpt-4.1-mini"
    monkeypatch.setenv("WAYANG_ENABLE_CHAT", "false")
    assert not Settings.from_env().llm_api_key


@pytest.mark.parametrize("raw", [RAW, '{"evidence_ids":[]}', '{"evidence_ids":["profile:19:deskripsi:0"]}'])
def test_chat_rejects_bad_output_and_does_not_cache_fallback(settings, raw):
    calls = []
    def handle(request):
        calls.append(1)
        return httpx.Response(200, json=response(raw))
    async def run():
        client = adapter(settings, handle)
        retrieval = mock_service()
        retrieval.semantic.scores = lambda *_: {0: .99}
        service = ChatService(retrieval, LLMRouter(config(settings), client, None))
        try:
            for _ in range(2):
                reply = await service.chat("Jelaskan Jayangrana")
                assert reply.mode == "answer" and not reply.cache_hit
                assert reply.character.id == 0 and not reply.catalog
                assert reply.llm_status == "invalid_response" and not reply.llm_used
                assert "Berikut informasi profil lokal" in reply.answer
                assert reply.provider_requests == 1 and not reply.llm_used
            assert not service.rag_cache.entries
        finally:
            await client.close()
    asyncio.run(run())
    assert len(calls) == 2


def test_fastapi_default_loader_routes_openai_and_health_is_not_search_success(settings, fake_resources, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import create_app
    captured = []
    def handle(request):
        captured.append(request.url.host)
        body = json.loads(request.content)
        if body.get("tools"):
            from test_web_chat import web_response_json
            return httpx.Response(200, json=web_response_json())
        payload = json.loads(body["input"])
        rows = payload.get("evidence", payload)["untrusted_evidence"]
        raw = ({"status": "supported", "sections": [{"text": rows[0]["text"], "kind": "explanation",
            "subject_ids": [rows[0]["id"]], "evidence_ids": [rows[0]["evidence_id"]]}]}
            if body["text"]["format"]["name"] == "evidence_narrative" else {"evidence_ids": [rows[0]["evidence_id"]]})
        return httpx.Response(200, json=response(json.dumps(raw)))
    def fake_client(configured):
        return AsyncOpenAI(api_key=configured.llm_api_key, base_url="https://api.openai.com/v1",
            max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr("app.provider.create_openai_client", fake_client)
    monkeypatch.setattr("app.openai_rag.create_openai_client", fake_client)
    retrieval = mock_service()
    retrieval.semantic.scores = lambda *_: {0: .99}
    fake_resources.characters = retrieval.catalog.characters
    app = create_app(config(settings), lambda _: fake_resources,
        lambda *_: NS(semantic=retrieval.semantic, close=lambda: None))
    with TestClient(app) as http:
        assert not captured
        assert http.get("/health").json()["provider_client_ready"] is True
        first = http.post("/chat", json={"message": "Jelaskan Jayangrana"}).json()
        assert first["mode"] == "answer" and first["provider_requests"] == 1
        second = http.post("/chat", json={"message": "Jelaskan Jayangrana"}).json()
        assert second["cache_hit"] and second["provider_requests"] == 0
        web = http.post("/chat", json={"message": "Sejarah Wayang Sasak"}).json()
        assert web["mode"] == "web_answer" and web["provider_requests"] == 1
    assert captured == ["api.openai.com", "api.openai.com"]
