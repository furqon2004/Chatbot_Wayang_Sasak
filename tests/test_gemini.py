import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.gemini import GeminiClient, GeminiFailure, load_gemini


class VendorError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__("private vendor detail")


def fake_sdk(actions):
    calls = []
    async def generate_content(**kwargs):
        calls.append(kwargs)
        value = actions.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(text=value)
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))), calls


def test_transient_retry_is_bounded_with_backoff(settings):
    sdk, calls = fake_sdk([VendorError(503), VendorError(503), '{"evidence_ids":[]}'])
    sleeps = []
    async def sleep(delay):
        sleeps.append(delay)
    client = GeminiClient(replace(settings, gemini_max_attempts=3), client=sdk, sleep=sleep)
    assert asyncio.run(client.select("query", [])) == '{"evidence_ids":[]}'
    assert len(calls) == 3 and sleeps == [0.5, 1.0]
    config = calls[0]["config"]
    assert "HANYA" in config.system_instruction
    assert config.response_mime_type == "application/json"
    from app.chat_schemas import EvidenceSelection
    assert config.response_json_schema == EvidenceSelection.model_json_schema()
    assert config.response_schema is None
    assert config.tools == []
    assert config.automatic_function_calling.disable is True
    assert calls[0]["model"] == settings.gemini_model


@pytest.mark.parametrize("code,category", [(429, "rate_limit"), (401, "invalid_key"),
    (403, "invalid_key"), (400, "invalid_request"), (404, "model_unavailable"), (408, "timeout")])
def test_nonretryable_errors_do_not_retry(settings, code, category):
    sdk, calls = fake_sdk([VendorError(code)])
    client = GeminiClient(settings, client=sdk)
    with pytest.raises(GeminiFailure) as caught:
        asyncio.run(client.select("query", []))
    assert caught.value.category == category and len(calls) == 1
    assert "private" not in str(caught.value)


def test_retry_exhaustion(settings):
    sdk, calls = fake_sdk([VendorError(503), VendorError(503)])
    async def no_sleep(_):
        pass
    client = GeminiClient(settings, client=sdk, sleep=no_sleep)
    with pytest.raises(GeminiFailure, match="unavailable"):
        asyncio.run(client.select("query", []))
    assert len(calls) == 2


def test_missing_key_never_constructs_client(settings):
    assert load_gemini(settings) is None


def test_startup_initializes_without_network_and_disables_sdk_retries(settings, monkeypatch):
    from google import genai
    captured = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: captured.append(kwargs) or object())
    client = load_gemini(replace(settings, gemini_api_key="unit-test-sentinel"))
    assert client is not None and len(captured) == 1
    assert captured[0]["http_options"].timeout == 15000
    assert captured[0]["http_options"].retry_options.attempts == 1


def test_official_sdk_serializes_schema_over_mock_transport(settings):
    import httpx
    from google import genai
    from google.genai import types

    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": '{"evidence_ids":["profile:0:deskripsi:0"]}'}]
        }, "finishReason": "STOP"}]})

    async def run():
        sdk = genai.Client(api_key="unit-test-sentinel", http_options=types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
        adapter = GeminiClient(settings, client=sdk)
        try:
            return await adapter.select("Jelaskan Jayangrana", [{
                "evidence_id": "profile:0:deskripsi:0", "id": 0,
                "label": "jayangrana", "nama": "Jayangrana", "text": "Profil fixture",
            }])
        finally:
            await adapter.close()

    assert json.loads(asyncio.run(run()))["evidence_ids"] == ["profile:0:deskripsi:0"]
    assert len(requests) == 1
    assert requests[0]["generationConfig"]["responseMimeType"] == "application/json"
    config = requests[0]["generationConfig"]
    from app.chat_schemas import EvidenceSelection
    assert config["responseJsonSchema"] == EvidenceSelection.model_json_schema()
    assert config["responseJsonSchema"]["additionalProperties"] is False
    assert "responseSchema" not in config
    assert "thinkingConfig" not in config
    assert config["maxOutputTokens"] == 1024
    assert config["temperature"] == 0
    assert not requests[0].get("tools")
    assert "systemInstruction" in requests[0]


def test_official_invalid_key_reason_is_sanitized():
    from google.genai.errors import APIError
    from app.gemini import error_category
    error = APIError(400, {"error": {"message": "private", "details": [{"reason": "API_KEY_INVALID"}]}})
    assert error_category(error) == "invalid_key"


def test_adapter_uses_operator_model(settings):
    sdk, calls = fake_sdk(['{"evidence_ids":[]}'])
    client = GeminiClient(replace(settings, gemini_model="gemini-operator-model"), client=sdk)
    asyncio.run(client.select("query", []))
    assert calls[0]["model"] == "gemini-operator-model"
