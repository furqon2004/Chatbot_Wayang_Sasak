import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.provider import ProviderClient, ProviderFailure, load_provider


def VendorError(code):
    import httpx
    from openai import APIStatusError
    return APIStatusError("private vendor detail", response=httpx.Response(code,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses")), body=None)


def fake_sdk(actions):
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        value = actions.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(output_text=value, status="completed")
    return SimpleNamespace(responses=SimpleNamespace(create=create)), calls


def test_transient_retry_is_bounded_with_backoff(settings):
    sdk, calls = fake_sdk([VendorError(503), '{"evidence_ids":[]}'])
    sleeps = []
    async def sleep(delay):
        sleeps.append(delay)
    client = ProviderClient(replace(settings, llm_max_attempts=2), client=sdk, sleep=sleep)
    assert asyncio.run(client.select("query", [])) == '{"evidence_ids":[]}'
    assert len(calls) == 2 and sleeps == [0.5]
    config = calls[0]
    assert "HANYA" in config["instructions"]
    assert config["text"]["format"]["strict"]
    assert config["tools"] == []
    assert calls[0]["model"] == settings.llm_model



@pytest.mark.parametrize("code,category", [(429, "rate_limit"), (401, "invalid_key"),
    (403, "invalid_key"), (400, "invalid_request"), (404, "model_unavailable"), (408, "timeout")])
def test_nonretryable_errors_do_not_retry(settings, code, category):
    sdk, calls = fake_sdk([VendorError(code)])
    client = ProviderClient(settings, client=sdk)
    with pytest.raises(ProviderFailure) as caught:
        asyncio.run(client.select("query", []))
    assert caught.value.category == category and len(calls) == 1
    assert "private" not in str(caught.value)


def test_retry_exhaustion(settings):
    sdk, calls = fake_sdk([VendorError(503), VendorError(503)])
    async def no_sleep(_):
        pass
    client = ProviderClient(settings, client=sdk, sleep=no_sleep)
    with pytest.raises(ProviderFailure, match="unavailable"):
        asyncio.run(client.select("query", []))
    assert len(calls) == 2


def test_missing_key_never_constructs_client(settings):
    assert load_provider(settings) is None


def test_startup_initializes_without_network_and_disables_sdk_retries(settings, monkeypatch):
    import openai
    captured = []
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: captured.append(kwargs) or object())
    client = load_provider(replace(settings, llm_api_key="unit-test-sentinel"))
    assert client is not None and len(captured) == 1
    assert captured[0]["timeout"] == 45
    assert captured[0]["max_retries"] == 0


def test_adapter_uses_operator_model(settings):
    sdk, calls = fake_sdk(['{"evidence_ids":[]}'])
    client = ProviderClient(replace(settings, llm_model="provider-operator-model"), client=sdk)
    asyncio.run(client.select("query", []))
    assert calls[0]["model"] == "provider-operator-model"
