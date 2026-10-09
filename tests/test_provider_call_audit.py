"""Count provider HTTP requests across UI -> HTTP -> FastAPI -> official SDK.

All transports are in memory; conftest forbids external network and isolates env.
Only flow labels and request counts are printed, never request/response contents.
"""
from collections import Counter
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from app.api import create_app
from app.gemini import GeminiClient
from evaluation.mock_search import mock_service


@pytest.fixture
def audit(settings, fake_resources, monkeypatch, tmp_path):
    from google import genai
    from google.genai import types
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    counts = Counter()
    control = NS(error=None, now=0)

    def provider(request):
        body = json.loads(request.content)
        web = bool(body.get("tools"))
        counts["web" if web else "rag"] += 1
        if control.error in {429, 503, 504}:
            return httpx.Response(control.error, json={"error": {"code": control.error, "message": "synthetic"}})
        if web:
            candidate = {"content": {"parts": [{"text": "Budaya Sasak."}]}, "finishReason": "STOP"}
            if control.error != "invalid_response":
                candidate["groundingMetadata"] = {
                    "webSearchQueries": ["synthetic"],
                    "groundingChunks": [{"web": {"title": "Budaya", "uri": "https://example.org/budaya"}}],
                    "groundingSupports": [{"segment": {"text": "Budaya Sasak."}, "groundingChunkIndices": [0]}]}
        else:
            payload = json.loads(body["contents"][0]["parts"][0]["text"])
            candidate = {"content": {"parts": [{"text": json.dumps({"evidence_ids": [
                payload["untrusted_evidence"][0]["evidence_id"]]})}]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [candidate]})

    sdk = genai.Client(api_key="synthetic-test-value", http_options=types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
        retry_options=types.HttpRetryOptions(attempts=1)))
    client = GeminiClient(replace(settings, gemini_max_attempts=2), client=sdk,
                          sleep=AsyncMock(), clock=lambda: control.now)
    service = mock_service()
    service.semantic.scores = lambda query, top_k: {0: .99}
    fake_resources.characters = service.catalog.characters
    bundle = NS(semantic=service.semantic, close=lambda: None)
    app = create_app(settings, lambda _: fake_resources, lambda *_: bundle, lambda _: client)
    real_http_client = httpx.Client
    with TestClient(app) as backend:
        def route(request):
            counts[request.url.path] += 1
            response = backend.request(request.method, request.url.path,
                                       content=request.content, headers=request.headers)
            return httpx.Response(response.status_code, content=response.content, headers=response.headers)

        monkeypatch.setattr("ui_client.httpx.Client", lambda **kwargs: real_http_client(
            **kwargs, transport=httpx.MockTransport(route)))
        at = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py").run()
        assert not at.exception

        def submit(query):
            next(w for w in at.text_input if w.label == "Pesan").input(query)
            next(b for b in at.button if b.label == "Kirim").click().run()
            assert not at.exception

        def record(flow, rag, web):
            print(json.dumps({"flow": flow, "rag_provider_requests": counts["rag"],
                              "web_provider_requests": counts["web"],
                              "chat_requests": counts["/chat"], "search_requests": counts["/search"]}))
            assert counts["rag"] == rag and counts["web"] == web

        yield NS(ui=at, submit=submit, record=record, counts=counts, control=control,
                 gemini=client, monkeypatch=monkeypatch, catalog=service.catalog)


def test_rag_submit_rerender_and_resubmit(audit):
    audit.submit("Siapa Jayangrana?")
    audit.record("rag_submit", 1, 0)
    for _ in range(3):
        audit.ui.run()
        assert not audit.ui.exception
    audit.record("rag_after_three_reruns", 1, 0)
    assert audit.counts["/chat"] == 1
    audit.submit("Siapa Jayangrana?")
    audit.record("rag_explicit_second_submit_cache_hit", 1, 0)


def test_web_submit_rerender_and_cache(audit):
    audit.submit("Sejarah Wayang Sasak")
    audit.record("web_submit", 0, 1)
    for _ in range(3):
        audit.ui.run()
    audit.record("web_after_three_reruns", 0, 1)
    assert audit.counts["/chat"] == 1
    audit.submit("Sejarah Wayang Sasak")
    audit.record("web_cache_hit_second_submit", 0, 1)
    assert audit.counts["/chat"] == 2


def test_catalog_yes_and_character_buttons(audit):
    next(b for b in audit.ui.button if b.label == "Lihat katalog").click().run()
    audit.record("catalog_button", 0, 0)
    next(b for b in audit.ui.button if b.label == "Jayangrana").click().run()
    audit.record("catalog_character_button", 0, 0)
    next(b for b in audit.ui.button if b.label == "Ya, kenalkan tokoh lain").click().run()
    audit.record("yes_button", 0, 0)
    choices = audit.ui.session_state["messages"][-1]["data"]["choices"]
    assert len(choices) == 3
    next(b for b in audit.ui.button if b.label == choices[0]["nama"]).click().run()
    audit.record("suggested_character_button", 0, 0)
    audit.ui.run()
    assert not audit.ui.exception
    audit.record("buttons_after_rerun", 0, 0)


@pytest.mark.parametrize("error,expected", [(429, 1), (503, 2), (504, 2), ("invalid_response", 2)])
def test_web_failures_from_ui(audit, error, expected):
    audit.control.error = error
    audit.submit("Sejarah Wayang Sasak")
    audit.record("web_failure_" + str(error), 0, expected)
    assert audit.ui.session_state["messages"][-1]["data"]["mode"] == "fallback_catalog"
    audit.ui.run()
    audit.record("failure_rerun_" + str(error), 0, expected)
    if error == 429:
        audit.submit("Apa itu Wayang Sasak?")
        audit.record("open_breaker_submit", 0, 1)


@pytest.mark.parametrize("accepted", [True, False])
def test_image_submit_narration_call(audit, accepted):
    # Inference is mocked; the UI's additional /chat uses the full real code path.
    identity = {"id": 0, "label": "jayangrana", "nama": "Jayangrana", "confidence": .95, "margin": .8}
    prediction = {"status": "accepted" if accepted else "uncertain",
        "confidence": .95 if accepted else .4, "margin": .8 if accepted else .1,
        "top_3": [{"id": i, "label": "synthetic", "nama": "Synthetic", "probability": .3} for i in range(3)],
        "top_3_status": "candidates", "prediksi": identity if accepted else None,
        "profil": audit.catalog.profiles[0] if accepted else None,
        "message": None if accepted else "Belum pasti.",
        "decision": {"accepted": accepted, "reasons": [] if accepted else ["low_confidence"]}}
    audit.monkeypatch.setattr("app.api.predict_image", lambda *args, **kwargs: prediction)
    upload = io.BytesIO()
    Image.new("RGB", (40, 60), "gold").save(upload, format="PNG")
    upload.name, upload.size = "synthetic.png", len(upload.getvalue())
    audit.monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: upload)
    audit.ui.run()
    next(b for b in audit.ui.button if b.label == "Kirim").click().run()
    assert not audit.ui.exception
    audit.record("image_accepted" if accepted else "image_uncertain", 0, 0)
    assert audit.counts["/predict-image"] == 1
    audit.ui.run()
    audit.record("image_after_rerun_" + str(accepted), 0, 0)
    assert audit.counts["/chat"] == 0
    if accepted:
        next(b for b in audit.ui.button if b.label == "Jelaskan lebih lanjut").click().run()
        assert not audit.ui.exception
        audit.record("explicit_image_explanation", 1, 0)
        next(b for b in audit.ui.button if b.label == "Jelaskan lebih lanjut").click().run()
        assert not audit.ui.exception
        audit.record("explicit_image_explanation_cached", 1, 0)
    else:
        assert not any(b.label == "Jelaskan lebih lanjut" for b in audit.ui.button)



@pytest.mark.parametrize("error,expected", [(429, 1), (503, 2), (504, 2)])
def test_rag_retry_is_single_layer(audit, error, expected):
    audit.control.error = error
    audit.submit("Siapa Jayangrana?")
    audit.record("rag_failure_" + str(error), expected, 0)


def test_cache_precedes_breaker_and_catalog_text_is_local(audit):
    audit.submit("Sejarah Wayang Sasak")
    audit.gemini.search_breaker.failed(immediate=True)
    audit.submit("Sejarah Wayang Sasak")
    audit.record("cached_web_with_open_breaker", 0, 1)
    audit.submit("Daftar tokoh")
    audit.record("typed_catalog_after_cached_web", 0, 1)


def test_mixed_character_and_general_topic_uses_only_one_branch(audit):
    audit.submit("Jelaskan Jayangrana dan sejarah Wayang Sasak")
    audit.record("mixed_topics_single_branch", 1, 0)
