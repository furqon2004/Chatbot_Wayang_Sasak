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
from app.provider import ProviderClient
from evaluation.mock_search import mock_service


@pytest.fixture
def audit(settings, fake_resources, monkeypatch, tmp_path):
    from openai import AsyncOpenAI
    from test_openai_rag import response as sdk_response
    from test_web_chat import web_response_json
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
            data = web_response_json()
            if control.error == "invalid_response":
                data["output"] = []
        else:
            payload = json.loads(body["input"])
            selected = {}
            for row in payload["untrusted_evidence"]:
                selected.setdefault(row["id"], row["evidence_id"])
            data = sdk_response(json.dumps({"evidence_ids": list(selected.values())}))
        return httpx.Response(200, json=data)

    sdk = AsyncOpenAI(api_key="synthetic-test-value", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    client = ProviderClient(replace(settings, llm_max_attempts=2), client=sdk,
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
            next(w for w in at.text_input if w.label == "Pesan").input(query).run()
            next(b for b in at.button if b.label == "Kirim").click().run()
            assert not at.exception

        def record(flow, rag, web):
            print(json.dumps({"flow": flow, "rag_provider_requests": counts["rag"],
                              "web_provider_requests": counts["web"],
                              "chat_requests": counts["/chat"], "search_requests": counts["/search"]}))
            assert counts["rag"] == rag and counts["web"] == web

        yield NS(ui=at, submit=submit, record=record, counts=counts, control=control,
                 provider=client, monkeypatch=monkeypatch, catalog=service.catalog, backend=backend)


def test_pair_context_ui_followup_and_actual_provider_count(audit):
    retrieval = audit.backend.app.state.retrieval
    retrieval.semantic.scores = lambda *_: {0: .99, 2: .99}
    for i in (0, 2):
        retrieval.catalog.profiles[i].update(filosofi=f"Filosofi fixture {i}.", ciri_ciri=f"Ciri fixture {i}.")
    audit.submit("Bandingkan Jayangrana dan Maktal")
    data = audit.ui.session_state["messages"][-1]["data"]
    assert data["provider_requests"] == 1 and not data["cache_hit"]
    assert not data.get("character")
    assert audit.ui.session_state["conversation_context"]["subject_ids"] == [0, 2]
    assert set(audit.ui.session_state["seen_ids"]) == {0, 2}
    audit.submit("Apa filosofi keduanya?")
    data = audit.ui.session_state["messages"][-1]["data"]
    assert {s["id"] for s in data["sources"]} == {0, 2}
    assert all(":filosofi:" in e for s in data["sources"] for e in s["evidence_ids"])
    audit.submit("Apa filosofinya?")
    assert audit.ui.session_state["messages"][-1]["data"]["mode"] == "conversation"
    audit.record("ambiguous_pair", 2, 0)
    audit.submit("Bandingkan Jayangrana dan Maktal")
    data = audit.ui.session_state["messages"][-1]["data"]
    assert data["cache_hit"] and data["provider_requests"] == 0
    assert data["llm_used"] and data["llm_status"] == "used"
    audit.record("pair_cache_hit", 2, 0)


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
    next(b for b in audit.ui.button if b.label == "Jelajahi Katalog").click().run()
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
    audit.provider.search_breaker.failed(immediate=True)
    audit.submit("Sejarah Wayang Sasak")
    audit.record("cached_web_with_open_breaker", 0, 1)
    audit.submit("Daftar tokoh")
    audit.record("typed_catalog_after_cached_web", 0, 1)


def test_mixed_character_and_general_topic_uses_only_one_branch(audit):
    audit.submit("Jelaskan Jayangrana dan sejarah Wayang Sasak")
    audit.record("mixed_topics_single_branch", 1, 0)


def test_role_discovery_ui_sets_context_and_follows_philosophy(audit, tmp_path):
    retrieval = audit.backend.app.state.retrieval
    retrieval.catalog.profiles[0].update(
        deskripsi="Jayangrana merupakan tokoh utama dalam fixture Wayang Sasak.",
        filosofi="Filosofi raga dalam fixture.")
    root = tmp_path / "dataset_asli"
    image = root / "jayangrana" / "1.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (40, 60), "gold").save(image)
    audit.monkeypatch.setenv("WAYANG_IMAGE_ROOT", str(root))
    retrieval.catalog.items[0].gambar_utama = "dataset_asli/jayangrana/1.png"
    audit.submit("Apa itu Wayang Sasak?")
    overview = audit.ui.session_state["messages"][-1]["data"]
    assert overview["mode"] == "web_answer"
    assert overview["provider_requests"] == 1 and not overview["cache_hit"]
    source_url = overview["web_sources"][0]["url"]
    assert "[1]" in overview["answer"]
    assert any(f"[\\[1\\]]({source_url})" in item.value for item in audit.ui.markdown)
    assert any(item.proto.url == source_url for item in audit.ui.get("link_button"))
    assert audit.ui.session_state["conversation_context"]["domain"] == "wayang_sasak"
    audit.submit("Siapa tokoh utamanya?")
    reply = audit.ui.session_state["messages"][-1]["data"]
    assert reply["character"]["id"] == 0 and reply["provider_requests"] == 0
    assert sum(w.value == reply["opening"] for w in audit.ui.text) == 1
    assert len(audit.ui.get("image")) == 1
    assert audit.ui.session_state["conversation_context"]["subject_ids"] == [0]
    audit.record("role_discovery", 0, 1)
    audit.submit("Apa filosofinya?")
    reply = audit.ui.session_state["messages"][-1]["data"]
    assert reply["character"]["id"] == 0
    assert reply["sources"][0]["evidence_ids"] == ["profile:0:filosofi:0"]
    audit.record("role_philosophy_followup", 1, 1)


def test_overview_web_context_does_not_authorize_off_domain_ui_request(audit):
    audit.submit("Apa itu Wayang Sasak?")
    assert audit.ui.session_state["messages"][-1]["data"]["mode"] == "web_answer"
    audit.submit("Buatkan program Python untuk kalkulator")
    reply = audit.ui.session_state["messages"][-1]["data"]
    assert reply["mode"] == "fallback_catalog" and not reply["llm_used"]
    assert reply["provider_requests"] == 0
    audit.record("off_domain_after_web_overview", 0, 1)


def test_openai_narrative_ui_and_simplification_use_local_citations(audit):
    service = audit.backend.app.state.chat
    service.retrieval.catalog.profiles[0]["filosofi"] = "Jayangrana menjadi lambang raga manusia."
    async def narrate(query, rows, context):
        assert rows[0]["evidence_id"] == "profile:0:filosofi:0"
        text = "Jayangrana melambangkan tubuh manusia." if context["style"] == "simplify" else "Jayangrana melambangkan raga manusia."
        return json.dumps({"status": "supported", "sections": [{"text": text,
            "kind": "explanation", "subject_ids": [0], "evidence_ids": [rows[0]["evidence_id"]]}]})
    provider = NS(rag_provider="openai", rag_model="synthetic", narrate=AsyncMock(side_effect=narrate))
    service.provider = provider
    audit.submit("Apa filosofi Jayangrana?")
    reply = audit.ui.session_state["messages"][-1]["data"]
    assert reply["narrative"] and reply["answer"] == "Jayangrana melambangkan raga manusia. [1]"
    assert any("[1] wayang_db.json" in row.value for row in audit.ui.text)
    audit.submit("Jelaskan lebih sederhana")
    reply = audit.ui.session_state["messages"][-1]["data"]
    assert reply["answer"] == "Jayangrana melambangkan tubuh manusia. [1]"
    assert reply["conversation_context"]["subject_ids"] == [0]
    assert provider.narrate.await_count == 2
    audit.ui.run()
    assert not audit.ui.exception and provider.narrate.await_count == 2
