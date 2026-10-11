import asyncio
import json
from dataclasses import replace
from unittest.mock import patch

import httpx
import pytest

from scripts.local_narrative_followup_smoke import followups, ui_replay
from test_narrative import setup, ask, encoded
from test_openai_rag import response
from test_ui import app_test


def previous(service):
    reply = ask(service, "Apa filosofi Jayangrana?")
    return {"mode": "answer", "llm_status": "used", "sources": [s.model_dump() for s in reply.sources],
            "evidence": [{"evidence_id": "profile:0:filosofi:0",
                          "text": service.retrieval.catalog.profiles[0]["filosofi"]}]}


@pytest.mark.parametrize("fail", [False, True])
def test_followup_smoke_budget_context_and_stop(settings, fail, monkeypatch, tmp_path):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    service, _, _ = setup()
    saved = previous(service)
    saved["rendered_narrative"] = "Jayangrana melambangkan raga manusia. [1]"
    calls = []
    def handle(request):
        payload = json.loads(json.loads(request.content)["input"])
        calls.append(payload)
        if fail:
            return httpx.Response(503, json={"error": {"code": "server_error"}})
        rows = payload["evidence"]["untrusted_evidence"]
        return httpx.Response(200, json=response(encoded(*[
            {"text": r["text"], "kind": "explanation", "subject_ids": [r["id"]],
             "evidence_ids": [r["evidence_id"]]} for r in rows])))
    result = asyncio.run(followups(replace(settings, llm_api_key="synthetic-only"),
        service.retrieval, saved, transport=httpx.MockTransport(handle)))
    assert len(calls) == (1 if fail else 2)
    assert result["provider_requests"] == len(calls)
    assert result["completed"] is not fail
    assert calls[0]["context"]["style"] == "simplify"
    assert calls[0]["context"]["previous_evidence"] == saved["evidence"]
    if not fail:
        assert {r["id"] for r in calls[1]["evidence"]["untrusted_evidence"]} == {0, 2}
        assert ui_replay(saved, result["steps"])["additional_provider_requests"] == 0


def test_changed_previous_evidence_stops_before_sdk(settings):
    service, _, _ = setup()
    saved = previous(service)
    saved["evidence"][0]["text"] = "changed"
    with pytest.raises(ValueError):
        asyncio.run(followups(settings, service.retrieval, saved))


def test_streamlit_context_and_numbered_sources_reach_display(monkeypatch, tmp_path):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    service, provider, _ = setup()
    requests = []
    def api(path, *, payload=None, **kwargs):
        if path == "/health":
            return {"status": "ok"}
        assert path == "/chat"
        requests.append(payload)
        fields = dict(payload)
        query = fields.pop("message")
        return ask(service, query, **fields).model_dump()
    with patch("ui_client.request_api", side_effect=api):
        at = app_test().run()
        for query in ["Apa filosofi Jayangrana?", "Jelaskan lebih sederhana",
                      "Bandingkan filosofi Jayangrana dan Maktal"]:
            next(w for w in at.text_input if w.label == "Pesan").input(query)
            next(b for b in at.button if b.label == "Kirim").click().run()
            assert not at.exception
        assert requests[1]["conversation_context"]["evidence_ids"] == ["profile:0:filosofi:0"]
        assert at.session_state["conversation_context"]["subject_ids"] == [0, 2]
        texts = [t.value for t in at.text]
        assert any("[1] wayang_db.json" in t and "jayangrana" in t for t in texts)
        assert any("[2] wayang_db.json" in t and "maktal" in t for t in texts)
        assert any("Maktal" in t and "[2]" in t for t in texts)
        assert provider.narrate.await_count == 3
