"""Both routes in the same application; no network or real configuration."""
import json
from dataclasses import replace
from types import SimpleNamespace as NS

import httpx
from fastapi.testclient import TestClient

from evaluation.mock_search import mock_service
from scripts.local_chat import build_app
from test_openai_rag import response
from test_web_chat import web_response_json


def test_rejected_narrative_does_not_stop_web_or_catalog(settings, fake_resources):
    retrieval = mock_service()
    retrieval.semantic.scores = lambda *_: {0: .99}
    fake_resources.characters = retrieval.catalog.characters
    calls = []
    def handle(request):
        body = json.loads(request.content)
        web = bool(body.get("tools"))
        calls.append("web" if web else "rag")
        return httpx.Response(200, json=web_response_json() if web else response("invalid fixture"))
    app = build_app(replace(settings, llm_api_key="synthetic-only"), direct=True,
        transport=httpx.MockTransport(handle), resource_loader=lambda _: fake_resources,
        retrieval_loader=lambda *_: NS(semantic=retrieval.semantic, close=lambda: None))
    with TestClient(app) as client:
        local = client.post("/chat", json={"message": "jelaskan saya tentang tokoh jayangrana"}).json()
        assert local["mode"] == "answer" and local["character"]["id"] == 0
        assert local["llm_status"] == "invalid_response" and not local["llm_used"]
        assert "Berikut informasi profil lokal" in local["answer"]
        assert not local.get("catalog")
        for query in ["apa itu wayang sasak", "Sejarah Wayang Sasak", "Serat Menak itu apa"]:
            web = client.post("/chat", json={"message": query}).json()
            assert web["mode"] == "web_answer" and web["web_sources"]
            assert web["provider_requests"] == 1
        assert calls == ["rag", "web", "web", "web"]
        blocked = client.post("/chat", json={"message": "Buat program Python"}).json()
        assert blocked["provider_requests"] == 0
        assert len(calls) == 4


def test_web_failure_does_not_stop_local_narration(settings, fake_resources):
    retrieval = mock_service()
    retrieval.semantic.scores = lambda *_: {0: .99}
    retrieval.catalog.characters[0]["filosofi"] = "Jayangrana melambangkan raga manusia dalam fixture."
    fake_resources.characters = retrieval.catalog.characters
    calls = []
    def handle(request):
        body = json.loads(request.content)
        if body.get("tools"):
            calls.append("web")
            return httpx.Response(503, json={"error": {"code": "server_error"}})
        calls.append("rag")
        rows = json.loads(body["input"])["evidence"]["untrusted_evidence"]
        raw = json.dumps({"status": "supported", "sections": [
            {"text": row["text"], "kind": "explanation", "subject_ids": [row["id"]],
             "evidence_ids": [row["evidence_id"]]} for row in rows]})
        return httpx.Response(200, json=response(raw))
    app = build_app(replace(settings, llm_api_key="synthetic-only"), direct=True,
        transport=httpx.MockTransport(handle), resource_loader=lambda _: fake_resources,
        retrieval_loader=lambda *_: NS(semantic=retrieval.semantic, close=lambda: None))
    with TestClient(app) as client:
        web = client.post("/chat", json={"message": "apa itu wayang sasak"}).json()
        assert web["mode"] == "fallback_catalog" and not web["llm_used"]
        local = client.post("/chat", json={"message": "Apa filosofi Jayangrana?"}).json()
        assert local["mode"] == "answer" and local["llm_used"] and local["sources"]
        assert calls == ["web", "rag"]
