"""Observational audit, not acceptance tests for conversational capabilities.

Real in-process FastAPI backend and Provider adapter; synthetic catalog,
semantic scores and SDK responses. Network is blocked by conftest.
"""
import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.provider import ProviderClient
from evaluation.mock_search import mock_service
from test_provider import VendorError


@pytest.mark.parametrize("flow", ["simplify", "philosophy", "switch", "compare", "insufficient", "failure"])
def test_observe_conversation(flow, settings, fake_resources):
    retrieval = mock_service()
    for i in (0, 2):
        name = retrieval.catalog.items[i].nama
        retrieval.catalog.characters[i].update(
            deskripsi=f"{name} adalah tokoh dalam fixture sintetis.",
            filosofi=f"Filosofi {name} adalah simbol sintetis.",
            ciri_ciri=f"Ciri {name} adalah busana sintetis.")
    fake_resources.characters = retrieval.catalog.characters
    retrieval.semantic.scores = lambda *_: {0: .99, 2: .99}
    state = NS(fail=False, empty=False, now=0)
    calls = []

    async def generate_content(**kwargs):
        payload = json.loads(kwargs["input"])
        calls.append(deepcopy(payload))
        assert kwargs["tools"] == []
        assert set(payload) == {"untrusted_query", "untrusted_evidence"}
        if state.fail:
            raise VendorError(400)
        rows = payload["untrusted_evidence"]
        selected = {}
        for row in rows:
            selected.setdefault(row["id"], row["evidence_id"])
        return NS(status="completed", output_text=json.dumps({"evidence_ids": [] if state.empty else list(selected.values())}))

    sdk = NS(responses=NS(create=generate_content), close=AsyncMock())
    provider = ProviderClient(settings, client=sdk, clock=lambda: state.now)
    bundle = NS(semantic=retrieval.semantic, close=lambda: None)
    app = create_app(settings, lambda _: fake_resources, lambda *_: bundle, lambda _: provider)
    trace = []
    with TestClient(app) as backend:
        service = app.state.chat
        service.retrieval.search = Mock(wraps=service.retrieval.search)

        def ask(message, **kwargs):
            before = len(calls)
            searches = service.retrieval.search.call_count
            response = backend.post("/chat", json={"message": message, **kwargs})
            assert response.status_code == 200
            reply = response.json()
            trace.append(dict(message=message, request=kwargs, mode=reply["mode"],
                status=reply["llm_status"], character=reply.get("character", {}).get("id"),
                answer=reply["answer"], sources=reply["sources"],
                provider_delta=len(calls)-before, provider_total=len(calls),
                cache_entries=len(service.rag_cache.entries),
                retrieval_queries=[c.args[0] for c in service.retrieval.search.call_args_list[searches:]],
                provider_context=calls[before:]))
            for source in reply["sources"]:
                for eid in source["evidence_ids"]:
                    _, identity, field, index = eid.split(":")
                    assert int(identity) == source["id"]
                    assert field in service.retrieval.catalog.profiles[int(identity)]
            return reply

        if flow == "compare":
            ask("Bandingkan Jayangrana dan Maktal")
            # Sparse context control: same request, one evidence per character.
            for i in (0, 2):
                service.retrieval.catalog.profiles[i]["filosofi"] = ""
                service.retrieval.catalog.profiles[i]["ciri_ciri"] = ""
            ask("Bandingkan Jayangrana dan Maktal")
            ask("Bandingkan Jayangrana dan Maktal")
        else:
            ask("Jelaskan Jayangrana")
            if flow == "simplify":
                ask("Jelaskan lebih sederhana", awaiting_more=True, seen_ids=[0])
                ask("Jelaskan Jayangrana lebih sederhana", character_id=0, explain_more=True)
            elif flow == "philosophy":
                ask("Apa filosofinya?", awaiting_more=True, seen_ids=[0])
                ask("Apa filosofi tokoh ini?", awaiting_more=True, seen_ids=[0])
                ask("Apa filosofi Jayangrana?")
            elif flow == "switch":
                ask("Jelaskan Maktal")
                ask("Apa filosofinya?", awaiting_more=True, seen_ids=[0, 2])
                ask("Apa filosofi tokoh ini?", character_id=0, explain_more=True)
            elif flow == "insufficient":
                ask("Apa hubungan Wayang Sasak dengan cerita Menak?")
                state.empty = True
                ask("Kapan Jayangrana lahir?")
            elif flow == "failure":
                state.fail = True
                ask("Jelaskan Jayangrana")
                ask("Apa filosofi Maktal?")
                ask("Apa filosofi Maktal?")
                ask("Kapan Jayangrana lahir?")
                state.fail = False
                ask("Apa filosofi Maktal?")
                state.now = 61
                ask("Apa filosofi Maktal?")
                ask("Apa filosofi Maktal?")
    print("AUDIT " + json.dumps({"flow": flow, "trace": trace}, ensure_ascii=True))
