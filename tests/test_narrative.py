"""Synthetic narrative evaluation: provenance is not semantic entailment."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from app.chat import ChatService
from app.provider import ProviderFailure
from app.narrative import render_narrative
from evaluation.mock_search import mock_service


def section(text, identity=0, field="filosofi", kind="explanation"):
    return {"text": text, "kind": kind, "subject_ids": [identity],
            "evidence_ids": [f"profile:{identity}:{field}:0"]}


def encoded(*sections):
    return json.dumps({"status": "supported", "sections": list(sections)})


def setup():
    retrieval = mock_service()
    retrieval.semantic.scores = lambda *_: {0: .99, 2: .99}
    retrieval.catalog.profiles[0].update(filosofi="Dalam fixture, Jayangrana menjadi lambang raga manusia.",
        ciri_ciri="Jayangrana memakai keris dalam fixture.")
    retrieval.catalog.profiles[2].update(filosofi="Dalam fixture, Maktal melambangkan keberanian.")
    async def narrate(query, rows, context):
        parts = [section(row["text"], row["id"], row["evidence_id"].split(":")[2]) for row in rows]
        return encoded(*parts)
    provider = NS(rag_provider="openai", rag_model="synthetic", narrate=AsyncMock(side_effect=narrate),
                  select=AsyncMock(), search=AsyncMock())
    now = [0]
    return ChatService(retrieval, provider, clock=lambda: now[0]), provider, now


def ask(service, query, **kwargs):
    return asyncio.run(service.chat(query, **kwargs))


def test_explanation_is_paraphrase_with_backend_sources_and_cache():
    service, provider, now = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = encoded(section("Jayangrana menggambarkan raga manusia."))
    first = ask(service, "Apa filosofi Jayangrana?")
    assert first.narrative and first.answer == "Jayangrana menggambarkan raga manusia. [1]"
    assert first.sources[0].document == "wayang_db.json"
    assert first.sources[0].evidence_ids == ["profile:0:filosofi:0"]
    assert ask(service, "Apa filosofi Jayangrana?").cache_hit
    assert provider.narrate.await_count == 1
    now[0] = 900
    ask(service, "Apa filosofi Jayangrana?")
    assert provider.narrate.await_count == 2
    provider.select.assert_not_called()
    provider.search.assert_not_called()


@pytest.mark.parametrize("query,style", [("Jelaskan lebih sederhana", "simplify"),
    ("Jelaskan lebih lanjut", "detail"), ("Ringkas penjelasannya", "summary")])
def test_followup_passes_validated_context_and_style(query, style):
    service, provider, _ = setup()
    previous = ask(service, "Apa filosofi Jayangrana?")
    follow = ask(service, query, conversation_context=previous.conversation_context)
    assert follow.mode == "answer" and follow.character.id == 0 and follow.narrative
    _, rows, context = provider.narrate.call_args.args
    assert context["style"] == style and context["subject_ids"] == [0]
    assert all(row["id"] == 0 for row in rows)
    if style in {"simplify", "summary"}:
        assert all(row["evidence_id"] == "profile:0:filosofi:0" for row in rows)


@pytest.mark.parametrize("query", ["coba jelaskan lebih terperinci lagi",
    "tolong jelaskan lebih rinci", "jelaskan secara terperinci", "bisa jelaskan lebih detail lagi"])
def test_natural_detail_followup_uses_last_validated_subject(query):
    service, provider, _ = setup()
    previous = ask(service, "Apa filosofi Jayangrana?")
    reply = ask(service, query, conversation_context=previous.conversation_context)
    assert reply.narrative and reply.character.id == 0
    assert provider.narrate.call_args.args[2]["style"] == "detail"
    assert provider.narrate.call_args.args[2]["subject_ids"] == [0]
    assert ask(service, query).mode == "conversation"
    before = provider.narrate.await_count
    ask(service, query + " tentang politik", conversation_context=previous.conversation_context)
    assert provider.narrate.await_count == before


def test_field_and_comparison_coverage():
    service, provider, _ = setup()
    traits = ask(service, "Apa ciri Jayangrana?")
    assert traits.sources[0].evidence_ids == ["profile:0:ciri_ciri:0"]
    reply = ask(service, "Bandingkan filosofi Jayangrana dan Maktal")
    assert reply.mode == "answer" and reply.character is None
    assert {row.id for row in reply.sources} == {0, 2}
    assert reply.conversation_context.subject_ids == [0, 2]


@pytest.mark.parametrize("failure", [ProviderFailure("rate_limit"), ProviderFailure("unavailable"),
    TimeoutError(), ValueError("synthetic")])
def test_provider_failures_never_cached(failure):
    service, provider, _ = setup()
    provider.narrate.side_effect = failure
    for _ in range(2):
        reply = ask(service, "Apa filosofi Jayangrana?")
        assert reply.mode == "fallback_catalog" and not reply.narrative and not reply.sources
    assert not service.rag_cache.entries and provider.narrate.await_count == 2


@pytest.mark.parametrize("case,status,message", [
    ("provider", "unavailable", "Layanan penjelasan sedang tidak tersedia."),
    ("insufficient", "insufficient_evidence", "Basis pengetahuan lokal belum memuat bukti yang cukup"),
    ("rejected", "invalid_response", "Penjelasan belum dapat diverifikasi terhadap bukti lokal.")])
def test_fallback_explains_actual_category(case, status, message):
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    if case == "provider":
        provider.narrate.side_effect = ProviderFailure("unavailable")
    elif case == "insufficient":
        provider.narrate.return_value = '{"status":"insufficient_evidence","sections":[]}'
    else:
        provider.narrate.return_value = encoded(section("Maktal melambangkan raga manusia."))
    for _ in range(2):
        reply = ask(service, "Apa filosofi Jayangrana?")
        assert reply.llm_status == status and message in reply.answer
        assert reply.mode == "fallback_catalog" and not reply.sources and not reply.llm_used
        assert "Anda tetap dapat melihat informasi profil lokal." in reply.answer
        assert not reply.cache_hit
    assert not service.rag_cache.entries and provider.narrate.await_count == 2


@pytest.mark.parametrize("missing", ["retrieval", "field"])
def test_missing_local_evidence_uses_knowledge_limit_message(missing):
    service, provider, _ = setup()
    if missing == "retrieval":
        service.retrieval.semantic.scores = lambda *_: {}
        service.retrieval.threshold = 1
    else:
        service.retrieval.catalog.profiles[0]["filosofi"] = ""
    reply = ask(service, "Apa filosofi Jayangrana?")
    assert reply.llm_status == "insufficient_evidence"
    assert "Basis pengetahuan lokal" in reply.answer
    provider.narrate.assert_not_called()


@pytest.mark.parametrize("raw", ["invalid-json", '{"status":"insufficient_evidence","sections":[]}',
    encoded(section("Maktal melambangkan raga manusia.")),
    encoded(section("Jayangrana tidak melambangkan raga manusia.")),
    encoded(section("Jayangrana lahir tahun 1200.")),
    encoded(section("Jayangrana lebih kuat daripada Maktal.", kind="comparison")),
    encoded(section("Menurut museum, Jayangrana melambangkan raga manusia.")),
    encoded(section("Jayangrana melambangkan raga manusia.", field="deskripsi"))])
def test_rejected_or_insufficient_narrative_falls_back(raw):
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = raw
    reply = ask(service, "Apa filosofi Jayangrana?")
    assert reply.mode == "fallback_catalog" and not reply.sources and not service.rag_cache.entries


def test_comparison_cannot_omit_one_subject():
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = encoded(section("Jayangrana melambangkan raga manusia."))
    reply = ask(service, "Bandingkan filosofi Jayangrana dan Maktal")
    assert reply.mode == "fallback_catalog" and not service.rag_cache.entries


def test_contract_and_style_invalidate_cache(monkeypatch):
    service, provider, _ = setup()
    ask(service, "Apa filosofi Jayangrana?")
    ask(service, "Ringkas filosofi Jayangrana")
    assert provider.narrate.await_count == 2
    monkeypatch.setattr("app.narrative.NARRATIVE_VERSION", "next-contract")
    ask(service, "Apa filosofi Jayangrana?")
    assert provider.narrate.await_count == 3
    service.retrieval.catalog.profiles[0]["filosofi"] = "Jayangrana memiliki filosofi fixture baru."
    ask(service, "Apa filosofi Jayangrana?")
    assert provider.narrate.await_count == 4


def test_factual_evaluation_records_semantic_gap_not_a_truth_guarantee():
    service, _, _ = setup()
    rows = [{"evidence_id": "profile:0:filosofi:0", "id": 0, "label": "jayangrana",
             "nama": "Jayangrana", "text": service.retrieval.catalog.profiles[0]["filosofi"]}]
    # This fabricated nonnumeric fact passes deterministic provenance checks.
    # Keep this evaluation visible: passing IDs must never be reported as entailment.
    invented = "Jayangrana mendirikan kerajaan di pulau terpencil."
    reply = render_narrative(encoded(section(invented)), rows, service.retrieval.catalog, [0])
    assert reply.narrative and invented in reply.answer


def test_negative_evidence_cannot_be_rewritten_as_positive():
    service, _, _ = setup()
    rows = [{"evidence_id": "profile:0:deskripsi:0", "id": 0, "label": "jayangrana",
             "nama": "Jayangrana", "text": "Jayangrana tidak memakai mahkota."}]
    with pytest.raises(ValueError, match="polarity"):
        render_narrative(encoded(section("Jayangrana memakai mahkota.", field="deskripsi")),
                         rows, service.retrieval.catalog, [0])



def test_unsupported_comparison_rejected_even_with_both_valid_ids():
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = encoded({"text": "Jayangrana lebih berani daripada Maktal.",
        "kind": "comparison", "subject_ids": [0, 2],
        "evidence_ids": ["profile:0:filosofi:0", "profile:2:filosofi:0"]})
    reply = ask(service, "Bandingkan filosofi Jayangrana dan Maktal")
    assert reply.mode == "fallback_catalog" and not service.rag_cache.entries


def test_sensitive_narrative_not_cached():
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = encoded(section("Jayangrana melambangkan raga. Email: user@example.org"))
    ask(service, "Apa filosofi Jayangrana?")
    ask(service, "Apa filosofi Jayangrana?")
    assert not service.rag_cache.entries and provider.narrate.await_count == 2



def test_supported_comparison_paraphrase_has_two_local_sources():
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = encoded({"text": "Jayangrana melambangkan raga manusia, sedangkan Maktal melambangkan keberanian.",
        "kind": "comparison", "subject_ids": [0, 2],
        "evidence_ids": ["profile:0:filosofi:0", "profile:2:filosofi:0"]})
    reply = ask(service, "Bandingkan filosofi Jayangrana dan Maktal")
    assert reply.narrative and reply.answer.endswith("[1] [2]")
    assert [source.evidence_ids for source in reply.sources] == [["profile:0:filosofi:0"], ["profile:2:filosofi:0"]]
    assert reply.character is None and reply.conversation_context.subject_ids == [0, 2]


@pytest.mark.parametrize("text", ["   ", "\n\t"])
def test_blank_section_is_rejected_without_caching(text):
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = encoded(section(text))
    reply = ask(service, "Apa filosofi Jayangrana?")
    assert reply.llm_status == "invalid_response" and not service.rag_cache.entries


def test_comparison_must_declare_its_kind():
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    provider.narrate.return_value = encoded({
        "text": "Jayangrana melambangkan raga manusia, sedangkan Maktal melambangkan keberanian.",
        "kind": "explanation", "subject_ids": [0, 2],
        "evidence_ids": ["profile:0:filosofi:0", "profile:2:filosofi:0"]})
    reply = ask(service, "Bandingkan filosofi Jayangrana dan Maktal")
    assert reply.llm_status == "invalid_response" and not service.rag_cache.entries


def test_evaluation_records_subject_swap_with_valid_comparison_ids():
    service, provider, _ = setup()
    provider.narrate.side_effect = None
    # Both owners and IDs are valid, but their meanings are swapped. This is a
    # known semantic false acceptance, not an assertion of cultural correctness.
    provider.narrate.return_value = encoded({
        "text": "Jayangrana melambangkan keberanian, sedangkan Maktal melambangkan raga manusia.",
        "kind": "comparison", "subject_ids": [0, 2],
        "evidence_ids": ["profile:0:filosofi:0", "profile:2:filosofi:0"]})
    reply = ask(service, "Bandingkan filosofi Jayangrana dan Maktal")
    assert reply.narrative
