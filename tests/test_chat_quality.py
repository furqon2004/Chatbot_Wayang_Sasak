from app.chat import evidence_context
"""Presentation regressions with synthetic profiles; no external artifacts."""
import asyncio

import pytest

from app.chat import ChatService, display_excerpt, evidence_reply
from evaluation.mock_search import mock_service


def row(text, field="deskripsi", character=0):
    return {"id": character, "label": f"tokoh_{character}", "nama": f"Tokoh {character}",
            "text": text, "evidence_id": f"profile:{character}:{field}:0"}


@pytest.mark.parametrize("other", [
    "Tokoh utama dalam cerita sintetis.",
    "TOKOH UTAMA DALAM CERITA SINTETIS.",
    "Tokoh  utama\n dalam\tcerita sintetis.",
])
def test_normalized_duplicate_keeps_both_official_evidence_ids(other):
    rows = [row("Tokoh utama dalam cerita sintetis."), row(other, "deskripsi_lengkap")]
    reply = evidence_reply(rows, "not_configured")
    assert reply.answer.count("Tokoh 0:") == 1
    assert reply.sources[0].evidence_ids == [item["evidence_id"] for item in rows]
    assert not reply.llm_used
    assert evidence_reply(rows, "not_configured") == reply


@pytest.mark.parametrize("reverse", [False, True])
def test_contained_quote_uses_most_complete_version(reverse):
    short = "Tokoh utama dalam cerita sintetis."
    complete = short + " Ia membawa lentera dalam fixture."
    rows = [row(short), row(complete, "deskripsi_lengkap")]
    if reverse:
        rows.reverse()
    reply = evidence_reply(rows, "not_configured")
    assert reply.answer.count(short) == 1 and complete in reply.answer
    assert reply.sources[0].evidence_ids == [item["evidence_id"] for item in rows]


def test_identical_text_from_different_characters_is_not_merged():
    text = "Profil sintetis yang sama."
    reply = evidence_reply([row(text), row(text, character=1)], "not_configured")
    assert reply.answer.count(text) == 2
    assert [source.id for source in reply.sources] == [0, 1]


def test_similar_quotes_with_distinct_facts_are_preserved():
    first = "Tokoh sintetis ini membawa lentera merah."
    second = "Tokoh sintetis ini membawa lentera biru."
    reply = evidence_reply([row(first), row(second, "deskripsi_lengkap")], "not_configured")
    assert first in reply.answer and second in reply.answer


@pytest.mark.parametrize("sentence", [False, True])
def test_capped_evidence_ends_on_sentence_or_word_boundary(sentence):
    prefix = "Kalimat sintetis lengkap. " if sentence else ""
    original = prefix + "kata " * 110 + "sehingga selesai."
    excerpt = original[:500]
    reply = evidence_reply([row(excerpt)], "not_configured")
    displayed = reply.answer.split("Tokoh 0: ")[1]
    assert original.startswith(displayed)
    assert len(displayed) <= 500 and len(reply.answer) <= 8000
    if sentence:
        assert displayed == "Kalimat sintetis lengkap."
    else:
        assert displayed.endswith("kata")
        assert original[len(displayed)].isspace()


def test_single_overlong_word_is_not_rendered_as_a_fragment():
    assert display_excerpt("x" * 500) == ""


def test_maximum_context_stays_within_answer_limit():
    rows = [row((f"Fakta {index}. " + "kata " * 100)[:500], character=index)
            for index in range(12)]
    reply = evidence_reply(rows, "not_configured")
    assert len(reply.answer) <= 8000
    assert len(reply.sources) == 12


def test_chat_fallback_deduplicates_real_shape_and_preserves_provenance():
    retrieval = mock_service()
    result = retrieval.search("Siapa tokoh utama Wayang Sasak?")
    hit = result.results[0]
    text = "Tokoh utama dalam cerita sintetis."
    for field in ("deskripsi", "deskripsi_lengkap"):
        retrieval.catalog.profiles[hit.id][field] = text
    hit.evidence = [text, text]
    hit.evidence_ids = [f"profile:{hit.id}:{field}:0" for field in ("deskripsi", "deskripsi_lengkap")]
    result.results = [hit]
    retrieval.search = lambda *_: result
    reply = asyncio.run(ChatService(retrieval).chat("Siapa tokoh utama Wayang Sasak?"))
    assert reply.answer.count(text) == 1
    assert reply.llm_status == "not_configured" and not reply.llm_used
    assert reply.sources[0].id == hit.id and reply.sources[0].label == hit.label
    assert reply.sources[0].evidence_ids == hit.evidence_ids
    from scripts.colab_chat_smoke_test import _assert_sources
    _assert_sources(reply.model_dump(), evidence_context(result.results, retrieval.catalog))


def test_chat_trims_actual_partial_word_and_smoke_accepts_safe_excerpt():
    from scripts.colab_chat_smoke_test import _assert_sources
    retrieval = mock_service()
    result = retrieval.search("Jelaskan Jayangrana")
    hit = result.results[0]
    original = "Kalimat lengkap. " + "kata " * 96 + "sehingga selesai."
    assert original[:500].endswith("seh")
    retrieval.catalog.profiles[hit.id]["deskripsi"] = original
    hit.evidence = [original[:500]]
    hit.evidence_ids = [f"profile:{hit.id}:deskripsi:0"]
    retrieval.search = lambda *_: result
    reply = asyncio.run(ChatService(retrieval).chat("Jelaskan Jayangrana"))
    assert reply.answer.endswith("Kalimat lengkap.")
    assert reply.sources[0].evidence_ids == hit.evidence_ids
    _assert_sources(reply.model_dump(), evidence_context(result.results, retrieval.catalog))
    invented = reply.model_dump()
    invented["answer"] += " Fakta tambahan."
    with pytest.raises(AssertionError):
        _assert_sources(invented, evidence_context(result.results, retrieval.catalog))


@pytest.mark.parametrize("has_philosophy", [True, False])
def test_gunungan_philosophy_is_grounded_or_rejected_when_absent(has_philosophy):
    import json
    retrieval = mock_service()
    evidence_id = "profile:19:filosofi:0"
    # Synthetic fixture only: never substitute model knowledge for project data.
    original = retrieval.catalog.profiles[19]["filosofi"]
    if not has_philosophy:
        del retrieval.catalog.profiles[19]["filosofi"]
    sent = []
    class Selector:
        async def select(self, query, context):
            sent.extend(context)
            return json.dumps({"evidence_ids": [evidence_id]})
    reply = asyncio.run(ChatService(retrieval, Selector()).chat("Apa filosofi Gunungan?"))
    assert (evidence_id in {row["evidence_id"] for row in sent}) == has_philosophy
    if has_philosophy:
        assert reply.llm_used and reply.llm_status == "used"
        assert reply.answer == "Berdasarkan basis pengetahuan Wayang Sasak:\n\nGunungan: " + original
        assert reply.sources[0].evidence_ids == [evidence_id]
    else:
        assert not reply.llm_used and reply.llm_status == "invalid_response"
        assert original not in reply.answer
        assert all(evidence_id not in source.evidence_ids for source in reply.sources)
    # These project concepts cannot appear unless supplied by fixture evidence.
    assert all(word not in reply.answer for word in ("Khuldi", "Iblis", "Adam", "Hawa"))
