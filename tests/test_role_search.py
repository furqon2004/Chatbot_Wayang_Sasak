"""Synthetic role assertions, independent of names and production artifacts."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from app.chat import ChatService
from app.role_search import supporting_quote
from evaluation.mock_search import mock_service


def setup(descriptions=None):
    retrieval = mock_service()
    for item in retrieval.catalog.items:
        retrieval.catalog.profiles[item.id]["deskripsi"] = f"{item.nama} adalah tokoh fixture."
    for identity, text in (descriptions or {0: "Jayangrana merupakan tokoh utama dalam pewayangan Sasak."}).items():
        retrieval.catalog.profiles[identity]["deskripsi"] = text
    retrieval.catalog.profiles[0]["filosofi"] = "Filosofi sintetis tentang raga manusia."
    retrieval.semantic.scores = Mock(return_value={i: .99 for i in range(20)})
    async def select(query, rows):
        return json.dumps({"evidence_ids": [rows[0]["evidence_id"]]})
    provider = NS(select=AsyncMock(side_effect=select), search=AsyncMock())
    return ChatService(retrieval, provider), provider


def ask(service, query, **kwargs):
    return asyncio.run(service.chat(query, **kwargs))


def test_nameless_role_uses_description_and_sets_subject():
    service, provider = setup()
    reply = ask(service, "Siapa tokoh utama Wayang Sasak?")
    assert reply.mode == "answer" and reply.character.id == 0
    assert "tokoh utama" in reply.opening and "Jayangrana" in reply.opening
    assert reply.answer.startswith(reply.opening)
    assert reply.sources[0].evidence_ids == ["profile:0:deskripsi:0"]
    assert reply.conversation_context.subject_ids == [0]
    provider.select.assert_not_called()
    provider.search.assert_not_called()
    follow = ask(service, "Apa filosofinya?", conversation_context=reply.conversation_context)
    assert follow.character.id == 0
    assert follow.sources[0].evidence_ids == ["profile:0:filosofi:0"]


def test_identity_is_not_hardcoded():
    service, _ = setup({2: "Maktal merupakan tokoh utama dalam fixture Wayang Sasak."})
    reply = ask(service, "Siapa tokoh utama Wayang Sasak?")
    assert reply.character.id == 2


@pytest.mark.parametrize("context", [{"domain": "wayang_sasak"},
    {"subject_ids": [2], "domain": "wayang_sasak"}])
def test_implicit_role_uses_domain_not_active_character(context):
    service, _ = setup()
    reply = ask(service, "Siapa tokoh utamanya?", conversation_context=context)
    assert reply.character.id == 0


def test_ambiguous_context_requires_clarification_without_retrieval():
    service, provider = setup()
    service.retrieval.rank = Mock(side_effect=AssertionError("Unexpected retrieval"))
    reply = ask(service, "Siapa tokoh utamanya?")
    assert reply.mode == "conversation" and "konteks" in reply.answer
    assert reply.character is None
    provider.select.assert_not_called()


@pytest.mark.parametrize("text", [
    "Tokoh utama dalam cerita sintetis.",
    "Jayangrana merupakan kesatria yang penting dalam Wayang Sasak.",
    "Jayangrana bukan tokoh utama dalam Wayang Sasak.",
    "Jayangrana merupakan pendamping tokoh utama dalam Wayang Sasak.",
    "Jayangrana merupakan sahabat dari tokoh utama dalam Wayang Sasak.",
    "Jayangrana dikenal sebagai putra tokoh utama dalam Wayang Sasak.",
    "Jayangrana mungkin merupakan tokoh utama dalam Wayang Sasak."])
def test_high_similarity_without_direct_claim_never_identifies(text):
    from app.provider import ProviderFailure

    service, provider = setup({0: text})
    provider.search.side_effect = ProviderFailure("unavailable")
    reply = ask(service, "Siapa tokoh utama Wayang Sasak?")
    assert reply.mode == "fallback_catalog" and reply.character is None
    assert not any(source.id is not None for source in reply.sources)
    provider.search.assert_awaited_once()
    provider.select.assert_not_called()


def test_multiple_candidates_show_reasons_without_selecting_active_subject():
    service, _ = setup({0: "Jayangrana adalah tokoh utama dalam fixture Wayang Sasak.",
                        2: "Maktal merupakan tokoh utama dalam fixture Wayang Sasak."})
    reply = ask(service, "Siapa tokoh utama Wayang Sasak?")
    assert reply.mode == "suggestions" and {item.id for item in reply.choices} == {0, 2}
    assert len(reply.sources) == 2 and "Jayangrana adalah" in reply.answer and "Maktal merupakan" in reply.answer
    assert not reply.conversation_context.subject_ids and reply.character is None


@pytest.mark.parametrize("query,text", [
    ("Siapa penasihat utama Wayang Sasak?", "Jayangrana dikenal sebagai penasihat utama dalam fixture Wayang Sasak."),
    ("Siapa tokoh yang bijaksana?", "Jayangrana dikenal sebagai tokoh yang bijaksana."),
    ("Tokoh mana yang membawa keris?", "Jayangrana membawa keris.")])
def test_other_role_and_trait_questions(query, text):
    service, _ = setup({0: text})
    reply = ask(service, query, conversation_context={"domain": "wayang_sasak"})
    assert reply.mode == "answer" and reply.character.id == 0


def test_overview_does_not_rewrite_into_role_search():
    service, provider = setup()
    service.retrieval.search = Mock(wraps=service.retrieval.search)
    reply = ask(service, "Apa itu Wayang Sasak?")
    assert reply.mode == "web_answer" and reply.character is None and reply.catalog is None
    assert reply.web_sources and "Serat Menak" in reply.answer
    service.retrieval.search.assert_called_once_with("Apa itu Wayang Sasak", 3)
    provider.select.assert_not_called()


def test_entire_description_is_searched_not_just_first_excerpt():
    text = "Kalimat fixture. " * 40 + "Jayangrana merupakan tokoh utama dalam Wayang Sasak."
    assert supporting_quote({"deskripsi": text}, "Jayangrana", "tokoh utama", ["Jayangrana"])[1] == \
        "Jayangrana merupakan tokoh utama dalam Wayang Sasak."


def test_failed_web_does_not_erase_explicit_domain_for_role_followup():
    from app.provider import ProviderFailure
    service, provider = setup()
    provider.search.side_effect = ProviderFailure("unavailable")
    first = ask(service, "Sejarah Wayang Sasak")
    assert first.mode == "fallback_catalog"
    assert first.conversation_context.domain == "wayang_sasak"
    second = ask(service, "Siapa tokoh utamanya?", conversation_context=first.conversation_context)
    assert second.character.id == 0
    assert provider.search.await_count == 1


@pytest.mark.parametrize("identity,name", [(0, "Jayangrana"), (2, "Maktal")])
def test_strongest_explains_main_role_without_inventing_ranking(identity, name):
    service, provider = setup({identity: f"{name} merupakan tokoh utama dalam Wayang Sasak."})
    reply = ask(service, "dalam pewayangan sasak tokoh yang paling kuat apa namanya")
    assert f"Jika yang Anda maksud tokoh utama, namanya {name}. [1]" in reply.answer
    assert "perlu sumber yang membandingkan" in reply.answer
    assert reply.sources[0].id == identity
    assert reply.conversation_context.subject_ids == [identity]
    provider.select.assert_not_called()
    provider.search.assert_not_called()


def test_strongest_does_not_invent_main_character_without_evidence():
    service, provider = setup({0: "Jayangrana bukan tokoh utama dalam Wayang Sasak."})
    reply = ask(service, "Siapa paling kuat dalam Wayang Sasak?")
    assert reply.character is None and not reply.sources
    assert "belum memiliki sumber" in reply.answer
    provider.select.assert_not_called()


def test_origin_colloquial_question_reaches_search():
    from app.provider import ProviderFailure
    service, provider = setup()
    provider.search.side_effect = ProviderFailure("unavailable")
    query = "asal mula wayang sasak gimana ceritanya"
    reply = ask(service, query)
    provider.search.assert_awaited_once_with(query)
    assert reply.mode == "fallback_catalog"
    provider.select.assert_not_called()


@pytest.mark.parametrize("query", ["tokoh utamanya siapa", "tokoh utama dalam cerita ini siapa"])
def test_inverted_main_role_question_uses_local_description(query):
    service, provider = setup()

    reply = ask(service, query)

    assert reply.mode == "answer" and reply.character.id == 0
    assert "tokoh utama" in reply.answer and "Jayangrana" in reply.answer
    assert reply.opening == "Berdasarkan deskripsi lokal, Jayangrana merupakan tokoh utama."
    assert reply.sources[0].evidence_ids == ["profile:0:deskripsi:0"]
    provider.search.assert_not_called()


def test_inverted_trait_question_reads_ciri_ciri_profile_field():
    service, provider = setup({2: "Maktal adalah tokoh dalam fixture."})
    service.retrieval.catalog.profiles[2]["ciri_ciri"] = "Dikenal licik dan cerdik."

    reply = ask(service, "Tokoh yang licik siapa?")

    assert reply.mode == "answer" and reply.character.id == 2
    assert "Dikenal licik dan cerdik." in reply.answer
    assert reply.sources[0].evidence_ids == ["profile:2:ciri_ciri:0"]
    provider.search.assert_not_called()


def test_trait_keyword_is_used_with_its_own_evidence_id():
    service, provider = setup({2: "Maktal adalah tokoh dalam fixture."})
    service.retrieval.catalog.profiles[2]["kata_kunci"] = ["pemanah", "licik"]

    reply = ask(service, "Tokoh yang licik siapa?")

    assert reply.mode == "answer" and reply.character.id == 2
    assert reply.sources[0].evidence_ids == ["profile:2:kata_kunci:1"]
    assert "licik" in reply.answer
    provider.search.assert_not_called()


@pytest.mark.parametrize(("field", "text"), [
    ("filosofi", "Maktal melambangkan sifat licik."),
    ("deskripsi_lengkap", "Maktal merupakan tokoh yang licik."),
])
def test_role_search_uses_other_public_profile_fields(field, text):
    service, provider = setup({2: "Maktal adalah tokoh dalam fixture."})
    service.retrieval.catalog.profiles[2][field] = text

    reply = ask(service, "Tokoh yang licik siapa?")

    assert reply.mode == "answer" and reply.character.id == 2
    assert text in reply.answer
    assert reply.sources[0].evidence_ids == [f"profile:2:{field}:0"]
    provider.search.assert_not_called()


def test_negated_trait_in_profile_metadata_never_identifies_a_character():
    service, provider = setup({2: "Maktal adalah tokoh dalam fixture."})
    service.retrieval.catalog.profiles[2]["ciri_ciri"] = "Maktal tidak licik."

    reply = ask(service, "Tokoh yang licik siapa?")

    assert reply.mode == "fallback_catalog" and reply.character is None
    provider.select.assert_not_called()


def test_role_question_without_local_evidence_uses_grounded_web_fallback():
    from app.chat_schemas import WebSource

    service, provider = setup()
    provider.search.return_value = ("Belum ada catatan lokal. [1]",
        [WebSource(title="Sumber fixture", url="https://example.org/wayang")], None)

    reply = ask(service, "Tokoh yang licik siapa?")

    assert reply.mode == "web_answer"
    assert reply.web_sources[0].url == "https://example.org/wayang"
    assert reply.conversation_context.domain == "wayang_sasak"
    provider.search.assert_awaited_once_with("licik dalam Wayang Sasak")
    provider.select.assert_not_called()


def test_role_question_with_explicit_non_wayang_subject_is_not_retrieved():
    service, provider = setup()
    service.retrieval.rank = Mock(side_effect=AssertionError("Out-of-domain retrieval"))

    reply = ask(service, "Tokoh presiden siapa?")

    assert reply.mode == "fallback_catalog"
    provider.search.assert_not_called()
    provider.select.assert_not_called()


def test_character_main_role_question_uses_validated_domain_context():
    service, provider = setup()

    reply = ask(service, "Siapa karakter utamanya?", conversation_context={"domain": "wayang_sasak"})

    assert reply.mode == "answer" and reply.character.id == 0
    assert reply.sources[0].evidence_ids == ["profile:0:deskripsi:0"]
    provider.search.assert_not_called()


def test_informal_more_characters_followup_uses_existing_recommendation_flow():
    service, _ = setup()

    reply = ask(service, "ada yang lain", awaiting_more=True, seen_ids=[0])

    assert reply.mode == "suggestions" and len(reply.choices) == 3
    assert 0 not in {item.id for item in reply.choices}
