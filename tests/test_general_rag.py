"""Synthetic evidence tests: fixtures do not represent audited production facts."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from app.chat_schemas import WebSource
from app.provider import ProviderFailure
from app.chat import ChatService
from app.query_normalizer import local_overview_intent
from app.semantic import RetrievalUnavailable
from app.schemas import AnswerSearch, SearchHit
from evaluation.mock_search import mock_service


QUERIES = ["Apa itu Wayang Sasak?", "Ceritakan tentang Wayang Sasak.",
           "Apa hubungan Wayang Sasak dengan cerita Menak?"]
DEFINITION = "Wayang Sasak merupakan seni pertunjukan dalam fixture sintetis."
RELATION = "Wayang Sasak mengangkat cerita Menak dalam fixture sintetis."


def service_with(text):
    retrieval = mock_service()
    retrieval.catalog.profiles[0]["deskripsi"] = text
    retrieval.semantic.scores = lambda *_: {0: 1.0}
    # Mock retrieval independently of lexical coverage. The real catalog has
    # no general document; these tests exercise the sufficient-evidence branch.
    def search(query, top_k):
        if query.lower().startswith("daftar"):
            return retrieval.catalog.response(query, query)
        evidence, ids = retrieval.catalog.evidence(0, query)
        return AnswerSearch(query=query, normalized_query=query, results=[SearchHit(
            **retrieval.catalog.items[0].model_dump(), score=.9,
            semantic_score=1., lexical_score=.8, evidence=evidence,
            evidence_ids=ids, profil=retrieval.catalog.profiles[0])])
    retrieval.search = Mock(side_effect=search)

    async def select(query, context):
        return json.dumps({"evidence_ids": [context[0]["evidence_id"]]})

    provider = NS(select=AsyncMock(side_effect=select), search=AsyncMock(side_effect=ProviderFailure("unavailable")),
                settings=NS(llm_model="synthetic"))
    now = [0]
    return ChatService(retrieval, provider, clock=lambda: now[0]), provider, now


@pytest.mark.parametrize("query", QUERIES)
def test_sufficient_direct_evidence_uses_rag_cache_and_exact_source(query):
    text = RELATION if "Menak" in query else DEFINITION
    service, provider, now = service_with(text + " Tokoh fixture memakai keris.")
    first = asyncio.run(service.chat(query))
    assert first.mode == "answer" and first.answer == text
    assert first.character is None and not first.web_sources
    assert first.sources[0].document == "wayang_db.json"
    assert first.sources[0].evidence_ids == ["profile:0:deskripsi:0"]
    assert provider.select.call_args.args[1][0]["text"] == text
    expected_query = "Apa itu Wayang Sasak" if local_overview_intent(query) == "overview" else query
    service.retrieval.search.assert_called_with(expected_query, 3)
    first.answer = "mutated"
    assert asyncio.run(service.chat(query.upper())).answer == text
    assert provider.select.await_count == 1
    now[0] = 900
    asyncio.run(service.chat(query))
    assert provider.select.await_count == 2
    provider.search.assert_not_called()


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("text", ["Jayangrana merupakan tokoh utama Wayang Sasak.",
    "Kembung merupakan tokoh Wayang Sasak yang dikenal dengan julukan Menak Senduk.",
    "Wayang Sasak merupakan tokoh fiktif dalam fixture."])
def test_character_evidence_never_becomes_overview(query, text):
    service, provider, _ = service_with(text)
    reply = asyncio.run(service.chat(query))
    assert reply.mode == "conversation" and reply.catalog is None
    assert reply.llm_status == "unavailable"
    assert "Pencarian web belum dapat diverifikasi" in reply.answer
    assert text not in reply.answer and not reply.web_sources
    assert not service.rag_cache.entries
    provider.select.assert_not_called()
    provider.search.assert_awaited_once_with(query)


@pytest.mark.parametrize("response", ['{"evidence_ids":[]}',
    '{"evidence_ids":["profile:19:filosofi:0"]}',
    '{"evidence_ids":["profile:0:deskripsi:0"],"answer":"Klaim baru"}'])
def test_llm_cannot_add_claims_or_cache_invalid_output(response):
    service, provider, _ = service_with(DEFINITION)
    provider.select.side_effect = None
    provider.select.return_value = response
    reply = asyncio.run(service.chat(QUERIES[0]))
    assert reply.mode == "conversation" or reply.answer == DEFINITION
    assert reply.catalog is None
    assert "Klaim baru" not in reply.answer
    assert not service.rag_cache.entries
    provider.search.assert_awaited_once_with(QUERIES[0])


def test_unavailable_retrieval_uses_web_fallback():
    service, provider, _ = service_with(DEFINITION)
    service.retrieval.search.side_effect = RetrievalUnavailable()
    reply = asyncio.run(service.chat(QUERIES[0]))
    assert reply.mode == "conversation" and reply.catalog is None
    provider.search.assert_awaited_once_with(QUERIES[0])


def test_low_scoring_direct_evidence_does_not_bypass_threshold():
    service, provider, _ = service_with(DEFINITION)
    result = service.retrieval.search(QUERIES[0], 3)
    result.results[0].score = service.retrieval.threshold - .001
    service.retrieval.search.side_effect = None
    service.retrieval.search.return_value = result
    reply = asyncio.run(service.chat(QUERIES[0]))
    assert reply.mode == "conversation" and reply.catalog is None
    provider.select.assert_not_called()
    provider.search.assert_awaited_once_with(QUERIES[0])


def test_changed_evidence_invalidates_general_cache():
    service, provider, _ = service_with(DEFINITION)
    asyncio.run(service.chat(QUERIES[0]))
    service.retrieval.catalog.profiles[0]["deskripsi"] = "Jayangrana adalah tokoh Wayang Sasak."
    reply = asyncio.run(service.chat(QUERIES[0]))
    assert reply.mode == "conversation" and reply.catalog is None
    assert provider.select.await_count == 1
    provider.search.assert_awaited_once_with(QUERIES[0])


@pytest.mark.parametrize("query", ["ceritakan saya tentang wayang sasak",
    "Ceritain dong tentang wayang Sasak", "Tolong jelaskan tentang Wayang Sasak",
    "Berikan saya informasi mengenai wayang sasak", "Apa itu Wayang Sasak ya?",
    "Saya ingin mengenal wayang sasak"])
def test_overview_variants_stay_local(query):
    service, provider, _ = service_with(DEFINITION)
    assert asyncio.run(service.chat(query)).answer == DEFINITION
    provider.search.assert_not_called()


def test_casual_typo_general_query_tries_local_retrieval_first():
    service, provider, _ = service_with(DEFINITION)
    reply = asyncio.run(service.chat("Ceritakan aku soal wayang sasak"))
    assert reply.mode == "answer" and reply.answer == DEFINITION
    service.retrieval.search.assert_called_once()
    provider.search.assert_not_called()


def test_contextual_history_question_uses_local_then_grounded_web():
    service, provider, _ = service_with(DEFINITION)
    service.retrieval.search.side_effect = None
    service.retrieval.search.return_value = service.retrieval.catalog.response("Sejarah Wayang Sasak", "", True)
    provider.search.side_effect = None
    provider.search.return_value = ("Sejarah bersumber dari halaman terverifikasi. [1]",
        [WebSource(title="Sumber fixture", url="https://example.org/sejarah")], None)
    reply = asyncio.run(service.chat("Sejarahnya gimana?",
        conversation_context={"domain": "wayang_sasak"}))
    assert reply.mode == "web_answer" and reply.web_sources[0].url == "https://example.org/sejarah"
    service.retrieval.search.assert_called_once_with("Sejarah Wayang Sasak", 3)
    provider.search.assert_awaited_once_with("Sejarah Wayang Sasak")


def test_general_local_retrieval_insufficiency_uses_web_fallback():
    service, provider, _ = service_with("Jayangrana memakai keris dalam fixture.")
    provider.select.side_effect = None
    provider.select.return_value = json.dumps({"evidence_ids": []})
    provider.search.side_effect = None
    provider.search.return_value = ("Sejarah dari sumber terverifikasi. [1]",
        [WebSource(title="Sumber fixture", url="https://example.org/sejarah")], None)
    reply = asyncio.run(service.chat("Sejarah awal Wayang Sasak"))
    assert reply.mode == "web_answer" and reply.web_sources[0].url == "https://example.org/sejarah"
    service.retrieval.search.assert_called_once()
    provider.search.assert_awaited_once_with("Sejarah awal Wayang Sasak")


def test_mixed_request_keeps_wayang_answer_and_uses_short_boundary_notice():
    service, provider, _ = service_with(DEFINITION)
    reply = asyncio.run(service.chat("Jelaskan iPhone dulu, lalu ceritakan Wayang Sasak"))
    assert reply.mode == "answer" and reply.answer == DEFINITION
    assert reply.scope_notice == "Aku khusus membantu membahas Wayang Sasak, jadi kita fokus ke Wayang Sasak, ya! 🎭"
    provider.search.assert_not_called()


def test_character_profile_explanation_and_catalog_still_work():
    service, provider, _ = service_with("Jayangrana adalah tokoh utama Wayang Sasak.")
    assert asyncio.run(service.chat("Jelaskan Jayangrana", character_id=0)).character.id == 0
    provider.select.assert_not_called()
    assert asyncio.run(service.chat("Jelaskan Jayangrana", character_id=0, explain_more=True)).character.id == 0
    assert asyncio.run(service.chat("Daftar tokoh Wayang Sasak")).mode == "catalog"
    assert provider.select.await_count == 1
    provider.search.assert_not_called()


@pytest.mark.parametrize("query", QUERIES)
def test_missing_overview_uses_grounded_web_without_local_generation(query):
    service, provider, _ = service_with("Jayangrana merupakan tokoh utama Wayang Sasak.")
    provider.search.side_effect = None
    provider.search.return_value = ("Penjelasan sintetis dari sumber web. [1]",
        [WebSource(title="Sumber fixture", url="https://example.org/budaya")], None)
    reply = asyncio.run(service.chat(query))
    assert reply.mode == "web_answer"
    assert reply.answer == "Penjelasan sintetis dari sumber web. [1]"
    assert reply.web_sources[0].url == "https://example.org/budaya"
    assert reply.llm_used
    provider.search.assert_awaited_once_with(query)
    provider.select.assert_not_called()


@pytest.mark.parametrize("query", [
    "apa itu wayang sasak", "wayang sasak itu apa", "wayang sasak apaan sih",
    "aku pengen tahu soal wayang sasak", "ceritain wayang sasak dong",
    "coba jelasin tentang wayang sasak", "wayng sasak itu apaan",
    "aku mau tau wayang sasak", "sebenarnya wayang sasak itu apa sih?",
    "coba critain wayng sasak",
])
def test_semantic_overview_variants_use_one_local_retrieval_query(query):
    service, provider, _ = service_with(DEFINITION)
    reply = asyncio.run(service.chat(query))

    assert reply.mode == "answer" and reply.answer == DEFINITION
    service.retrieval.search.assert_called_once_with("Apa itu Wayang Sasak", 3)
    assert provider.select.call_args.args[0] == query
    provider.search.assert_not_called()


@pytest.mark.parametrize("query", ["Wayang Sasak?", "Wayang Sasak adalah"])
def test_bare_and_incomplete_wayang_requests_use_overview_retrieval(query):
    service, provider, _ = service_with(DEFINITION)

    reply = asyncio.run(service.chat(query))

    assert reply.mode == "answer" and reply.answer == DEFINITION
    service.retrieval.search.assert_called_once_with("Apa itu Wayang Sasak", 3)
    provider.search.assert_not_called()


def test_mixed_hp_then_wayang_request_answers_wayang_overview():
    service, provider, _ = service_with(DEFINITION)

    reply = asyncio.run(service.chat("Jelaskan HP dulu, lalu Wayang Sasak"))

    assert reply.mode == "answer" and reply.answer == DEFINITION
    assert reply.scope_notice
    service.retrieval.search.assert_called_once_with("Apa itu Wayang Sasak", 3)
    provider.search.assert_not_called()


def test_out_of_scope_request_gets_notice_and_catalog_without_retrieval():
    service, provider, _ = service_with(DEFINITION)

    reply = asyncio.run(service.chat("Jelaskan iPhone"))

    assert reply.mode == "fallback_catalog"
    assert "khusus membantu membahas Wayang Sasak" in reply.answer
    assert len(reply.catalog) == 20
    service.retrieval.search.assert_not_called()
    provider.search.assert_not_called()


@pytest.mark.parametrize("query", ["apa itu wayang sasak", "wayang sasak itu apa"])
def test_overview_variants_use_web_when_local_retrieval_is_unavailable(query):
    service, provider, _ = service_with(DEFINITION)
    service.retrieval.search.side_effect = RetrievalUnavailable()
    provider.search.side_effect = None
    provider.search.return_value = ("Penjelasan bersumber [1]",
        [WebSource(title="Sumber fixture", url="https://example.org/budaya")], None)

    reply = asyncio.run(service.chat(query))

    assert reply.mode == "web_answer" and reply.web_sources
    service.retrieval.search.assert_called_once_with("Apa itu Wayang Sasak", 3)
    provider.search.assert_awaited_once_with(query)


@pytest.mark.parametrize("query", ["apa itu wayang sasak", "wayang sasak itu apa"])
def test_overview_variants_use_web_when_local_evidence_is_insufficient(query):
    service, provider, _ = service_with("Jayangrana adalah tokoh utama Wayang Sasak.")
    provider.search.side_effect = None
    provider.search.return_value = ("Penjelasan bersumber [1]",
        [WebSource(title="Sumber fixture", url="https://example.org/budaya")], None)

    reply = asyncio.run(service.chat(query))

    assert reply.mode == "web_answer" and reply.web_sources
    service.retrieval.search.assert_called_once_with("Apa itu Wayang Sasak", 3)
    provider.search.assert_awaited_once_with(query)


@pytest.mark.parametrize("query", ["Wayang Sasak itu apa", "Jelaskan soal Wayang Sasak",
                                   "Jelasin gw tentang Wayang Sasak"])
def test_overview_without_local_or_web_evidence_does_not_show_catalog(query):
    service, _, _ = service_with("Jayangrana adalah tokoh utama Wayang Sasak.")
    service.provider = None

    reply = asyncio.run(service.chat(query))

    assert reply.mode == "conversation" and reply.llm_status == "not_configured"
    assert reply.catalog is None
    assert "belum menemukan sumber" in reply.answer.casefold()
    assert "tidak ingin mengarang" in reply.answer.casefold()
