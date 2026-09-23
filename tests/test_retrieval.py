import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.evaluation import calibrate_threshold, evaluate
from app.lexical import lexical_score
from app.query_normalizer import normalize_query
from app.retrieval import hybrid_score
from app.semantic import (RetrievalUnavailable, SemanticSearch, collection_metric,
                          distance_score, metadata_class_id)
from evaluation.mock_search import CASES_PATH, mock_service


@pytest.mark.parametrize("metric,distance,expected", [
    ("cosine", 0.2, 0.8), ("cosine", 1.5, 0), ("l2", 3, 0.25),
    ("ip", 1, 0.5), ("ip", -999, 1), ("ip", 1001, 0),
])
def test_metric_conversion(metric, distance, expected):
    assert distance_score(distance, metric) == pytest.approx(expected)


def test_unknown_metric_and_nonfinite_distance_fail_closed():
    with pytest.raises(RetrievalUnavailable):
        collection_metric(SimpleNamespace(metadata={}, configuration={}))
    with pytest.raises(RetrievalUnavailable):
        distance_score(float("nan"), "cosine")
    with pytest.raises(RetrievalUnavailable):
        collection_metric(SimpleNamespace(metadata={"hnsw:space": "l2"}, configuration={"hnsw": {"space": "cosine"}}))


def test_exact_fuzzy_and_random_name_matching():
    catalog = mock_service().catalog
    assert lexical_score("jelaskan jayangrana", 0, catalog) == 1
    assert 0.8 < lexical_score("jelaskan jayagrana", 0, catalog) < 1
    assert 0.8 < lexical_score("gununggan", 19, catalog) < 1
    assert lexical_score("xyzqvv abczzz", 0, catalog) == 0
    assert hybrid_score(0.8, 0.6) == pytest.approx(0.55 * 0.8 + 0.45 * 0.6)


def test_description_repetition_does_not_raise_score():
    catalog = mock_service().catalog
    first = lexical_score("utama", 0, catalog)
    catalog.profiles[0]["deskripsi"] *= 100
    assert lexical_score("utama", 0, catalog) == first


class FakeCollection:
    metadata = {"hnsw:space": "cosine"}
    def count(self):
        return 3
    def get(self, **kwargs):
        return {"ids": ["a", "b", "c"], "metadatas": [{"label": "0"}, {"nama": "Jayangrana"}, {"slug": "gunungan"}]}
    def query(self, **kwargs):
        assert "query_embeddings" in kwargs and "query_texts" not in kwargs
        return {"ids": [["a", "b", "c"]], "distances": [[0.3, 0.1, 0.2]]}


def test_chunk_grouping_uses_best_semantic_score():
    embedder = SimpleNamespace(encode=lambda *args, **kwargs: np.ones((1, 384)))
    semantic = SemanticSearch(FakeCollection(), embedder, mock_service().catalog)
    assert semantic.scores("query", 3) == pytest.approx({0: 0.9, 19: 0.8})
    service = mock_service()
    service.semantic = semantic
    response = service.search("Jayangrana", 5)
    assert len({row.id for row in response.results}) == len(response.results)
    assert response.results[0].evidence_ids[0].startswith("profile:0:")


def test_metadata_requires_unambiguous_catalog_match():
    catalog = mock_service().catalog
    assert metadata_class_id({"nama_tokoh": "Darun Dia"}, catalog) == 10
    with pytest.raises(RetrievalUnavailable):
        metadata_class_id({"class_id": 0, "label": "gunungan"}, catalog)
    with pytest.raises(RetrievalUnavailable):
        metadata_class_id({"source": "private-path"}, catalog)


def test_threshold_calibration_and_evaluation():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    service = mock_service()
    threshold, stats = calibrate_threshold(service, cases)
    # Kalibrasi fixture sintetis terpisah dari default hasil artefak asli.
    assert threshold == pytest.approx(0.457)
    assert stats["negative_max"] < threshold < stats["positive_min"]
    service.threshold = threshold
    assert all(row["passed"] for row in evaluate(service, cases))


@pytest.mark.parametrize("query,score,expected_mode", [
    ("sesuatu", 0.49093165576457987, "fallback_catalog"),
    ("Jelaskan Jayangrana", 0.6272194087505341, "answer"),
])
def test_initial_calibrated_threshold_with_recorded_score_bounds(
    settings, monkeypatch, query, score, expected_mode
):
    service = mock_service(threshold=settings.search_threshold)
    # Replay batas skor Colab, bukan reproduksi embedding/identitas query asli.
    hit = service.rank("jelaskan jayangrana")[0].model_copy(update={"score": score})
    monkeypatch.setattr(service, "rank", lambda *_: [hit])
    response = service.search(query)
    assert response.mode == expected_mode
    if expected_mode == "answer":
        assert response.results[0].score == score


def test_default_threshold_preserves_mock_relevance():
    service = mock_service()
    assert service.search("sesuatu").mode == "fallback_catalog"
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    weakest = min(
        (case for case in cases if case["mode"] == "answer"),
        key=lambda case: service.rank(normalize_query(case["query"]))[0].score,
    )
    response = service.search(weakest["query"])
    assert response.mode == "answer"
    assert response.results[0].id == weakest["id"]


def test_threshold_boundary():
    service = mock_service()
    score = service.rank(normalize_query("Jelaskan Jayagrana"))[0].score
    service.threshold = score
    assert service.search("Jelaskan Jayagrana").mode == "answer"
    service.threshold = score + 0.00001
    assert service.search("Jelaskan Jayagrana").mode == "fallback_catalog"


def test_calibration_refuses_overlapping_positive_negative_scores():
    service = SimpleNamespace(rank=lambda query: [SimpleNamespace(id=0, score=0.5)])
    cases = [{"query": "positive", "mode": "answer", "id": 0},
             {"query": "negative", "mode": "fallback_catalog"}]
    with pytest.raises(ValueError, match="overlap"):
        calibrate_threshold(service, cases)


def test_empty_collection_is_unavailable():
    collection = FakeCollection()
    collection.count = lambda: 0
    with pytest.raises(RetrievalUnavailable, match="kosong"):
        SemanticSearch(collection, object(), mock_service().catalog)


def test_query_failure_does_not_leak_vendor_exception():
    def broken(*args, **kwargs):
        raise RuntimeError("/content/private-secret")
    semantic = SemanticSearch(FakeCollection(), SimpleNamespace(encode=broken), mock_service().catalog)
    with pytest.raises(RetrievalUnavailable) as caught:
        semantic.scores("query", 3)
    assert "private-secret" not in str(caught.value)
