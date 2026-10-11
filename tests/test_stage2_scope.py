"""Regresi ruang lingkup Tahap 2; seluruh resource adalah mock lokal."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.catalog import Catalog
from app.config import Settings
from app.lexical import lexical_score
from app.semantic import SemanticSearch
from evaluation.mock_search import mock_service
from scripts.colab_search_smoke_test import assert_search_payload
from test_retrieval import FakeCollection


def test_default_app_never_reads_provider_key_or_registers_chat(monkeypatch, settings, fake_resources):
    original = os.getenv

    def guarded(name, default=None):
        assert name != "LLM_API_KEY", "Tahap 2 tidak boleh membaca key"
        return original(name, default)

    monkeypatch.setattr(os, "getenv", guarded)
    assert Settings.from_env().llm_api_key == ""
    with TestClient(create_app(settings, lambda _: fake_resources, lambda *_: None)) as client:
        assert client.post("/chat", json={"message": "contoh tokoh"}).status_code == 404
        response = client.post("/search", json={"query": "Coba berikan saya contoh tokoh Wayang Saksak?"})
        assert response.status_code == 200
        assert response.json()["mode"] == "catalog"
        assert len(response.json()["catalog"]) == 20
        assert client.app.state.chat is None


def test_http_search_with_mock_collection_and_embedder(settings, fake_resources):
    catalog = mock_service().catalog
    fake_resources.characters = catalog.characters
    calls = []

    def encode(query, **kwargs):
        calls.append(query)
        return np.ones((1, 384))

    semantic = SemanticSearch(FakeCollection(), SimpleNamespace(encode=encode), catalog)
    bundle = SimpleNamespace(semantic=semantic, close=lambda: None)
    with TestClient(create_app(settings, lambda _: fake_resources, lambda *_: bundle)) as client:
        query = "Coba berikan saya contoh tokoh Wayang Saksak?"
        payload = client.post("/search", json={"query": query}).json()
        assert_search_payload(payload, {"mode": "catalog"}, catalog)
        assert payload["query"] == query
        assert "sasak" in payload["normalized_query"]
        assert calls == []  # Intent katalog melewati encode/query collection.
        payload = client.post("/search", json={"query": "Jelaskan Jayagrana"}).json()
        assert_search_payload(payload, {"mode": "answer", "id": 0}, catalog)
        assert calls == [["jelaskan jayagrana"]]


def test_fuzzy_keyword_boost_is_bounded_and_ignores_random_tokens():
    catalog = mock_service().catalog
    catalog.profiles[2]["kata_kunci"] = ["selendang"]
    exact = lexical_score("selendang", 2, catalog)
    fuzzy = lexical_score("selendangg", 2, catalog)
    assert 0 < fuzzy < exact < 1
    assert lexical_score("xyzqvv", 2, catalog) == 0


def test_catalog_rejects_missing_json_name(fake_resources):
    fake_resources.characters[10].pop("nama")
    with pytest.raises(ValueError, match="Nama katalog"):
        Catalog(fake_resources.labels, fake_resources.characters)


@pytest.mark.parametrize("change", ["name", "missing", "duplicate"])
def test_colab_verifier_rejects_invalid_catalog(change):
    service = mock_service()
    payload = service.search("contoh tokoh").model_dump(exclude_none=True)
    if change == "name":
        payload["catalog"][10]["nama"] = "Nama rekaan"
    elif change == "missing":
        payload["catalog"].pop()
    else:
        payload["catalog"][19] = payload["catalog"][0]
    with pytest.raises(AssertionError):
        assert_search_payload(payload, {"mode": "catalog"}, service.catalog)
