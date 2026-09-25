from app.config import EXPECTED_CHROMA_DOCUMENT_COUNT
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from evaluation.mock_search import mock_service
from app.resources import parse_knowledge_base
from conftest import image_bytes


@pytest.mark.parametrize("traits,expected", [
    ({"ciri_ciri": "Tidak memakai perhiasan ... dan membawa keris."},
     "Tidak memakai perhiasan ... dan membawa keris."),
    ({"ciri_ciri": ["Tidak memakai perhiasan", "membawa keris"]},
     "Tidak memakai perhiasan; membawa keris"),
    ({"ciri_ciri": None}, ""),
    ({}, ""),
    ({"ciri_ciri": []}, ""),
])
def test_profile_compatibility_search_and_prediction(settings, fake_resources, traits, expected):
    character = fake_resources.characters[19]
    character.pop("ciri_ciri")
    character.update(traits, gambar_utama="dataset_asli/gunungan/1.PNG")
    # Lewati parser DB sebelum kedua endpoint, tanpa mengubah input sumber.
    fake_resources.characters = parse_knowledge_base(
        list(fake_resources.characters.values()), fake_resources.labels
    )
    bundle = SimpleNamespace(semantic=mock_service().semantic, close=lambda: None)
    with TestClient(create_app(settings, lambda _: fake_resources, lambda *_: bundle)) as client:
        search = client.post("/search", json={"query": "Apa fungsi Gunungan?"})
        prediction = client.post("/predict-image", files={
            "file": ("sample.png", image_bytes("PNG"), "image/png")
        })
    assert search.status_code == prediction.status_code == 200
    hit = search.json()["results"][0]
    assert hit["id"] == prediction.json()["prediksi"]["id"] == 19
    assert hit["profil"] == prediction.json()["profil"] == {
        "deskripsi": "Deskripsi gunungan",
        "deskripsi_lengkap": "Deskripsi lengkap gunungan",
        "filosofi": "Filosofi gunungan",
        "ciri_ciri": expected,
        "kata_kunci": ["gunungan"],
        "gambar_utama": "dataset_asli/gunungan/1.PNG",
    }
    if "ciri_ciri" in traits:
        assert character["ciri_ciri"] == traits["ciri_ciri"]
    else:
        assert "ciri_ciri" not in character


@pytest.fixture
def search_client(settings, fake_resources):
    service = mock_service()
    fake_resources.characters = service.catalog.characters
    bundle = SimpleNamespace(semantic=service.semantic, close=lambda: None)
    with TestClient(create_app(settings, lambda _: fake_resources, lambda *_: bundle)) as client:
        yield client


@pytest.mark.parametrize("query,mode", [
    ("Apa fungsi Gunungan?", "answer"),
    ("Coba berikan saya contoh tokoh Wayang Saksak?", "catalog"),
    ("Siapa presiden Jepang?", "fallback_catalog"),
])
def test_search_modes(search_client, query, mode):
    response = search_client.post("/search", json={"query": query})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == mode
    if mode == "answer":
        assert payload["results"][0]["id"] == 19
        assert "embedding" not in response.text and "metadata" not in response.text
    else:
        assert [row["id"] for row in payload["catalog"]] == list(range(20))


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "   "}, {"query": "!?"}, {"query": "x" * 1001},
    {"query": "test", "top_k": 0}, {"query": "test", "top_k": 6},
    {"query": "test", "top_k": True}, {"query": 123},
])
def test_search_validation(search_client, payload):
    assert search_client.post("/search", json=payload).status_code == 422


def test_unavailable_search_does_not_break_catalog_or_prediction(settings, fake_resources):
    def unavailable(*args):
        raise RuntimeError("secret internal path")
    with TestClient(create_app(settings, lambda _: fake_resources, unavailable)) as client:
        response = client.post("/search", json={"query": "Jelaskan Jayangrana"})
        assert response.status_code == 503 and "secret" not in response.text
        assert client.post("/search", json={"query": "contoh tokoh"}).status_code == 200
        health = client.get("/health").json()
        assert health["model_loaded"] and not health["retrieval_loaded"]


def test_retrieval_loads_once_and_reports_count(settings, fake_resources):
    calls = []
    bundle = SimpleNamespace(semantic=mock_service().semantic, close=lambda: calls.append("close"))
    def loader(*args):
        calls.append("load")
        return bundle
    with TestClient(create_app(settings, lambda _: fake_resources, loader)) as client:
        for _ in range(2):
            assert client.post("/search", json={"query": "Jayangrana"}).status_code == 200
        health = client.get("/health").json()
        assert health["retrieval_loaded"] and health["document_count"] == EXPECTED_CHROMA_DOCUMENT_COUNT
    assert calls == ["load", "close"]
