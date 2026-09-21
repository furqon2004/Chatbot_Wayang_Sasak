from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.inference import probabilities_from_logits
from app.resources import ModelResources
from conftest import image_bytes


def test_health_and_resources_are_loaded_only_once(settings, fake_resources):
    calls = 0

    def loader(_settings):
        nonlocal calls
        calls += 1
        return fake_resources

    app = create_app(settings, loader)
    with TestClient(app) as client:
        first = client.get("/health")
        second = client.get("/health")

    assert calls == 1
    assert first.status_code == second.status_code == 200
    assert first.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "CLIP ViT-B/32 + linear classifier",
        "device": "cpu",
        "num_classes": 20,
        "knowledge_base_loaded": True,
    }


@pytest.mark.parametrize(
    ("image_format", "mime_type"), [("PNG", "image/png"), ("JPEG", "image/jpeg")]
)
def test_predict_valid_images_matches_explicit_mock_reference(
    settings, fake_resources, image_format, mime_type
):
    app = create_app(settings, lambda _: fake_resources)
    with TestClient(app) as client:
        response = client.post(
            "/predict-image",
            files={"file": (f"sample.{image_format.lower()}", image_bytes(image_format), mime_type)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["top_3"]) == 3
    assert payload["prediksi"]["id"] == 19
    assert payload["prediksi"]["label"] == "gunungan"
    assert payload["profil"]["deskripsi"] == "Deskripsi gunungan"
    assert fake_resources.image_processor.seen_modes[-1] == "RGB"

    # Referensi independen untuk classifier sintetis fixture ini.
    expected = torch.softmax(fake_resources.classifier.bias / 0.5, dim=0)
    values, ids = expected.topk(3)
    assert [item["id"] for item in payload["top_3"]] == ids.tolist()
    assert [item["probability"] for item in payload["top_3"]] == pytest.approx(
        [round(float(value.detach()), 6) for value in values]
    )
    assert payload["prediksi"]["confidence"] == pytest.approx(
        round(float(values[0].detach()), 6)
    )
    assert payload["prediksi"]["margin"] == pytest.approx(
        round(float((values[0] - values[1]).detach()), 6)
    )


def test_two_predictions_do_not_reload_resources(settings, fake_resources):
    calls = 0

    def loader(_settings):
        nonlocal calls
        calls += 1
        return fake_resources

    app = create_app(settings, loader)
    files = {"file": ("sample.png", image_bytes("PNG"), "image/png")}
    with TestClient(app) as client:
        assert client.post("/predict-image", files=files).status_code == 200
        files = {"file": ("sample.png", image_bytes("PNG"), "image/png")}
        assert client.post("/predict-image", files=files).status_code == 200
    assert calls == 1


@pytest.mark.parametrize(
    ("files", "expected_status"),
    [
        ({"file": ("fake.png", b"teks biasa", "image/png")}, 400),
        ({"file": ("broken.jpg", b"\xff\xd8\xffbroken", "image/jpeg")}, 400),
        ({"file": ("notes.txt", b"hello", "text/plain")}, 415),
        ({"file": ("empty.png", b"", "image/png")}, 400),
    ],
)
def test_invalid_uploads(settings, fake_resources, files, expected_status):
    app = create_app(settings, lambda _: fake_resources)
    with TestClient(app) as client:
        response = client.post("/predict-image", files=files)
    assert response.status_code == expected_status


def test_missing_file_returns_422(settings, fake_resources):
    app = create_app(settings, lambda _: fake_resources)
    with TestClient(app) as client:
        response = client.post("/predict-image")
    assert response.status_code == 422


def test_upload_limit_returns_413(tmp_path, fake_resources):
    settings = Settings(
        model_dir=tmp_path / "model",
        knowledge_base_path=tmp_path / "db.json",
        max_upload_bytes=8,
    )
    app = create_app(settings, lambda _: fake_resources)
    with TestClient(app) as client:
        response = client.post(
            "/predict-image", files={"file": ("big.png", b"x" * 9, "image/png")}
        )
    assert response.status_code == 413


def test_health_is_503_before_lifespan(settings):
    app = create_app(settings, lambda _: None)  # lifespan sengaja tidak dijalankan
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "Model belum siap"}


def test_logits_are_divided_by_temperature_before_softmax(monkeypatch):
    logits_raw = torch.tensor([[1.0, 2.0, 3.0]])
    temperature = 0.25
    captured: dict[str, object] = {}
    original_softmax = torch.softmax

    def softmax_spy(logits_scaled, *, dim):
        captured["logits_scaled"] = logits_scaled.clone()
        captured["dim"] = dim
        return original_softmax(logits_scaled, dim=dim)

    monkeypatch.setattr(torch, "softmax", softmax_spy)
    actual = probabilities_from_logits(logits_raw, temperature)

    expected_scaled = logits_raw / temperature
    assert torch.equal(captured["logits_scaled"], expected_scaled)
    assert captured["dim"] == -1
    assert torch.allclose(actual, original_softmax(expected_scaled, dim=-1))


def test_internal_inference_error_hides_details(settings, fake_resources):
    class BrokenClassifier(torch.nn.Module):
        def forward(self, _value):
            raise RuntimeError("detail internal rahasia")

    fake_resources.classifier = BrokenClassifier()
    app = create_app(settings, lambda _: fake_resources)
    with TestClient(app) as client:
        response = client.post(
            "/predict-image",
            files={"file": ("sample.png", image_bytes("PNG"), "image/png")},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Inferensi gambar gagal"}
    assert "rahasia" not in response.text
