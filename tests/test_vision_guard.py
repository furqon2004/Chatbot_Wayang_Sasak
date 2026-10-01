import json
import logging
import math
from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import TypeAdapter, ValidationError

from app.api import create_app
from app.config import Settings
from app.inference import UNCERTAIN_MESSAGE, predict_image, prediction_decision
from app.schemas import PredictionResponse
from conftest import image_bytes


@pytest.mark.parametrize("confidence,margin,minimum_margin,reasons", [
    (0.90, 0.80, 0.20, []),
    (0.85, 0.70, 0.20, []),
    (0.90, 0.20, 0.20, []),
    (0.84, 0.70, 0.20, ["low_confidence"]),
    (0.90, 0.19, 0.20, ["low_margin"]),
    (0.50, 0.10, 0.20, ["low_confidence", "low_margin"]),
    (0.95, 0.91, 0.92, ["low_margin"]),
    (math.nextafter(0.85, 0), 0.70, 0.20, ["low_confidence"]),
    (0.90, math.nextafter(0.20, 0), 0.20, ["low_margin"]),
])
def test_decision_boundaries(confidence, margin, minimum_margin, reasons):
    assert prediction_decision(confidence, margin, 0.85, minimum_margin) == {
        "accepted": not reasons, "reasons": reasons,
    }


@pytest.mark.parametrize("mode", ["development", "production"])
@pytest.mark.parametrize("name,field", [
    ("WAYANG_MIN_CONFIDENCE", "min_confidence"), ("WAYANG_MIN_MARGIN", "min_margin"),
])
@pytest.mark.parametrize("value", ["0", "1", "0.85", "0.20", "nan", "inf", "-inf", "-0.1", "1.01", "abc", ""])
def test_environment(monkeypatch, mode, name, field, value):
    monkeypatch.setenv("WAYANG_ENV", mode)
    monkeypatch.setenv("WAYANG_SERVICE_TOKEN", "synthetic-test-value-" * 3)
    monkeypatch.setenv(name, value)
    if value in {"0", "1", "0.85", "0.20"}:
        assert getattr(Settings.from_env(), field) == float(value)
    else:
        with pytest.raises(ValueError, match=name):
            Settings.from_env()


def test_defaults(settings):
    assert (settings.min_confidence, settings.min_margin) == (0.85, 0.20)
    parsed = Settings.from_env()
    assert (parsed.min_confidence, parsed.min_margin) == (0.85, 0.20)


@pytest.mark.parametrize("field", ["min_confidence", "min_margin"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
def test_direct_settings_validation(settings, field, value):
    with pytest.raises(ValueError, match="WAYANG_MIN_"):
        replace(settings, **{field: value})


def test_guard_preserves_probabilities_and_does_not_build_rejected_profile(fake_resources, monkeypatch):
    with Image.new("RGB", (12, 12)) as image:
        accepted = predict_image(image, fake_resources, min_confidence=0, min_margin=0)
        def forbidden_profile(*args):
            raise AssertionError("Rejected prediction must not serialize a profile")
        monkeypatch.setattr("app.inference._profile", forbidden_profile)
        uncertain = predict_image(image, fake_resources)
    assert accepted["status"] == "accepted"
    assert uncertain["status"] == "uncertain"
    for field in ("confidence", "margin", "top_3"):
        assert uncertain[field] == accepted[field]
    expected = torch.softmax(fake_resources.classifier.bias / fake_resources.temperature, dim=0)
    values, ids = expected.topk(3)
    assert [v["probability"] for v in uncertain["top_3"]] == [round(v.item(), 6) for v in values]
    assert [v["id"] for v in uncertain["top_3"]] == ids.tolist()
    assert uncertain["prediksi"] is uncertain["profil"] is None
    assert uncertain["top_3_status"] == "candidates"
    assert uncertain["message"] == UNCERTAIN_MESSAGE
    TypeAdapter(PredictionResponse).validate_python(uncertain)
    with pytest.raises(ValidationError):
        TypeAdapter(PredictionResponse).validate_python({**uncertain, "profil": accepted["profil"]})


@pytest.mark.parametrize("confident,margin_threshold,status", [
    (False, 0.20, "uncertain"), (True, 0.20, "accepted"), (True, 1.0, "uncertain"),
])
def test_endpoint_decision_and_safe_logging(settings, fake_resources, caplog, confident, margin_threshold, status):
    if confident:
        with torch.no_grad():
            fake_resources.classifier.bias[19] = 5.0
    settings = replace(settings, min_margin=margin_threshold)
    with caplog.at_level(logging.INFO), TestClient(create_app(settings, lambda _: fake_resources)) as client:
        response = client.post("/predict-image", files={
            "file": ("private-upload-name.png", image_bytes("PNG"), "image/png"),
        })
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == status
    if status == "uncertain":
        assert payload["prediksi"] is payload["profil"] is None
        assert payload["message"] == UNCERTAIN_MESSAGE
    else:
        assert payload["prediksi"]["confidence"] > 0.85  # No confidence cap.
        assert payload["prediksi"]["label"] == "gunungan"
    records = [json.loads(r.message) for r in caplog.records if r.name == "wayang.security"]
    decision = next(r for r in records if r.get("mode") == status)
    assert set(decision) == {"mode", "category", "latency_ms", "request_id"}
    assert decision["request_id"] == response.headers["x-request-id"]
    assert "private-upload-name" not in caplog.text
    assert "Deskripsi gunungan" not in caplog.text
