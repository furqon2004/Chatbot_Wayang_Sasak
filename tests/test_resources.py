from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app.config import Settings
from app.resources import (
    OFFICIAL_LABELS,
    ResourceValidationError,
    classifier_state,
    load_resources,
    parse_knowledge_base,
    parse_label_mapping,
    parse_temperature,
    validate_required_artifacts,
)
from conftest import FakeProcessor, FakeVisionEncoder


def test_label_mapping_supports_both_directions():
    id_to_label = {str(i): label for i, label in enumerate(OFFICIAL_LABELS)}
    label_to_id = {label: i for i, label in enumerate(OFFICIAL_LABELS)}
    expected = dict(enumerate(OFFICIAL_LABELS))
    assert parse_label_mapping(id_to_label) == expected
    assert parse_label_mapping(label_to_id) == expected


def test_knowledge_base_accepts_numeric_string_labels():
    labels = dict(enumerate(OFFICIAL_LABELS))
    raw = [{"label": str(i), "nama": label} for i, label in labels.items()]
    parsed = parse_knowledge_base(raw, labels)
    assert set(parsed) == set(range(20))
    assert parsed[0]["label"] == "0"


def test_model_temperature_is_used_and_supcon_temperature_is_ignored():
    model_config = {
        "temperature": 0.21597230434417725,
        "SUPCON_TEMPERATURE": 0.10,
    }
    assert parse_temperature(model_config) == pytest.approx(0.21597230434417725)


def test_missing_artifact_is_reported_without_modifying_files(tmp_path):
    settings = Settings(tmp_path / "missing-model", tmp_path / "missing-db.json")
    with pytest.raises(ResourceValidationError, match="Artefak wajib tidak ditemukan"):
        validate_required_artifacts(settings)


def test_classifier_checkpoint_shapes():
    state = {
        "classifier.weight": torch.zeros((20, 512)),
        "classifier.bias": torch.zeros(20),
    }
    parsed = classifier_state({"state_dict": state})
    assert tuple(parsed["weight"].shape) == (20, 512)
    with pytest.raises(ResourceValidationError, match="Shape bobot"):
        classifier_state({"weight": torch.zeros((20, 10)), "bias": torch.zeros(20)})


def _write_fake_artifacts(base: Path) -> Settings:
    model = base / "model"
    vision = model / "vision_encoder"
    processor = model / "image_processor"
    vision.mkdir(parents=True)
    processor.mkdir()
    (vision / "config.json").write_text("{}", encoding="utf-8")
    (vision / "model.safetensors").write_bytes(b"fake")
    (processor / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (model / "classifier_head.pt").write_bytes(b"fake")
    (model / "label_mapping.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(OFFICIAL_LABELS)}),
        encoding="utf-8",
    )
    (model / "model_config.json").write_text(
        json.dumps({"temperature": 0.21597230434417725}), encoding="utf-8"
    )
    database = base / "wayang_db.json"
    database.write_text(
        json.dumps(
            [{"label": str(i), "nama": label} for i, label in enumerate(OFFICIAL_LABELS)]
        ),
        encoding="utf-8",
    )
    return Settings(model, database)


def test_loader_uses_cpu_when_cuda_is_unavailable(tmp_path, monkeypatch):
    settings = _write_fake_artifacts(tmp_path)
    calls: dict[str, object] = {}

    def fake_vision_loader(path, *, local_files_only):
        calls["vision"] = (path, local_files_only)
        return FakeVisionEncoder()

    def fake_processor_loader(path, *, local_files_only):
        calls["processor"] = (path, local_files_only)
        return FakeProcessor()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "app.resources.CLIPVisionModelWithProjection.from_pretrained", fake_vision_loader
    )
    monkeypatch.setattr(
        "app.resources.CLIPImageProcessor.from_pretrained", fake_processor_loader
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "weight": torch.zeros((20, 512)),
            "bias": torch.zeros(20),
        },
    )

    resources = load_resources(settings)
    assert resources.device.type == "cpu"
    assert resources.vision_encoder.training is False
    assert resources.classifier.training is False
    assert calls["vision"][1] is True
    assert calls["processor"][1] is True
