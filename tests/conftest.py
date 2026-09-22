from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from app.config import Settings
from app.resources import ModelResources, OFFICIAL_LABELS


class FakeProcessor:
    def __init__(self) -> None:
        self.seen_modes: list[str] = []

    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        self.seen_modes.append(images.mode)
        return {"pixel_values": torch.ones((1, 3, 224, 224))}


class FakeVisionEncoder(nn.Module):
    def forward(self, *, pixel_values):
        batch_size = pixel_values.shape[0]
        embedding = torch.arange(1, 513, dtype=torch.float32).repeat(batch_size, 1)
        return SimpleNamespace(image_embeds=embedding.to(pixel_values.device))


@pytest.fixture
def fake_resources() -> ModelResources:
    classifier = nn.Linear(512, 20)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.bias.copy_(torch.linspace(-1.0, 1.0, 20))
    labels = dict(enumerate(OFFICIAL_LABELS))
    characters = {
        class_id: {
            "label": str(class_id),
            "nama": label.replace("_", " ").title(),
            "deskripsi": f"Deskripsi {label}",
            "deskripsi_lengkap": f"Deskripsi lengkap {label}",
            "filosofi": f"Filosofi {label}",
            "ciri_ciri": ["ciri"],
            "kata_kunci": [label],
        }
        for class_id, label in labels.items()
    }
    return ModelResources(
        vision_encoder=FakeVisionEncoder().eval(),
        image_processor=FakeProcessor(),
        classifier=classifier.eval(),
        device=torch.device("cpu"),
        labels=labels,
        characters=characters,
        model_config={"temperature": 0.5},
        temperature=0.5,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_dir=tmp_path / "model",
        knowledge_base_path=tmp_path / "wayang_db.json",
        max_upload_bytes=1024 * 1024,
        chroma_dir=tmp_path / "chroma-missing",
    )


def image_bytes(image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA" if image_format == "PNG" else "RGB", (12, 12), "red").save(
        buffer, format=image_format
    )
    return buffer.getvalue()
