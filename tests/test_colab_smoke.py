from __future__ import annotations

import pytest
import torch

from conftest import image_bytes
from scripts.colab_smoke_test import _notebook_equivalent_reference


def test_colab_reference_uses_real_resource_temperature(tmp_path, fake_resources):
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(image_bytes("PNG"))

    reference = _notebook_equivalent_reference(image_path, fake_resources)
    expected = torch.softmax(
        fake_resources.classifier.bias / fake_resources.temperature, dim=-1
    )
    values, ids = expected.topk(3)

    assert reference["prediksi"]["id"] == int(ids[0])
    assert reference["prediksi"]["confidence"] == pytest.approx(
        float(values[0].detach())
    )
    assert reference["prediksi"]["margin"] == pytest.approx(
        float((values[0] - values[1]).detach())
    )
    assert [row["id"] for row in reference["top_3"]] == ids.tolist()
