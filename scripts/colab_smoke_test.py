"""Smoke test artefak asli; jalankan setelah Google Drive di-mount.

Secara default skrip menghitung referensi independen dengan urutan operasi dari
notebook. JSON keluaran notebook dapat diberikan untuk validasi tambahan.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def _close(actual: float, expected: float, tolerance: float, name: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{name} berbeda: endpoint={actual}, evaluasi={expected}, tol={tolerance}"
        )


def _notebook_equivalent_reference(image_path: Path, resources) -> dict:
    """Implementasi pembanding yang tidak memanggil pipeline endpoint."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = resources.image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(resources.device)
    with torch.inference_mode():
        image_embeddings = resources.vision_encoder(
            pixel_values=pixel_values
        ).image_embeds
        image_embeddings = torch.nn.functional.normalize(
            image_embeddings, p=2, dim=-1
        )
        logits_raw = resources.classifier(image_embeddings)
        logits_scaled = logits_raw / resources.temperature
        probabilities = torch.softmax(logits_scaled, dim=-1)
        values, ids = probabilities[0].topk(3)
    values_list = [float(value) for value in values.cpu().tolist()]
    ids_list = [int(class_id) for class_id in ids.cpu().tolist()]
    return {
        "prediksi": {
            "id": ids_list[0],
            "confidence": values_list[0],
            "margin": values_list[0] - values_list[1],
        },
        "top_3": [
            {"id": class_id, "probability": probability}
            for class_id, probability in zip(ids_list, values_list)
        ],
    }


def _compare(actual: dict, reference: dict, tolerance: float, source: str) -> None:
    assert actual["prediksi"]["id"] == reference["prediksi"]["id"], source
    assert [row["id"] for row in actual["top_3"]] == [
        row["id"] for row in reference["top_3"]
    ], source
    for index, (actual_row, reference_row) in enumerate(
        zip(actual["top_3"], reference["top_3"])
    ):
        _close(
            actual_row["probability"],
            reference_row["probability"],
            tolerance,
            f"{source}.top_3[{index}].probability",
        )
    _close(
        actual["prediksi"]["confidence"],
        reference["prediksi"]["confidence"],
        tolerance,
        f"{source}.confidence",
    )
    _close(
        actual["prediksi"]["margin"],
        reference["prediksi"]["margin"],
        tolerance,
        f"{source}.margin",
    )


def main() -> None:
    args = _arguments()
    if not args.image.is_file():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image}")
    if args.reference_json is not None and not args.reference_json.is_file():
        raise FileNotFoundError(f"Referensi tidak ditemukan: {args.reference_json}")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from main import app

    mime = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        direct_reference = _notebook_equivalent_reference(
            args.image, app.state.resources
        )
        response = client.post(
            "/predict-image",
            files={"file": (args.image.name, args.image.read_bytes(), mime)},
        )
        response.raise_for_status()

    actual = response.json()
    _compare(actual, direct_reference, args.tolerance, "referensi langsung")
    if args.reference_json is not None:
        notebook_reference = json.loads(
            args.reference_json.read_text(encoding="utf-8")
        )
        _compare(actual, notebook_reference, args.tolerance, "JSON notebook")
    print(
        "LULUS - health, formula divide, top-1, urutan/probabilitas top-3, "
        "confidence, dan margin cocok"
    )
    print(json.dumps({"health": health.json(), "prediction": actual}, indent=2))


if __name__ == "__main__":
    main()
