from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .resources import EMBEDDING_DIM, ModelResources
from .public import display_name as _display_name, public_profile as _profile


def probabilities_from_logits(
    logits_raw: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Rumus model aktif dari Untitled33.ipynb: softmax(logits / T)."""
    logits_scaled = logits_raw / temperature
    return torch.softmax(logits_scaled, dim=-1)


def prediction_decision(confidence: float, margin: float,
                        min_confidence: float, min_margin: float) -> dict[str, Any]:
    """Compare unrounded values with >=; exact equality passes, without epsilon."""
    reasons = []
    if not confidence >= min_confidence:
        reasons.append("low_confidence")
    if not margin >= min_margin:
        reasons.append("low_margin")
    return {"accepted": not reasons, "reasons": reasons}


UNCERTAIN_MESSAGE = (
    "Gambar belum dapat dikenali dengan yakin. Gunakan foto yang lebih jelas, "
    "menampilkan seluruh tubuh wayang, dan berlatar sederhana."
)


def predict_image(image: Image.Image, resources: ModelResources, *,
                  min_confidence: float = 0.85, min_margin: float = 0.20) -> dict[str, Any]:
    inputs = resources.image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(resources.device)
    if pixel_values.ndim != 4 or tuple(pixel_values.shape[-2:]) != (224, 224):
        raise RuntimeError("Ukuran tensor input model bukan 224x224")
    with torch.inference_mode():
        outputs = resources.vision_encoder(pixel_values=pixel_values)
        image_embeds = outputs.image_embeds
        if image_embeds.ndim != 2 or image_embeds.shape[-1] != EMBEDDING_DIM:
            raise RuntimeError("Dimensi image embedding bukan 512")
        embeddings = F.normalize(image_embeds, p=2, dim=-1)
        logits = resources.classifier(embeddings)
        probabilities = probabilities_from_logits(logits, resources.temperature)
        top_probabilities, top_ids = probabilities[0].topk(k=3)

    ids = [int(value) for value in top_ids.detach().cpu().tolist()]
    probs = [float(value) for value in top_probabilities.detach().cpu().tolist()]
    top_3: list[dict[str, Any]] = []
    for class_id, probability in zip(ids, probs):
        character = resources.characters[class_id]
        top_3.append(
            {
                "id": class_id,
                "label": resources.labels[class_id],
                "nama": _display_name(character, resources.labels[class_id]),
                "probability": round(probability, 6),
            }
        )

    winner_id = ids[0]
    winner = resources.characters[winner_id]
    decision = prediction_decision(probs[0], probs[0] - probs[1], min_confidence, min_margin)
    accepted = decision["accepted"]
    return {
        "status": "accepted" if accepted else "uncertain",
        "decision": decision,
        "confidence": round(probs[0], 6),
        "margin": round(probs[0] - probs[1], 6),
        "top_3_status": "candidates",
        "message": None if accepted else UNCERTAIN_MESSAGE,
        "prediksi": {
            "id": winner_id,
            "label": resources.labels[winner_id],
            "nama": _display_name(winner, resources.labels[winner_id]),
            "confidence": round(probs[0], 6),
            "margin": round(probs[0] - probs[1], 6),
        } if accepted else None,
        "top_3": top_3,
        "profil": _profile(winner) if accepted else None,
    }
