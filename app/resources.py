from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .config import Settings

LOGGER = logging.getLogger(__name__)

OFFICIAL_LABELS = (
    "jayangrana",
    "umar_maya",
    "maktal",
    "taptanus",
    "samtanus",
    "selandir",
    "baktak",
    "umar_madi",
    "smakun",
    "hermus",
    "darun_die",
    "ibnu_jarah",
    "kius_nendar",
    "munigarim",
    "nursiwan",
    "lurah",
    "kembung",
    "gulanggi",
    "samardikaran",
    "gunungan",
)
NUM_CLASSES = 20
EMBEDDING_DIM = 512


class ResourceValidationError(RuntimeError):
    """Artefak tidak lengkap, tidak terbaca, atau tidak konsisten."""


@dataclass(slots=True)
class ModelResources:
    vision_encoder: nn.Module
    image_processor: Any
    classifier: nn.Module
    device: torch.device
    labels: dict[int, str]
    characters: dict[int, dict[str, Any]]
    model_config: dict[str, Any]
    temperature: float


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ResourceValidationError(f"JSON tidak valid: {path.name}") from exc
    except OSError as exc:
        raise ResourceValidationError(f"Gagal membaca artefak: {path.name}") from exc


def required_paths(settings: Settings) -> dict[str, Path]:
    model_dir = settings.model_dir
    return {
        "vision config": model_dir / "vision_encoder" / "config.json",
        "vision weights": model_dir / "vision_encoder" / "model.safetensors",
        "image processor": model_dir / "image_processor" / "preprocessor_config.json",
        "classifier": model_dir / "classifier_head.pt",
        "label mapping": model_dir / "label_mapping.json",
        "model config": model_dir / "model_config.json",
        "knowledge base": settings.knowledge_base_path,
    }


def validate_required_artifacts(settings: Settings) -> None:
    missing = [name for name, path in required_paths(settings).items() if not path.is_file()]
    if missing:
        raise ResourceValidationError(
            "Artefak wajib tidak ditemukan: " + ", ".join(sorted(missing))
        )


def parse_label_mapping(raw: Any) -> dict[int, str]:
    # Nama field legacy bisa terbalik; arah ditentukan dari isi, bukan nama key.
    if isinstance(raw, dict):
        nested = [raw[key] for key in (
            "id2label", "label2id", "id_to_label", "label_to_id",
            "idx_to_class", "class_to_idx",
        ) if isinstance(raw.get(key), (dict, list))]
        if nested:
            mappings = [parse_label_mapping(value) for value in nested]
            if any(mapping != mappings[0] for mapping in mappings[1:]):
                raise ResourceValidationError("Mapping nested tidak konsisten")
            return mappings[0]
    parsed: dict[int, str] = {}
    if isinstance(raw, list):
        parsed = {index: str(label) for index, label in enumerate(raw)}
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if str(key).isdigit():
                class_id, label = int(key), str(value)
            elif isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                class_id, label = int(value), str(key)
            else:
                raise ResourceValidationError("Format label_mapping.json tidak dikenali")
            if class_id in parsed:
                raise ResourceValidationError("ID kelas duplikat pada label_mapping.json")
            parsed[class_id] = label
    else:
        raise ResourceValidationError("label_mapping.json harus berupa object atau array")

    if (set(parsed) != set(range(NUM_CLASSES))
            or set(parsed.values()) != set(OFFICIAL_LABELS)):
        raise ResourceValidationError(
            "label_mapping.json harus memuat tepat 20 label resmi dengan ID 0-19"
        )
    return parsed


def _character_rows(raw: Any) -> list[tuple[Any, Any]]:
    if isinstance(raw, list):
        return [(None, value) for value in raw]
    if isinstance(raw, dict):
        for key in ("tokoh", "characters", "data", "wayang"):
            if isinstance(raw.get(key), list):
                return [(None, value) for value in raw[key]]
        return list(raw.items())
    raise ResourceValidationError("wayang_db.json harus berupa object atau array")


def parse_knowledge_base(raw: Any, labels: dict[int, str]) -> dict[int, dict[str, Any]]:
    characters: dict[int, dict[str, Any]] = {}
    for outer_key, item in _character_rows(raw):
        if not isinstance(item, dict):
            raise ResourceValidationError("Setiap entri wayang_db.json harus berupa object")
        raw_id = item.get("label", item.get("id", outer_key))
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ResourceValidationError(
                "Setiap tokoh harus memiliki field label/ID numerik"
            ) from exc
        if class_id not in labels:
            raise ResourceValidationError(f"ID tokoh di luar rentang 0-19: {class_id}")
        if class_id in characters:
            raise ResourceValidationError(f"ID tokoh duplikat: {class_id}")
        # Bila DB juga menyediakan slug, pastikan tidak bertentangan.
        slug = item.get("slug", item.get("class_name"))
        if slug is not None and str(slug) != labels[class_id]:
            raise ResourceValidationError(
                f"Slug tokoh ID {class_id} tidak konsisten dengan label_mapping.json"
            )
        characters[class_id] = dict(item)

    if set(characters) != set(range(NUM_CLASSES)):
        raise ResourceValidationError(
            "wayang_db.json harus memuat tepat 20 tokoh dengan ID 0-19"
        )
    return characters


def parse_temperature(model_config: Any) -> float:
    if not isinstance(model_config, dict):
        raise ResourceValidationError("model_config.json harus berupa object")
    if "temperature" not in model_config:
        raise ResourceValidationError(
            "Field temperature tidak ada di model_config.json"
        )
    if model_config["temperature"] is None:
        return 1.0
    try:
        temperature = float(model_config["temperature"])
    except (TypeError, ValueError) as exc:
        raise ResourceValidationError("Temperature model harus berupa angka") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise ResourceValidationError("Temperature model harus positif dan finite")
    return temperature


def classifier_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "classifier_state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ResourceValidationError("Checkpoint classifier harus berupa state_dict")

    weight = state.get("weight")
    bias = state.get("bias")
    if weight is None or bias is None:
        weights = [value for key, value in state.items() if key.endswith(".weight")]
        biases = [value for key, value in state.items() if key.endswith(".bias")]
        if len(weights) == 1 and len(biases) == 1:
            weight, bias = weights[0], biases[0]
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ResourceValidationError("Bobot/bias classifier tidak ditemukan")
    if tuple(weight.shape) != (NUM_CLASSES, EMBEDDING_DIM):
        raise ResourceValidationError("Shape bobot classifier harus (20, 512)")
    if tuple(bias.shape) != (NUM_CLASSES,):
        raise ResourceValidationError("Shape bias classifier harus (20,)")
    return {"weight": weight, "bias": bias}


def _validate_encoder_config(vision_encoder: nn.Module) -> None:
    config = getattr(vision_encoder, "config", None)
    projection_dim = getattr(config, "projection_dim", None)
    image_size = getattr(config, "image_size", None)
    if projection_dim is not None and int(projection_dim) != EMBEDDING_DIM:
        raise ResourceValidationError("Projection dimension vision encoder harus 512")
    if image_size is not None and int(image_size) != 224:
        raise ResourceValidationError("Image size vision encoder harus 224")


def load_resources(settings: Settings) -> ModelResources:
    validate_required_artifacts(settings)
    artifact_labels = parse_label_mapping(
        _read_json(settings.model_dir / "label_mapping.json")
    )
    labels = dict(enumerate(OFFICIAL_LABELS))
    model_config = _read_json(settings.model_dir / "model_config.json")
    temperature = parse_temperature(model_config)
    characters = parse_knowledge_base(
        _read_json(settings.knowledge_base_path), labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Memuat model pada device=%s", device.type)
    try:
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(settings.model_dir / "vision_encoder"), local_files_only=True
        )
        _validate_encoder_config(vision_encoder)
        image_processor = CLIPImageProcessor.from_pretrained(
            str(settings.model_dir / "image_processor"), local_files_only=True
        )
        if model_config.get("resize_mode") == "aspect_ratio_preserving_white_padding":
            image_processor.do_resize = False
            image_processor.do_center_crop = False
        checkpoint = torch.load(
            settings.model_dir / "classifier_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        classifier = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)
        state = classifier_state(checkpoint)
        artifact_ids = {label: class_id for class_id, label in artifact_labels.items()}
        order = torch.tensor(
            [artifact_ids[label] for label in OFFICIAL_LABELS], dtype=torch.long
        )
        classifier.load_state_dict(
            {key: value.index_select(0, order) for key, value in state.items()},
            strict=True,
        )
        vision_encoder.to(device).eval()
        classifier.to(device).eval()
    except ResourceValidationError:
        raise
    except Exception as exc:
        raise ResourceValidationError("Gagal memuat model dari artefak lokal") from exc

    LOGGER.info("Resource siap: %d kelas", len(labels))
    return ModelResources(
        vision_encoder=vision_encoder,
        image_processor=image_processor,
        classifier=classifier,
        device=device,
        labels=labels,
        characters=characters,
        model_config=model_config,
        temperature=temperature,
    )
