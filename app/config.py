from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("/content/drive/MyDrive/skripsi_wayang")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SEARCH_THRESHOLD = 0.457  # midpoint gap fixture: (0.220 + 0.694) / 2


def _unit_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} harus berupa angka") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} harus dalam rentang 0-1")
    return value


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} harus true atau false")
    return value == "true"


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} harus berupa bilangan bulat") from exc
    if value <= 0:
        raise ValueError(f"{name} harus lebih besar dari nol")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    model_dir: Path
    knowledge_base_path: Path
    max_upload_bytes: int = 10 * 1024 * 1024
    chroma_dir: Path = DEFAULT_ROOT / "chroma_db_wayang"
    collection_name: str = "wayang_sasak"
    search_threshold: float = DEFAULT_SEARCH_THRESHOLD
    normalize_embeddings: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,63}", self.collection_name):
            raise ValueError("Nama collection tidak valid")
        if not math.isfinite(self.search_threshold) or not 0 <= self.search_threshold <= 1:
            raise ValueError("Threshold harus dalam rentang 0-1")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_dir=Path(
                os.getenv(
                    "WAYANG_MODEL_DIR",
                    str(DEFAULT_ROOT / "model_clip_b32_terbaik"),
                )
            ),
            knowledge_base_path=Path(
                os.getenv("WAYANG_DB_PATH", str(DEFAULT_ROOT / "wayang_db.json"))
            ),
            max_upload_bytes=_positive_int_env(
                "WAYANG_MAX_UPLOAD_BYTES", 10 * 1024 * 1024
            ),
            chroma_dir=Path(os.getenv("WAYANG_CHROMA_DIR", str(DEFAULT_ROOT / "chroma_db_wayang"))),
            collection_name=os.getenv("WAYANG_COLLECTION", "wayang_sasak"),
            search_threshold=_unit_float_env("WAYANG_SEARCH_THRESHOLD", DEFAULT_SEARCH_THRESHOLD),
            normalize_embeddings=_bool_env("WAYANG_NORMALIZE_EMBEDDINGS", False),
        )
