from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("/content/drive/MyDrive/skripsi_wayang")


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
        )
