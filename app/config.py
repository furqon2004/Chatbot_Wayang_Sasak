from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_ROOT = Path("/content/drive/MyDrive/skripsi_wayang")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Baseline artefak proyek setelah penambahan chunk gunungan__filosofi.
EXPECTED_CHROMA_DOCUMENT_COUNT = 48
# Initial calibration artefak asli Colab; belum memakai holdout terpisah.
DEFAULT_SEARCH_THRESHOLD = 0.5590755322575569


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
class SecuritySettings:
    production: bool = False
    service_token: str = field(default="", repr=False, compare=False)
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "testserver")
    cors_origins: tuple[str, ...] = ()
    requests_per_minute: int = 60
    concurrency: int = 2
    max_pixels: int = 16_000_000
    body_timeout: float = 15
    public_docs: bool = False
    strict_artifacts: bool = False  # DI mocks may contain fewer than the official chunk baseline.

    def __post_init__(self):
        if self.production and (len(self.service_token) < 32 or not self.allowed_hosts):
            raise ValueError("Production membutuhkan service token kuat dan allowed hosts")
        if any(not host or any(c in host for c in "*/\\\r\n") for host in self.allowed_hosts):
            raise ValueError("Allowed hosts harus eksplisit")
        if any(c.isspace() for c in self.service_token):
            raise ValueError("Service token tidak valid")
        for origin in self.cors_origins:
            parsed = urlsplit(origin)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path
                    or parsed.query or parsed.fragment or parsed.username or '*' in origin
                    or (self.production and parsed.scheme != "https")):
                raise ValueError("CORS membutuhkan origin eksplisit; production harus HTTPS")
        if not 1 <= self.requests_per_minute <= 10000 or not 1 <= self.concurrency <= 32:
            raise ValueError("Rate limit/concurrency tidak valid")
        if not 1 <= self.max_pixels <= 40_000_000 or not 0 < self.body_timeout <= 60:
            raise ValueError("Batas upload tidak valid")

    @classmethod
    def from_env(cls):
        mode = os.getenv("WAYANG_ENV", "development")
        if mode not in {"development", "production"}:
            raise ValueError("WAYANG_ENV tidak valid")
        def csv(name, default):
            return tuple(v.strip() for v in os.getenv(name, default).split(",") if v.strip())
        return cls(production=mode == "production",
            service_token=os.getenv("WAYANG_SERVICE_TOKEN", ""),
            allowed_hosts=csv("WAYANG_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"),
            cors_origins=csv("WAYANG_CORS_ORIGINS", ""),
            requests_per_minute=_positive_int_env("WAYANG_RATE_LIMIT", 60),
            concurrency=_positive_int_env("WAYANG_MAX_CONCURRENCY", 2),
            max_pixels=_positive_int_env("WAYANG_MAX_PIXELS", 16_000_000),
            public_docs=_bool_env("WAYANG_PUBLIC_DOCS", False), strict_artifacts=True)


@dataclass(frozen=True, slots=True)
class Settings:
    model_dir: Path
    knowledge_base_path: Path
    max_upload_bytes: int = 10 * 1024 * 1024
    chroma_dir: Path = DEFAULT_ROOT / "chroma_db_wayang"
    collection_name: str = "wayang_sasak"
    search_threshold: float = DEFAULT_SEARCH_THRESHOLD
    normalize_embeddings: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_api_key: str = field(default="", repr=False, compare=False)
    llm_timeout_seconds: float = 45.0
    llm_max_attempts: int = 2
    web_timeout_seconds: float = 45.0
    web_max_attempts: int = 2
    wayang_web_cache_ttl_seconds: int = 900
    wayang_web_cache_max_entries: int = 32
    chat_enabled: bool = False
    security: SecuritySettings = field(default_factory=SecuritySettings)
    embedding_local_only: bool = False
    embedding_revision: str | None = None
    min_confidence: float = 0.85
    min_margin: float = 0.20

    def __post_init__(self):
        for name, value in (("WAYANG_MIN_CONFIDENCE", self.min_confidence),
                            ("WAYANG_MIN_MARGIN", self.min_margin)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} harus finite dalam rentang 0-1")
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,63}", self.collection_name):
            raise ValueError("Nama collection tidak valid")
        if not math.isfinite(self.search_threshold) or not 0 <= self.search_threshold <= 1:
            raise ValueError("Threshold harus dalam rentang 0-1")
        if self.llm_provider != "openai":
            raise ValueError("LLM_PROVIDER harus openai")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", self.llm_model):
            raise ValueError("LLM_MODEL tidak valid")
        if not math.isfinite(self.llm_timeout_seconds) or not 0 < self.llm_timeout_seconds <= 60:
            raise ValueError("LLM_TIMEOUT_SECONDS harus > 0 dan <= 60")
        if not 1 <= self.llm_max_attempts <= 2:
            raise ValueError("LLM_MAX_ATTEMPTS harus 1-2")
        if not math.isfinite(self.web_timeout_seconds) or not 0 < self.web_timeout_seconds <= 90:
            raise ValueError("WEB_TIMEOUT_SECONDS harus > 0 dan <= 90")
        if not 1 <= self.web_max_attempts <= 2:
            raise ValueError("WEB_MAX_ATTEMPTS harus 1-2")
        if not 1 <= self.wayang_web_cache_max_entries <= 32 or self.wayang_web_cache_ttl_seconds <= 0:
            raise ValueError("Batas cache web tidak valid")

    @classmethod
    def from_env(cls) -> "Settings":
        chat_enabled = _bool_env("WAYANG_ENABLE_CHAT", False)
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
            min_confidence=_unit_float_env("WAYANG_MIN_CONFIDENCE", 0.85),
            min_margin=_unit_float_env("WAYANG_MIN_MARGIN", 0.20),
            normalize_embeddings=_bool_env("WAYANG_NORMALIZE_EMBEDDINGS", False),
            embedding_local_only=_bool_env("WAYANG_EMBEDDING_LOCAL_ONLY", False),
            embedding_revision=os.getenv("WAYANG_EMBEDDING_REVISION", "").strip() or None,
            # Read server environment only when explicitly enabling Stage 3.
            llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip() if chat_enabled else "",
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
            llm_max_attempts=_positive_int_env("LLM_MAX_ATTEMPTS", 2),
            chat_enabled=chat_enabled,
            security=SecuritySettings.from_env(),
            web_timeout_seconds=float(os.getenv("WEB_TIMEOUT_SECONDS", "45")),
            web_max_attempts=_positive_int_env("WEB_MAX_ATTEMPTS", 2),
            wayang_web_cache_ttl_seconds=_positive_int_env("WAYANG_WEB_CACHE_TTL_SECONDS", 900),
            wayang_web_cache_max_entries=_positive_int_env("WAYANG_WEB_CACHE_MAX_ENTRIES", 32),
        )
