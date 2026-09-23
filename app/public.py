"""Serialisasi publik bersama: hanya field profil yang diizinkan."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit


def display_name(character: dict[str, Any], fallback: str) -> str:
    for key in ("nama", "name", "nama_tokoh"):
        if character.get(key):
            return str(character[key])
    return fallback.replace("_", " ").title()


def public_image(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().replace("\\", "/")
    parts = urlsplit(value)
    if parts.query or parts.fragment or parts.username or parts.password:
        return None
    if parts.scheme == "https" and parts.hostname:
        return value
    if parts.scheme or parts.netloc or value.startswith("/"):
        return None
    # Hanya namespace gambar publik, bukan path training/Drive lokal.
    if value.startswith("dataset_asli/") and ".." not in value.split("/"):
        return value
    return None


def public_profile(character: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "deskripsi": "", "deskripsi_lengkap": "", "filosofi": "",
        "ciri_ciri": "", "kata_kunci": [],
    }
    profile = {key: deepcopy(character.get(key, default)) for key, default in defaults.items()}
    # Kontrak Tahap 1 tetap string, termasuk untuk DB lama berbentuk list.
    traits = profile["ciri_ciri"]
    if isinstance(traits, list):
        profile["ciri_ciri"] = "; ".join(
            item.strip() for item in traits if isinstance(item, str) and item.strip()
        )
    elif not isinstance(traits, str):
        profile["ciri_ciri"] = ""
    image = public_image(character.get("gambar_utama"))
    if image is not None:
        profile["gambar_utama"] = image
    return profile
