"""Resolve only the public image referenced by a character's JSON record."""
import os
from pathlib import Path

from app.public import public_image


def character_image(value):
    safe = public_image(value)
    if safe is None:
        return None
    if safe.startswith("https://"):
        return safe
    relative = Path(safe).relative_to("dataset_asli")
    project = Path(__file__).resolve().parent
    roots = []
    if os.getenv("WAYANG_IMAGE_ROOT"):
        roots.append(Path(os.environ["WAYANG_IMAGE_ROOT"]))
    if os.getenv("WAYANG_DB_PATH"):
        roots.append(Path(os.environ["WAYANG_DB_PATH"]).resolve().parent / "dataset_asli")
    roots.extend([project / "dataset_asli", project / "artifacts" / "dataset_asli"])
    for root in roots:
        root = root.resolve()
        target = (root / relative).resolve()
        if target.is_relative_to(root) and target.is_file():
            return str(target)
    return None
