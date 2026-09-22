from __future__ import annotations

from .public import display_name, public_image, public_profile
from .query_normalizer import terms
from .schemas import CatalogItem, CatalogSearch


class Catalog:
    def __init__(self, labels: dict[int, str], characters: dict[int, dict]):
        if set(labels) != set(range(20)) or set(characters) != set(labels):
            raise ValueError("Katalog harus memuat tepat 20 ID")
        self.labels = labels
        self.characters = characters
        self.items = [CatalogItem(
            id=i, label=labels[i], nama=display_name(characters[i], labels[i]),
            gambar_utama=public_image(characters[i].get("gambar_utama")),
        ) for i in range(20)]
        self.profiles = {i: public_profile(character) for i, character in characters.items()}

    def response(self, query: str, normalized: str, fallback: bool = False) -> CatalogSearch:
        message = (
            "Informasi tersebut tidak tersedia atau belum cukup didukung dalam basis pengetahuan. "
            "Sistem berfokus pada Wayang Sasak dan memiliki 20 tokoh. "
            if fallback else "Sistem memiliki 20 tokoh Wayang Sasak. "
        ) + "Silakan pilih salah satu untuk melihat deskripsi, filosofi, atau ciri-cirinya."
        return CatalogSearch(
            mode="fallback_catalog" if fallback else "catalog", query=query,
            normalized_query=normalized, message=message, catalog=self.items,
            suggestions=[f"Jelaskan {self.items[0].nama}", f"Apa filosofi {self.items[19].nama}?"],
        )

    def evidence(self, class_id: int, query: str) -> tuple[list[str], list[str]]:
        # Bukti publik berasal dari JSON tervalidasi, bukan metadata/path chunk mentah.
        rows = []
        wanted = terms(query)
        for key in ("deskripsi", "deskripsi_lengkap", "filosofi", "ciri_ciri", "kata_kunci"):
            value = self.profiles[class_id].get(key)
            for index, text in enumerate(value if isinstance(value, list) else [value]):
                if isinstance(text, str) and text.strip():
                    priority = 2 if key in query else 0
                    rows.append((priority + len(wanted & terms(text)), f"profile:{class_id}:{key}:{index}", text[:500]))
        rows.sort(key=lambda row: -row[0])
        chosen = rows[:3]
        return [row[2] for row in chosen], [row[1] for row in chosen]
