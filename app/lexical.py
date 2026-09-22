from __future__ import annotations

from rapidfuzz.fuzz import ratio

from .catalog import Catalog
from .query_normalizer import normalize_query, terms

FUZZY_NAME_THRESHOLD = 88.0


def lexical_score(query: str, class_id: int, catalog: Catalog) -> float:
    item = catalog.items[class_id]
    character = catalog.characters[class_id]
    aliases = character.get("alias", character.get("aliases", []))
    if isinstance(aliases, str):
        aliases = [aliases]
    names = [normalize_query(name) for name in [item.nama, item.label, *aliases] if isinstance(name, str)]
    if any(f" {name} " in f" {query} " for name in names if name):
        return 1.0
    tokens = query.split()
    fuzzy = 0.0
    for name in names:
        size = len(name.split())
        if not size or len(name) < 5:
            continue
        for index in range(len(tokens) - size + 1):
            window = " ".join(tokens[index:index + size])
            similarity = ratio(window, name)
            if similarity >= FUZZY_NAME_THRESHOLD:
                fuzzy = max(fuzzy, 0.9 * similarity / 100)
    query_terms = terms(query)
    if not query_terms:
        return fuzzy
    profile = catalog.profiles[class_id]
    keywords = terms(" ".join(profile.get("kata_kunci", [])))
    text = " ".join(str(profile.get(key, "")) for key in (
        "deskripsi", "deskripsi_lengkap", "filosofi", "ciri_ciri"
    ))
    # Coverage query, bukan frekuensi token/ukuran deskripsi.
    keyword_score = 0.85 * len(query_terms & keywords) / len(query_terms)
    body_score = 0.65 * len(query_terms & terms(text)) / len(query_terms)
    return max(fuzzy, keyword_score, body_score)
