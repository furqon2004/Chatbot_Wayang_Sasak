"""Fixture sintetis HANYA untuk evaluasi lokal; bukan data produksi."""
import json
from pathlib import Path

from app.catalog import Catalog
from app.config import DEFAULT_SEARCH_THRESHOLD
from app.query_normalizer import normalize_query
from app.resources import OFFICIAL_LABELS
from app.retrieval import RetrievalService

CASES_PATH = Path(__file__).with_name("search_cases.json")


def mock_service(threshold=DEFAULT_SEARCH_THRESHOLD):
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    labels = dict(enumerate(OFFICIAL_LABELS))
    characters = {i: {"label": str(i), "nama": label.replace("_", " ").title(),
                      "deskripsi": f"Profil fixture {label}", "kata_kunci": []}
                  for i, label in labels.items()}
    characters[10]["nama"] = "Darun Dia"
    characters[0]["deskripsi"] = "Tokoh utama Wayang Sasak dalam fixture sintetis."
    characters[2].update(deskripsi="Tokoh berselendang ungu sintetis.", filosofi="Lentera sintetis.")
    characters[19]["filosofi"] = "Filosofi Gunungan dalam fixture sintetis."

    class MockSemantic:
        document_count = 47
        metric = "cosine"

        def scores(self, query, top_k):
            for case in cases:
                if normalize_query(case["query"]) == query:
                    return {int(k): v for k, v in case["mock_semantic"].items()}
            return {}

    return RetrievalService(Catalog(labels, characters), MockSemantic(), threshold)
