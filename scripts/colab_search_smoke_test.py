"""Smoke asli: tidak menulis JSON/model/Chroma sumber, hanya snapshot sementara."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.evaluation import calibrate_threshold, evaluate


def fingerprints(settings: Settings) -> dict:
    paths = [settings.knowledge_base_path, *sorted(settings.chroma_dir.rglob("*"))]
    result = {}
    for path in paths:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(path)] = digest.hexdigest()
    return result


def actual_cases(catalog):
    path = Path(__file__).resolve().parents[1] / "evaluation" / "search_cases.json"
    cases = [row for row in json.loads(path.read_text(encoding="utf-8")) if not row.get("mock_only")]
    # Ciri tambahan dipilih dari JSON asli, tidak menebak fakta tokoh.
    for item in catalog.items:
        traits = catalog.profiles[item.id].get("ciri_ciri", [])
        if traits and isinstance(traits[0], str) and traits[0].strip():
            cases.append({"query": f"Jelaskan ciri {item.nama}: {traits[0][:200]}", "mode": "answer", "id": item.id})
            break
    return cases


def assert_search_payload(payload, case, catalog):
    """Periksa kontrak HTTP terhadap JSON yang dimuat, termasuk seluruh katalog."""
    assert payload["mode"] == case["mode"], "Mode tidak cocok"
    if payload["mode"] in {"catalog", "fallback_catalog"}:
        expected = [item.model_dump(exclude_none=True) for item in catalog.items]
        assert payload["catalog"] == expected, "Katalog bukan 20 entri JSON resmi"
        assert [row["id"] for row in payload["catalog"]] == list(range(20))
    else:
        results = payload["results"]
        assert len({row["id"] for row in results}) == len(results), "Tokoh duplikat"
        if "id" in case:
            assert results[0]["id"] == case["id"], "Top-1 tidak cocok"
        for row in results:
            assert abs(row["score"] - (0.55 * row["semantic_score"] + 0.45 * row["lexical_score"])) < 1e-9
            assert row["evidence"], "Evidence kosong"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Hitung threshold dengan artefak asli untuk sesi smoke ini")
    parser.add_argument("--expected-count", type=int, default=47)
    args = parser.parse_args()
    settings = Settings.from_env()
    before = fingerprints(settings)
    app = create_app(settings)
    try:
        with TestClient(app) as client:
            health = client.get("/health")
            health.raise_for_status()
            assert health.json()["retrieval_loaded"], "Retrieval asli belum siap"
            assert health.json()["document_count"] == args.expected_count, "Jumlah chunk berbeda"
            service = app.state.retrieval
            cases = actual_cases(service.catalog)
            if args.calibrate:
                threshold, stats = calibrate_threshold(service, cases)
                service.threshold = threshold
                print(json.dumps({"scope": "ARTEFAK ASLI", "threshold": threshold, "calibration": stats}))
            report = evaluate(service, cases)
            assert all(row["passed"] for row in report), "Evaluation artefak asli belum lulus"
            for case in cases:
                response = client.post("/search", json={"query": case["query"]})
                if case["mode"] == "invalid":
                    assert response.status_code == 422
                else:
                    response.raise_for_status()
                    assert_search_payload(response.json(), case, service.catalog)
            print(json.dumps({"metric": service.semantic.metric, "document_count": service.semantic.document_count,
                              "case_count": len(report), "threshold": service.threshold}))
    finally:
        assert fingerprints(settings) == before, "Artefak sumber berubah selama smoke test"
    print("LULUS - smoke retrieval artefak asli; hash JSON/Chroma sumber tidak berubah")


if __name__ == "__main__":
    main()
