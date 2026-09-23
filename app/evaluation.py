"""Kalibrasi ambang terukur; tanpa model/embedding/database baru."""
from __future__ import annotations

from pydantic import ValidationError

from .query_normalizer import normalize_query
from .schemas import SearchRequest


def calibrate_threshold(service, cases: list[dict]) -> tuple[float, dict]:
    scored = []
    for case in cases:
        if case["mode"] not in {"answer", "fallback_catalog"}:
            continue
        ranked = service.rank(normalize_query(case["query"]))
        score = ranked[0].score if ranked else 0.0
        if case["mode"] == "answer" and (not ranked or ranked[0].id != case["id"]):
            raise ValueError("Ranking evaluation tidak cocok; threshold tidak bisa memperbaiki urutan")
        scored.append((score, case["mode"] == "answer"))
    positives = [score for score, positive in scored if positive]
    negatives = [score for score, positive in scored if not positive]
    if not positives or not negatives:
        raise ValueError("Evaluasi membutuhkan query relevan dan tidak relevan")
    # Ambil midpoint separation gap, bukan konstanta hasil tebakan.
    negative_max, positive_min = max(negatives), min(positives)
    if negative_max >= positive_min:
        raise ValueError("Skor overlap; perlu evaluasi/tuning lebih lanjut sebelum menetapkan threshold")
    threshold = (negative_max + positive_min) / 2
    return threshold, {"negative_max": negative_max, "positive_min": positive_min,
                       "positive_count": len(positives), "negative_count": len(negatives)}


def evaluate(service, cases: list[dict]) -> list[dict]:
    report = []
    for case in cases:
        scores = {}
        try:
            request = SearchRequest(query=case["query"])
        except ValidationError:
            passed = case["mode"] == "invalid"
            mode, top_id = "invalid", None
        else:
            response = service.search(request.query)
            mode = response.mode
            top_id = response.results[0].id if mode == "answer" else None
            if mode == "answer":
                top = response.results[0]
                scores = {"score": top.score, "semantic_score": top.semantic_score,
                          "lexical_score": top.lexical_score}
            passed = mode == case["mode"] and ("id" not in case or top_id == case["id"])
            if mode in {"catalog", "fallback_catalog"}:
                passed = passed and [item.id for item in response.catalog] == list(range(20))
        report.append({"query": case["query"], "mode": mode, "top_id": top_id, "passed": passed, **scores})
    return report
