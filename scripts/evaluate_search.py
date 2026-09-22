"""python scripts/evaluate_search.py --mock (tidak mengakses artefak asli)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import calibrate_threshold, evaluate
from evaluation.mock_search import CASES_PATH, mock_service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", required=True)
    parser.parse_args()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    service = mock_service()
    threshold, stats = calibrate_threshold(service, cases)
    service.threshold = threshold
    report = evaluate(service, cases)
    print(json.dumps({"scope": "MOCK ONLY", "threshold": threshold, "stats": stats, "cases": report}, indent=2))
    if not all(row["passed"] for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
