import json
from types import SimpleNamespace

import pytest

from app.evaluation import evaluate
from evaluation.mock_search import CASES_PATH, mock_service
from scripts.colab_search_smoke_test import actual_cases, check_report


@pytest.mark.parametrize("profile,expected", [
    ({"ciri_ciri": "  Tidak memakai perhiasan ... dan membawa keris.  "},
     "Tidak memakai perhiasan ... dan membawa keris."),
    ({"ciri_ciri": [None, 42, "", "  ", "  membawa keris  ", "mahkota"]}, "membawa keris"),
    ({"ciri_ciri": "x" * 250}, "x" * 200),
    ({"ciri_ciri": ["  " + "x" * 250]}, "x" * 200),
    ({"ciri_ciri": ""}, None),
    ({"ciri_ciri": "  "}, None),
    ({"ciri_ciri": []}, None),
    ({"ciri_ciri": [None, " ", 42]}, None),
    ({"ciri_ciri": None}, None),
    ({"ciri_ciri": 42}, None),
    ({"ciri_ciri": {"0": "keris"}}, None),
    ({}, None),
])
def test_actual_cases_traits(profile, expected):
    catalog = SimpleNamespace(items=[SimpleNamespace(id=0, nama="Jayangrana")], profiles={0: profile})
    base = [row for row in json.loads(CASES_PATH.read_text(encoding="utf-8"))
            if not row.get("mock_only")]
    cases = actual_cases(catalog)
    assert cases[:len(base)] == base
    assert cases[len(base):] == ([] if expected is None else [
        {"query": f"Jelaskan ciri Jayangrana: {expected}", "mode": "answer", "id": 0}
    ])


def test_actual_cases_skips_empty_character():
    catalog = SimpleNamespace(
        items=[SimpleNamespace(id=i, nama=str(i)) for i in range(3)],
        profiles={0: {"ciri_ciri": None}, 1: {"ciri_ciri": "keris"}, 2: {"ciri_ciri": "mahkota"}},
    )
    assert actual_cases(catalog)[-1] == {"query": "Jelaskan ciri 1: keris", "mode": "answer", "id": 1}


def test_report_printed_before_failure_with_compact_diagnosis(capsys):
    cases = [
        {"query": "Apa fungsi Gunungan?", "mode": "answer", "id": 0},
        {"query": "Siapa presiden Jepang?", "mode": "answer", "id": 0},
        {"query": "Jelaskan Jayangrana", "mode": "answer", "id": 0},
    ]
    report = evaluate(mock_service(), cases)
    report[0]["query"] += " — ciri"
    with pytest.raises(SystemExit) as exc:
        check_report(report, cases)
    assert exc.value.code == 1
    output = capsys.readouterr().out
    printed, end = json.JSONDecoder().raw_decode(output)
    assert printed == report
    assert output.startswith("[\n  {") and "— ciri" in output
    failed = json.loads(output[end:])["failed_cases"]
    assert len(failed) == 2
    assert failed[0] == {
        "query": report[0]["query"], "expected_mode": "answer", "expected_id": 0,
        "actual_mode": "answer", "actual_top_id": 19,
        "score": report[0]["score"], "semantic_score": report[0]["semantic_score"],
        "lexical_score": report[0]["lexical_score"],
    }
    assert failed[1] == {
        "query": cases[1]["query"], "expected_mode": "answer", "expected_id": 0,
        "actual_mode": "fallback_catalog", "actual_top_id": None,
    }
    assert all(set(row) <= {"query", "mode", "top_id", "passed", "score", "semantic_score", "lexical_score"}
               for row in printed)


def test_successful_report_printed_without_failure_summary(capsys):
    cases = [{"query": "Apa fungsi Gunungan?", "mode": "answer", "id": 19}]
    report = evaluate(mock_service(), cases)
    check_report(report, cases)
    assert json.loads(capsys.readouterr().out) == report
