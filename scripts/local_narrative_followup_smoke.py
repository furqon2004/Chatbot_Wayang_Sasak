"""Two-call follow-up smoke using a saved successful first smoke result."""
import argparse
import contextlib
import io
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.local_narrative_smoke import run_local, smoke


def previous_context(previous, catalog):
    from app.conversation import validated_context
    if previous.get("mode") != "answer" or previous.get("llm_status") != "used":
        raise ValueError("A successful first response is required")
    sources = previous["sources"]
    ids = list(dict.fromkeys(eid for source in sources for eid in source["evidence_ids"]))
    if not ids or any(source["document"] != "wayang_db.json" or source["id"] != 0 for source in sources):
        raise ValueError("Expected Jayangrana local evidence")
    context = {"subject_ids": [0], "evidence_ids": ids, "intent": "philosophy", "domain": "wayang_sasak"}
    validated, evidence = validated_context(context, catalog)
    saved = {row["evidence_id"]: row["text"] for row in previous["evidence"]}
    if any(saved.get(row["evidence_id"]) != row["text"] for row in evidence):
        raise ValueError("Evidence changed since first response")
    return validated.model_dump()


async def followups(settings, retrieval, previous, *, transport=None):
    context = previous_context(previous, retrieval.catalog)
    results = []
    # Each invocation has a hard one-call guard; there are at most two invocations.
    for query, ctx in [("Jelaskan lebih sederhana", context),
                       ("Bandingkan filosofi Jayangrana dan Maktal", None)]:
        result = await smoke(settings, retrieval, transport=transport, query=query,
                             conversation_context=ctx)
        results.append(result)
        if result["llm_status"] != "used":
            break
    return {"steps": results, "provider_requests": sum(r["provider_requests"] for r in results),
            "completed": len(results) == 2 and all(r["llm_status"] == "used" for r in results)}


def ui_replay(previous, steps):
    """Render actual smoke results in Streamlit AppTest, with NO network/provider."""
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    responses = [previous, *steps]
    contexts, calls = [], []
    def api(path, *, payload=None, **kwargs):
        if path == "/health":
            return {"status": "ok"}
        if path != "/chat" or len(calls) >= 3:
            raise AssertionError("Unexpected UI request")
        index = len(calls)
        if index == 1:
            assert payload["conversation_context"] == contexts[0]
        calls.append(payload)
        row = responses[index]
        sources = row["sources"]
        context = row.get("conversation_context") or {
            "subject_ids": sorted({s["id"] for s in sources}),
            "evidence_ids": [eid for s in sources for eid in s["evidence_ids"]],
            "intent": "philosophy", "domain": "wayang_sasak"}
        contexts.append(context)
        return {**row, "answer": row["rendered_narrative"], "narrative": True,
                "conversation_context": context}
    with patch("ui_client.request_api", side_effect=api):
        at = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py",
                               default_timeout=30).run()
        for index, query in enumerate(["Apa filosofi Jayangrana?", "Jelaskan lebih sederhana",
                                      "Bandingkan filosofi Jayangrana dan Maktal"]):
            next(w for w in at.text_input if w.label == "Pesan").input(query).run()
            next(b for b in at.button if b.label == "Kirim").click().run()
            assert not at.exception
            texts = [t.value for kind in (at.text, at.markdown) for t in kind]
            assert any(responses[index]["rendered_narrative"] in value for value in texts)
            for number, source in enumerate(responses[index]["sources"], 1):
                assert any(f"[{number}] {source['document']}" in t and source["label"] in t for t in texts)
        assert len(calls) == 3
    return {"mode": "Streamlit AppTest replay", "context_forwarded": True,
            "narratives_visible": True, "numbered_sources_visible": True,
            "additional_provider_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path,
                        help="JSON output of the successful Jayangrana smoke")
    args = parser.parse_args()
    logging.basicConfig(handlers=[logging.NullHandler()], level=logging.CRITICAL, force=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            previous = json.loads(args.previous.read_text(encoding="utf-8-sig"))
            async def operation(settings, retrieval):
                return await followups(settings, retrieval, previous)
            result = run_local(operation)
            if result["completed"]:
                try:
                    result["ui_smoke"] = ui_replay(previous, result["steps"])
                except Exception:
                    result["ui_smoke"] = {"status": "failed", "category": "ui_replay_error"}
                    result["completed"] = False
    except Exception:
        result = {"error_metadata": [{"category": "local_preflight_or_execution_error"}],
                  "completed": False}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
