"""Opt-in smoke of working-tree code. Only --live contacts Gemini.

Configuration is process-local; source artifacts and .env remain untouched.
"""
import argparse
import asyncio
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OVERRIDES = {
    "GEMINI_TIMEOUT_SECONDS": "45", "GEMINI_SEARCH_TIMEOUT_SECONDS": "90",
    "GEMINI_MAX_ATTEMPTS": "2", "GEMINI_WEB_MAX_ATTEMPTS": "2",
    "WAYANG_MAX_CONCURRENCY": "1", "WAYANG_WEB_CACHE_TTL_SECONDS": "900",
    "WAYANG_WEB_CACHE_MAX_ENTRIES": "32", "WAYANG_ENABLE_CHAT": "true",
}


def report(flow, status, **details):
    print(json.dumps(dict(flow=flow, status=status, **details)), flush=True)


async def live(settings):
    from app.gemini import GeminiClient, GeminiFailure
    if not settings.gemini_api_key:
        report("live", "BLOCKED", reason="not_configured", provider_calls=0)
        return
    client = GeminiClient(settings)
    calls = 0
    generate = client.client.aio.models.generate_content

    async def counted(**kwargs):
        nonlocal calls
        calls += 1
        return await generate(**kwargs)

    client.client.aio.models.generate_content = counted
    query = "Apa itu Wayang Sasak?"
    try:
        answer = await client.search(query)
        initial = calls
        cached = await client.search(query)
        assert cached == answer and calls == initial
        report("live", "PASS", provider_calls=calls, cache_hit=True,
               additional_calls=0, https_sources=len(answer[1]))
    except GeminiFailure as exc:
        report("live", "BLOCKED", category=exc.category, provider_calls=calls,
               breaker_open=not client.search_breaker.available(), repeated=False)
    finally:
        await client.close()


def local(settings):
    import httpx
    import uvicorn
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from app.api import create_app
    from app.gemini import GeminiClient, GeminiFailure
    from scripts.colab_search_smoke_test import fingerprints
    from ui_media import character_image

    class QuotaError(Exception):
        code = 429

    generate = AsyncMock(side_effect=QuotaError("private mock response"))
    fake = GeminiClient(settings, client=NS(aio=NS(models=NS(generate_content=generate))))
    fake.select = AsyncMock(side_effect=GeminiFailure("unavailable"))
    fake.close = AsyncMock()
    before = fingerprints(settings)
    app = create_app(settings, gemini_loader=lambda _: fake)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=18080,
                                         log_level="critical", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    ui = None
    os.environ["WAYANG_API_URL"] = "http://127.0.0.1:18080"
    try:
        deadline = time.monotonic() + 120
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.2)
        if not server.started:
            report("servers", "BLOCKED", reason="backend_startup")
            return
        ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless=true", "--server.address=127.0.0.1", "--server.port=18501",
            "--browser.gatherUsageStats=false"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with httpx.Client(trust_env=False, timeout=105, headers={
            "Authorization": "Bearer " + settings.security.service_token}) as http:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if http.get("http://127.0.0.1:18501/_stcore/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.2)
            else:
                raise AssertionError("Streamlit health")
            report("servers", "PASS", fastapi=18080, streamlit=18501,
                   semantic_ready=app.state.retrieval.semantic is not None)

            def post(path, **kwargs):
                response = http.post("http://127.0.0.1:18080" + path, **kwargs)
                response.raise_for_status()
                return response.json()

            at = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=60).run()
            def send(text):
                next(w for w in at.text_input if w.label == "Pesan").input(text)
                next(b for b in at.button if b.label == "Kirim").click().run()
                assert not at.exception
                return at.session_state["messages"][-1]["data"]

            reply = send("Siapa Jayangrana?")
            assert reply["mode"] == "answer" and reply["sources"] and not reply["llm_used"]
            assert character_image(reply["character"]["gambar_utama"]) and len(at.get("image")) == 1
            report("jayangrana_ui", "PASS", local_evidence=True, official_image=True, provider_calls=0)
            next(b for b in at.button if b.label == "Ya, kenalkan tokoh lain").click().run()
            assert not at.exception
            choices = at.session_state["messages"][-1]["data"]
            assert len(choices["choices"]) == 3
            assert all(c["id"] in range(20) for c in choices["choices"])
            report("ya_ui", "PASS", official_choices=3)
            fallback = send("Sejarah Wayang Sasak")
            assert fallback["mode"] == "fallback_catalog" and len(fallback["catalog"]) == 20
            assert all(any(b.label == c["nama"] for b in at.button) for c in fallback["catalog"])
            rendered = " ".join(w.value for group in (at.text, at.caption) for w in group)
            assert "Gemini" not in rendered and "private mock response" not in rendered
            assert "Basis pengetahuan Wayang AI memuat 20 tokoh" in rendered
            send("Apa itu Wayang Sasak?")
            assert generate.await_count == 1 and not fake.search_breaker.available()
            report("fallback_ui", "PASS", mock_provider_calls=generate.await_count,
                   catalog_entries=20, breaker_next_calls=0)

            item = next(c for c in app.state.retrieval.catalog.items if "munigarim" in c.label.lower())
            path = character_image(item.gambar_utama)
            assert path and Path(path).is_file()
            prediction = post("/predict-image", files={"file": (Path(path).name, Path(path).read_bytes(), "image/jpeg")})
            report("munigarim_primary_inference", "PASS" if prediction["status"] == "accepted" and prediction["prediksi"]["id"] == item.id else "FAIL",
                   image=Path(path).name, inference_status=prediction["status"],
                   predicted_id=(prediction.get("prediksi") or {}).get("id"),
                   confidence=prediction["confidence"], margin=prediction["margin"])
            if prediction["status"] != "accepted" or prediction["prediksi"]["id"] != item.id:
                for sample in sorted(Path(path).parent.glob("*.png")):
                    if sample == Path(path):
                        continue
                    candidate = post("/predict-image", files={"file": (sample.name, sample.read_bytes(), "image/png")})
                    if candidate["status"] == "accepted" and candidate["prediksi"]["id"] == item.id:
                        prediction = candidate
                        report("munigarim_validated_sample", "PASS", image=sample.name,
                               confidence=prediction["confidence"], margin=prediction["margin"])
                        break
                else:
                    report("munigarim_validated_sample", "BLOCKED", reason="no_accepted_sample")
            blank = io.BytesIO()
            Image.new("RGB", (224, 224), "white").save(blank, format="PNG")
            uncertain = post("/predict-image", files={"file": ("uncertain.png", blank.getvalue(), "image/png")})
            assert uncertain["status"] == "uncertain" and uncertain["prediksi"] is None
            for flow, data in [("munigarim_ui", prediction), ("uncertain_ui", uncertain)]:
                image_ui = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=30)
                image_ui.session_state["messages"] = [{"role": "assistant", "kind": "image", "data": data}]
                image_ui.run()
                assert not image_ui.exception
                rendered = " ".join(w.value for group in (image_ui.text, image_ui.markdown, image_ui.warning) for w in group)
                if flow == "uncertain_ui":
                    assert "Dugaan terdekat" in rendered and "Gambar yang kamu kirim adalah tokoh" not in rendered
                else:
                    assert "Munigarim" in rendered
                report(flow, "PASS", real_inference=True, confidence=data["confidence"], margin=data["margin"])
    finally:
        if ui is not None:
            ui.terminate()
            ui.wait(timeout=15)
        server.should_exit = True
        thread.join(timeout=30)
        assert fingerprints(settings) == before
        report("source_artifacts", "PASS", unchanged=True)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.update(OVERRIDES)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logging.disable(logging.CRITICAL)
    from app.config import Settings
    settings = Settings.from_env()
    effective = dict(zip(OVERRIDES, [settings.gemini_timeout_seconds,
        settings.gemini_search_timeout_seconds, settings.gemini_max_attempts,
        settings.gemini_web_max_attempts, settings.security.concurrency,
        settings.wayang_web_cache_ttl_seconds, settings.wayang_web_cache_max_entries,
        settings.chat_enabled]))
    assert list(effective.values()) == [45, 90, 2, 2, 1, 900, 32, True]
    report("configuration", "PASS", values=effective)
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        if args.live:
            asyncio.run(live(settings))
        else:
            local(settings)
    except Exception as exc:
        import traceback
        report("smoke", "FAIL", exception_type=type(exc).__name__,
               frames=[(Path(f.filename).name, f.lineno) for f in traceback.extract_tb(exc.__traceback__)])
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
