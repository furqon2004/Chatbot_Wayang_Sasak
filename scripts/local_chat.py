"""Local application launcher, distinct from the bounded smoke harness."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_app(settings, *, direct=False, transport=None, **loaders):
    from dataclasses import replace
    import httpx
    from openai import AsyncOpenAI
    from app.api import create_app
    from app.llm import LLMRouter
    from app.openai_rag import OpenAIRagClient
    from app.provider import ProviderClient

    settings = replace(settings, chat_enabled=True, llm_provider="openai",
                       llm_max_attempts=1, web_max_attempts=1, embedding_local_only=True)

    def loader(config):
        if not config.llm_api_key:
            return LLMRouter(config, None, None)
        def sdk(timeout):
            return AsyncOpenAI(api_key=config.llm_api_key, base_url="https://api.openai.com/v1",
                max_retries=0, timeout=timeout,
                http_client=httpx.AsyncClient(trust_env=not direct, verify=True,
                    follow_redirects=False, timeout=timeout, transport=transport))
        # Separate clients and breakers: a rejected local narrative must not stop web.
        return LLMRouter(config,
            OpenAIRagClient(config, client=sdk(config.llm_timeout_seconds)),
            ProviderClient(config, client=sdk(config.web_timeout_seconds)))

    return create_app(settings, provider_loader=loader, **loaders)


def main():
    import argparse
    import logging
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", action="store_true",
        help="Use direct HTTPX connections for this process only; TLS stays verified")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ["WAYANG_ENABLE_CHAT"] = "true"
    from app.config import Settings
    import uvicorn
    logging.basicConfig(level=logging.WARNING)
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger("app.openai_rag").setLevel(logging.WARNING)
    print("Mode aplikasi lokal: narasi dan web aktif; satu attempt per operasi, tanpa retry SDK.")
    print("Bukan smoke: tidak ada batas tiga panggilan per sesi. Setiap permintaan dapat memakai kuota API.")
    try:
        uvicorn.run(build_app(Settings.from_env(), direct=args.direct), host="127.0.0.1",
                    port=8011, access_log=False, log_level="warning")
    except Exception:
        print("Aplikasi lokal gagal dimulai; periksa konfigurasi artefak atau port 8011.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
