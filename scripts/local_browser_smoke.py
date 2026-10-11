"""Loopback-only browser smoke backend; run locally, never with reload/workers."""
import asyncio
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class Budget:
    def __init__(self):
        self.calls = 0
        self.stopped = False
        self.lock = asyncio.Lock()

    async def call(self, create, **kwargs):
        from app.provider import ProviderFailure
        async with self.lock:
            if self.stopped or self.calls >= 3:
                raise ProviderFailure("unavailable")
            self.calls += 1
            try:
                result = await create(**kwargs)
                if result.status != "completed":
                    self.stopped = True
                return result
            except BaseException:
                self.stopped = True
                raise


def build_app(settings, *, transport=None, enable_search=False, search_factory=None, **loaders):
    from dataclasses import replace
    from contextlib import asynccontextmanager
    import httpx
    from openai import AsyncOpenAI
    from app.api import create_app
    from app.openai_rag import OpenAIRagClient
    from app.llm import LLMRouter
    from app.narrative import NARRATIVE_VERSION
    if NARRATIVE_VERSION != "local-narrative-v2" or not settings.llm_api_key:
        raise ValueError("Smoke configuration unavailable")
    settings = replace(settings, chat_enabled=True, llm_provider="openai", llm_model="gpt-4.1-mini",
                       llm_max_attempts=1, web_max_attempts=1,
                       embedding_local_only=True)
    if enable_search and not settings.llm_api_key:
        raise ValueError("Search credential unavailable")
    budget = Budget()
    def loader(config):
        http = httpx.AsyncClient(trust_env=False, verify=True, follow_redirects=False,
                                 timeout=config.llm_timeout_seconds, transport=transport)
        sdk = AsyncOpenAI(api_key=config.llm_api_key, base_url="https://api.openai.com/v1",
                         max_retries=0, timeout=config.llm_timeout_seconds, http_client=http)
        create = sdk.responses.create
        async def bounded(**kwargs):
            return await budget.call(create, **kwargs)
        sdk.responses.create = bounded
        web = None
        if enable_search:
            if search_factory:
                web = search_factory(config)
            else:
                from app.provider import ProviderClient
                web = ProviderClient(config, client=sdk)
        router = LLMRouter(config, OpenAIRagClient(config, client=sdk), web)
        narrate = router.narrate
        async def guarded(*args):
            from app.provider import ProviderFailure
            if budget.stopped or budget.calls >= 3:
                raise ProviderFailure("unavailable")
            return await narrate(*args)
        router.narrate = guarded
        search = router.search
        async def guarded_search(query):
            from app.provider import ProviderFailure
            if budget.stopped:
                raise ProviderFailure("unavailable")
            return await search(query)
        router.search = guarded_search
        return router
    app = create_app(settings, provider_loader=loader, **loaders)
    original_lifespan = app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(active):
        async with original_lifespan(active):
            chat = active.state.chat.chat
            async def observed(*args, **kwargs):
                reply = await chat(*args, **kwargs)
                if reply.llm_status not in {"used", "not_needed"}:
                    budget.stopped = True
                logging.getLogger("smoke").warning(
                    "mode=%s status=%s cache_hit=%s provider_requests=%s sdk_total=%s stopped=%s",
                    reply.mode, reply.llm_status, reply.cache_hit, reply.provider_requests,
                    budget.calls, budget.stopped)
                return reply
            active.state.chat.chat = observed
            yield
    app.router.lifespan_context = lifespan
    @app.get("/smoke-status")
    def status():
        return {"sdk_calls": budget.calls, "sdk_limit": 3, "stopped": budget.stopped,
                "search_enabled": enable_search}
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-search", action="store_true", help="Enable grounded OpenAI web search")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(LLM_MAX_ATTEMPTS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", ANONYMIZED_TELEMETRY="False")
    from dataclasses import replace
    from app.config import Settings
    import uvicorn
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app.openai_rag").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    settings = replace(Settings.from_env(), llm_api_key=os.getenv("LLM_API_KEY", "").strip())
    try:
        app = build_app(settings, enable_search=args.enable_search)
        uvicorn.run(app, host="127.0.0.1", port=8011, access_log=False, log_level="warning")
    except Exception:
        print("Backend smoke tidak dapat dijalankan; periksa konfigurasi lokal atau port 8011.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
