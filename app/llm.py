"""OpenAI-only routing; local narration and web search share LLM_API_KEY."""
import logging

from .provider import ProviderFailure, load_provider
from .openai_rag import OpenAIRagClient

LOGGER = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, settings, rag_client, web_client):
        self.settings = settings
        self.rag_client, self.web_client = rag_client, web_client
        self.rag_provider = settings.llm_provider
        self.rag_model = settings.llm_model

    async def select(self, query, evidence):
        if self.rag_client is None:
            raise ProviderFailure("not_configured")
        return await self.rag_client.select(query, evidence)

    async def narrate(self, query, evidence, context):
        if self.rag_provider != "openai" or self.rag_client is None:
            raise ProviderFailure("not_configured")
        return await self.rag_client.narrate(query, evidence, context)

    async def search(self, query):
        if self.web_client is None:
            raise ProviderFailure("not_configured")
        return await self.web_client.search(query)

    async def close(self):
        try:
            if self.rag_client is not None and self.rag_client is not self.web_client:
                await self.rag_client.close()
        finally:
            if self.web_client is not None:
                await self.web_client.close()


def load_llm(settings):
    if not settings.chat_enabled:
        return None
    rag = OpenAIRagClient(settings) if settings.llm_api_key else None
    web = load_provider(settings)
    return LLMRouter(settings, rag, web)
