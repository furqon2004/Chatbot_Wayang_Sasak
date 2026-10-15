"""Official OpenAI SDK for local evidence selection and narration; no search tools."""
import asyncio
import logging
import json
import random
import time

from .chat_schemas import EvidenceSelection
from .provider import ProviderFailure, REQUEST_METER, SYSTEM_INSTRUCTION, build_prompt, create_openai_client
from .security import Capacity, CircuitBreaker

LOGGER = logging.getLogger(__name__)

# Never accept arbitrary exception names, error codes, messages, headers or IDs.
SAFE_ERROR_CODES = frozenset({"rate_limit_exceeded", "insufficient_quota",
    "invalid_api_key", "model_not_found", "invalid_request_error", "server_error",
    "context_length_exceeded", "billing_hard_limit_reached"})
SAFE_EXCEPTION_TYPES = frozenset({"APITimeoutError", "APIConnectionError", "APIStatusError",
    "BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError",
    "ConflictError", "UnprocessableEntityError", "RateLimitError", "InternalServerError",
    "ProviderFailure", "TimeoutError"})
SAFE_CATEGORIES = frozenset({"timeout", "rate_limit", "invalid_key", "invalid_request",
    "unavailable", "invalid_response", "internal"})


def log_failure(exc, category):
    """Bounded metadata only; unknown values are deliberately discarded."""
    from openai import APIStatusError
    exception_type = type(exc).__name__
    if exception_type not in SAFE_EXCEPTION_TYPES:
        exception_type = "OtherError"
    status, code = None, None
    if isinstance(exc, APIStatusError):
        candidate = exc.status_code
        if type(candidate) is int and 100 <= candidate <= 599:
            status = candidate
        candidate = exc.code
        if type(candidate) is str and candidate in SAFE_ERROR_CODES:
            code = candidate
    LOGGER.warning("RAG request failed: provider=openai category=%s exception_type=%s http_status=%s provider_code=%s",
        category if category in SAFE_CATEGORIES else "internal", exception_type, status, code)


class OpenAIRagClient:
    rag_provider = "openai"

    def __init__(self, settings, client=None, sleep=asyncio.sleep, clock=time.monotonic):
        self.settings = settings
        self.rag_model = settings.llm_model
        self.sleep, self.clock = sleep, clock
        self.breaker = CircuitBreaker(clock=clock)
        self.capacity = Capacity(1)
        self.client = client or create_openai_client(settings)

    async def narrate(self, query, evidence, context):
        return await self.select(query, evidence, narrative_context=context)

    async def select(self, query, evidence, *, narrative_context=None):
        if not self.breaker.available() or not self.capacity.acquire():
            raise ProviderFailure("unavailable")
        try:
            return await asyncio.wait_for(self._select(query, evidence, narrative_context), self.settings.llm_timeout_seconds)
        except asyncio.TimeoutError as exc:
            log_failure(exc, "timeout")
            self.breaker.failed()
            raise ProviderFailure("timeout") from None
        finally:
            self.capacity.release()

    async def _select(self, query, evidence, narrative_context=None):
        from openai import APITimeoutError, APIConnectionError, APIStatusError
        try:
            payload = build_prompt(query, evidence)
        except ValueError:
            raise ProviderFailure("invalid_request") from None
        contract, instruction, name = EvidenceSelection, SYSTEM_INSTRUCTION, "evidence_selection"
        if narrative_context is not None:
            from .narrative import NARRATIVE_INSTRUCTION
            from .chat_schemas import EvidenceNarrative
            contract, instruction, name = EvidenceNarrative, NARRATIVE_INSTRUCTION, "evidence_narrative"
            # Context is resolved/re-read by the backend; no browser prose is trusted.
            payload = json.dumps({"evidence": json.loads(payload), "context": narrative_context}, ensure_ascii=False)
            if len(payload.encode("utf-8")) > 20000:
                raise ProviderFailure("invalid_request")
        schema = contract.model_json_schema()
        # Strict structured output requires every property, including defaults.
        schema["required"] = list(schema["properties"])
        for field in schema["properties"].values():
            field.pop("default", None)
        for attempt in range(self.settings.llm_max_attempts):
            try:
                meter = REQUEST_METER.get()
                if meter is not None:
                    meter["count"] += 1
                response = await self.client.responses.create(
                    model=self.rag_model, instructions=instruction, input=payload,
                    tools=[], store=False, temperature=0, max_output_tokens=2400 if narrative_context is not None else 1024,
                    text={"format": {"type": "json_schema", "name": name,
                                     "strict": True, "schema": schema}})
                raw = response.output_text
                if response.status != "completed" or not isinstance(raw, str) or not raw or len(raw) > 8192:
                    raise ProviderFailure("invalid_response")
                try:
                    selected = contract.model_validate_json(raw)
                    if narrative_context is None:
                        ids = selected.evidence_ids
                        if len(ids) != len(set(ids)) or not set(ids) <= {row["evidence_id"] for row in evidence}:
                            raise ValueError("Invalid evidence")
                except ValueError:
                    raise ProviderFailure("invalid_response") from None
                self.breaker.succeeded()
                return raw
            except Exception as exc:
                if isinstance(exc, ProviderFailure):
                    category = exc.category
                elif isinstance(exc, (APITimeoutError, TimeoutError)):
                    category = "timeout"
                elif isinstance(exc, APIStatusError):
                    code = exc.status_code
                    category = ("rate_limit" if code == 429 else "invalid_key" if code in {401, 403}
                                else "service_unavailable" if code >= 500 else "invalid_request")
                elif isinstance(exc, APIConnectionError):
                    category = "unavailable"
                else:
                    category = "internal"
                log_failure(exc, category)
                if category in {"unavailable", "service_unavailable"} and attempt + 1 < self.settings.llm_max_attempts:
                    await self.sleep(.5 * 2 ** attempt + random.SystemRandom().uniform(0, .25))
                    continue
                self.breaker.failed(immediate=category in {"rate_limit", "invalid_key"})
                raise ProviderFailure(category) from None

    async def close(self):
        await self.client.close()
