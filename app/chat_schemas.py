from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import CatalogItem, SearchRequest


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=1000, strict=True)

    @field_validator("message")
    @classmethod
    def meaningful_message(cls, value: str) -> str:
        return SearchRequest.meaningful_query(value)


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Tidak menerima teks fakta bebas dari LLM. Kosong berarti bukti tak cukup.
    evidence_ids: list[str] = Field(max_length=6)


class ChatSource(BaseModel):
    document: Literal["wayang_db.json"] = "wayang_db.json"
    id: int | None = None
    label: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


LLMStatus = Literal[
    "used", "not_needed", "not_configured", "timeout", "rate_limit",
    "invalid_key", "model_unavailable", "unavailable", "invalid_response",
    "insufficient_evidence", "internal",
]


class ChatResponse(BaseModel):
    mode: Literal["answer", "catalog", "fallback_catalog"]
    answer: str
    sources: list[ChatSource]
    catalog: list[CatalogItem] | None = Field(default=None, min_length=20, max_length=20)
    suggestions: list[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_status: LLMStatus = "not_needed"
