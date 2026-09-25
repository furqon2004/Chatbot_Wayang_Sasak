from typing import Annotated, Literal

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
    evidence_ids: list[Annotated[str, Field(strict=True, max_length=100,
        pattern=r"^profile:\d{1,2}:(deskripsi|deskripsi_lengkap|filosofi|ciri_ciri|kata_kunci):\d{1,4}$")]] = Field(max_length=6)


class PromptEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: str = Field(max_length=100)
    id: int = Field(ge=0, le=19)
    label: str = Field(max_length=80)
    nama: str = Field(max_length=120)
    text: str = Field(min_length=1, max_length=500)


class ChatSource(BaseModel):
    document: Literal["wayang_db.json"] = "wayang_db.json"
    id: int | None = None
    label: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


LLMStatus = Literal[
    "used", "not_needed", "not_configured", "timeout", "rate_limit",
    "invalid_key", "invalid_request", "model_unavailable", "unavailable", "invalid_response",
    "insufficient_evidence", "internal",
]


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["answer", "catalog", "fallback_catalog"]
    answer: str = Field(max_length=8000)
    sources: list[ChatSource]
    catalog: list[CatalogItem] | None = Field(default=None, min_length=20, max_length=20)
    suggestions: list[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_status: LLMStatus = "not_needed"
    scope_notice: str | None = Field(default=None, max_length=250)

    @field_validator("catalog")
    @classmethod
    def official_catalog(cls, value):
        if value is not None and [item.id for item in value] != list(range(20)):
            raise ValueError("Katalog tidak valid")
        return value
