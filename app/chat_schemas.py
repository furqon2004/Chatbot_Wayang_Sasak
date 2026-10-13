from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from urllib.parse import urlsplit
import ipaddress

from .schemas import CatalogItem, SearchRequest


class ConversationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_ids: list[Annotated[int, Field(ge=0, le=19, strict=True)]] = Field(default_factory=list, max_length=2)
    evidence_ids: list[Annotated[str, Field(strict=True, max_length=100,
        pattern=r"^profile:\d{1,2}:(deskripsi|deskripsi_lengkap|filosofi|ciri_ciri|kata_kunci):\d{1,4}$")]] = Field(default_factory=list, max_length=12)
    intent: Literal["profile", "philosophy", "traits", "comparison", "simplify"] = "profile"
    domain: Literal["wayang_sasak"] | None = None

    @model_validator(mode="after")
    def unique_ids(self):
        if len(set(self.subject_ids)) != len(self.subject_ids) or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Konteks duplikat")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=1000, strict=True)
    message_id: str | None = Field(default=None, strict=True, pattern=r"^[0-9a-f]{32}$")
    awaiting_more: bool = False
    explain_more: bool = False
    character_id: Annotated[int, Field(ge=0, le=19, strict=True)] | None = None
    seen_ids: list[Annotated[int, Field(ge=0, le=19, strict=True)]] = Field(default_factory=list, max_length=20)
    conversation_context: ConversationContext | None = None

    @model_validator(mode="after")
    def explanation_identity(self):
        if self.explain_more and self.character_id is None:
            raise ValueError("Penjelasan membutuhkan identitas tokoh")
        return self

    @field_validator("message")
    @classmethod
    def meaningful_message(cls, value: str) -> str:
        return SearchRequest.meaningful_query(value)


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Tidak menerima teks fakta bebas dari LLM. Kosong berarti bukti tak cukup.
    evidence_ids: list[Annotated[str, Field(strict=True, max_length=100,
        pattern=r"^profile:\d{1,2}:(deskripsi|deskripsi_lengkap|filosofi|ciri_ciri|kata_kunci):\d{1,4}$")]] = Field(max_length=6)
    follow_up: Literal[
        "Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?",
        "Mau berkenalan dengan tokoh Wayang Sasak lainnya?",
        "Ingin menjelajahi tokoh Wayang Sasak yang lain?",
    ] = "Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?"


class NarrativeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000, strict=True)
    kind: Literal["explanation", "comparison"]
    subject_ids: list[Annotated[int, Field(ge=0, le=19, strict=True)]] = Field(min_length=1, max_length=2)
    evidence_ids: list[Annotated[str, Field(strict=True, max_length=100,
        pattern=r"^profile:\d{1,2}:(deskripsi|deskripsi_lengkap|filosofi|ciri_ciri|kata_kunci):\d{1,4}$")]] = Field(min_length=1, max_length=6)


class EvidenceNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["supported", "insufficient_evidence"]
    sections: list[NarrativeSection] = Field(max_length=6)


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


class WebSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(max_length=4096)

    @field_validator("url")
    @classmethod
    def public_https(cls, value):
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not host or parsed.username or parsed.password or any(c.isspace() for c in value):
            raise ValueError("URL sumber tidak valid")
        if host == "localhost" or host.endswith((".local", ".localhost")):
            raise ValueError("URL sumber harus publik")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            if not address.is_global:
                raise ValueError("URL sumber harus publik")
        return value


class WebParagraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1800)
    source_ids: list[Annotated[int, Field(ge=0, le=3, strict=True)]] = Field(min_length=1, max_length=4)


class WebSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paragraphs: list[WebParagraph] = Field(min_length=1, max_length=4)


LLMStatus = Literal[
    "used", "not_needed", "not_configured", "timeout", "rate_limit",
    "invalid_key", "invalid_request", "model_unavailable", "service_unavailable", "unavailable", "invalid_response",
    "insufficient_evidence", "internal",
]


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["answer", "catalog", "fallback_catalog", "suggestions", "conversation", "web_answer"]
    answer: str = Field(max_length=8000)
    opening: str | None = Field(default=None, max_length=400)
    sources: list[ChatSource]
    catalog: list[CatalogItem] | None = Field(default=None, min_length=20, max_length=20)
    suggestions: list[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_status: LLMStatus = "not_needed"
    scope_notice: str | None = Field(default=None, max_length=250)
    character: CatalogItem | None = None
    subjects: list[CatalogItem] = Field(default_factory=list, max_length=2)
    conversation_context: ConversationContext | None = None
    narrative: bool = False
    cache_hit: bool = False
    provider_requests: int = Field(default=0, ge=0)
    choices: list[CatalogItem] = Field(default_factory=list, max_length=3)
    follow_up: str | None = Field(default=None, max_length=180)
    web_sources: list[WebSource] = Field(default_factory=list, max_length=20)
    search_suggestions_html: str | None = Field(default=None, max_length=30000)

    @field_validator("catalog")
    @classmethod
    def official_catalog(cls, value):
        if value is not None and [item.id for item in value] != list(range(20)):
            raise ValueError("Katalog tidak valid")
        return value
