from __future__ import annotations

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000, strict=True)
    top_k: int = Field(default=3, ge=1, le=5, strict=True)

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value: str) -> str:
        if not any(char.isalnum() for char in value):
            raise ValueError("Query harus berisi teks")
        return value


class PublicProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deskripsi: str = ""
    deskripsi_lengkap: str = ""
    filosofi: str = ""
    ciri_ciri: list[str] = Field(default_factory=list)
    kata_kunci: list[str] = Field(default_factory=list)
    gambar_utama: str | None = None


class CatalogItem(BaseModel):
    id: int
    label: str
    nama: str
    gambar_utama: str | None = None


class SearchHit(CatalogItem):
    score: float = Field(ge=0, le=1)
    semantic_score: float = Field(ge=0, le=1)
    lexical_score: float = Field(ge=0, le=1)
    evidence: list[str]
    evidence_ids: list[str]
    profil: PublicProfile


class AnswerSearch(BaseModel):
    mode: Literal["answer"] = "answer"
    query: str
    normalized_query: str
    results: list[SearchHit] = Field(min_length=1, max_length=5)


class CatalogSearch(BaseModel):
    mode: Literal["catalog", "fallback_catalog"]
    query: str
    normalized_query: str
    message: str
    catalog: list[CatalogItem] = Field(min_length=20, max_length=20)
    suggestions: list[str]


SearchResponse = Annotated[AnswerSearch | CatalogSearch, Field(discriminator="mode")]
