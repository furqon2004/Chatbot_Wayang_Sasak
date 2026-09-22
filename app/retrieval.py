from __future__ import annotations

from .catalog import Catalog
from .config import DEFAULT_SEARCH_THRESHOLD
from .lexical import lexical_score
from .query_normalizer import catalog_intent, normalize_query, unsupported_instruction
from .schemas import AnswerSearch, SearchHit
from .semantic import RetrievalUnavailable


def hybrid_score(semantic: float, lexical: float) -> float:
    return 0.55 * semantic + 0.45 * lexical


class RetrievalService:
    def __init__(self, catalog: Catalog, semantic=None, threshold: float = DEFAULT_SEARCH_THRESHOLD):
        self.catalog = catalog
        self.semantic = semantic
        self.threshold = threshold

    def rank(self, normalized: str, top_k: int = 3) -> list[SearchHit]:
        if self.semantic is None:
            raise RetrievalUnavailable("Retrieval belum siap")
        semantic_scores = self.semantic.scores(normalized, top_k)
        results = []
        for item in self.catalog.items:
            lexical = lexical_score(normalized, item.id, self.catalog)
            semantic = semantic_scores.get(item.id, 0.0)
            evidence, evidence_ids = self.catalog.evidence(item.id, normalized)
            if not evidence:
                continue
            results.append(SearchHit(
                **item.model_dump(), score=hybrid_score(semantic, lexical),
                semantic_score=semantic, lexical_score=lexical, evidence=evidence,
                evidence_ids=evidence_ids, profil=self.catalog.profiles[item.id],
            ))
        return sorted(results, key=lambda row: (-row.score, row.id))

    def search(self, query: str, top_k: int = 3):
        normalized = normalize_query(query)
        if unsupported_instruction(normalized):
            return self.catalog.response(query, normalized, fallback=True)
        if catalog_intent(normalized):
            return self.catalog.response(query, normalized)
        hits = [hit for hit in self.rank(normalized, top_k) if hit.score >= self.threshold]
        if not hits:
            return self.catalog.response(query, normalized, fallback=True)
        return AnswerSearch(query=query, normalized_query=normalized, results=hits[:top_k])
