from __future__ import annotations

import math
import re

from .catalog import Catalog
from .query_normalizer import normalize_query


class RetrievalUnavailable(RuntimeError):
    """Resource retrieval tidak siap; jangan expose exception vendor."""


def collection_metric(collection) -> str:
    metadata = collection.metadata or {}
    configuration = getattr(collection, "configuration", None) or {}
    if callable(configuration):
        configuration = configuration()
    current = (configuration.get("hnsw") or configuration.get("spann") or {}).get("space")
    legacy = metadata.get("hnsw:space")
    if current and legacy and current != legacy:
        raise RetrievalUnavailable("Metric collection bertentangan")
    metric = current or legacy
    if metric not in {"cosine", "l2", "ip"}:
        raise RetrievalUnavailable("Metric collection belum terverifikasi")
    return metric


def distance_score(distance: float, metric: str) -> float:
    if not math.isfinite(distance):
        raise RetrievalUnavailable("Distance tidak finite")
    if metric == "cosine":
        # Chroma cosine distance = 1-cosine. Abaikan kemiripan negatif.
        return min(1.0, max(0.0, 1.0 - distance))
    if metric == "l2":
        if distance < -1e-6:
            raise RetrievalUnavailable("Distance L2 negatif")
        return 1.0 / (1.0 + max(0.0, distance))
    if metric == "ip":
        # IP distance=1-dot; sigmoid menghindari asumsi embedding unit length.
        dot = 1.0 - distance
        return 1.0 / (1.0 + math.exp(-dot)) if dot >= 0 else math.exp(dot) / (1.0 + math.exp(dot))
    raise RetrievalUnavailable("Metric tidak didukung")


def metadata_class_id(metadata: dict, catalog: Catalog) -> int:
    candidates = set()
    names = {normalize_query(item.nama): item.id for item in catalog.items}
    names.update({normalize_query(item.label): item.id for item in catalog.items})
    for key in ("class_id", "label_id", "tokoh_id", "label", "id", "slug", "class_name", "nama", "name", "nama_tokoh", "tokoh"):
        value = metadata.get(key)
        if isinstance(value, bool) or value is None:
            continue
        text = str(value)
        if re.fullmatch(r"\d+", text):
            candidate = int(text)
            if candidate not in catalog.labels:
                raise RetrievalUnavailable("ID metadata di luar katalog")
            candidates.add(candidate)
        elif normalize_query(text) in names:
            candidates.add(names[normalize_query(text)])
    if len(candidates) != 1:
        raise RetrievalUnavailable("Metadata chunk tidak dapat dipetakan secara unik")
    return candidates.pop()


class SemanticSearch:
    def __init__(self, collection, embedder, catalog: Catalog, normalize_embeddings: bool = False):
        self.collection = collection
        self.embedder = embedder
        self.catalog = catalog
        self.normalize_embeddings = normalize_embeddings
        self.metric = collection_metric(collection)
        self.document_count = collection.count()
        if self.document_count <= 0:
            raise RetrievalUnavailable("Collection kosong")
        rows = collection.get(include=["metadatas"])
        self.chunk_classes = {
            chunk_id: metadata_class_id(metadata or {}, catalog)
            for chunk_id, metadata in zip(rows["ids"], rows["metadatas"])
        }
        if len(self.chunk_classes) != self.document_count:
            raise RetrievalUnavailable("Metadata collection tidak lengkap")

    def scores(self, query: str, top_k: int) -> dict[int, float]:
        try:
            embedding = self.embedder.encode(
                [query], normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True, show_progress_bar=False,
            ).tolist()
            result = self.collection.query(
                query_embeddings=embedding, n_results=min(self.document_count, max(20, top_k * 8)),
                include=["distances"],
            )
            scores: dict[int, float] = {}
            ids, distances = result["ids"][0], result["distances"][0]
            if len(ids) != len(distances):
                raise RetrievalUnavailable("Hasil collection tidak konsisten")
            for chunk_id, distance in zip(ids, distances):
                class_id = self.chunk_classes[chunk_id]
                score = distance_score(float(distance), self.metric)
                scores[class_id] = max(score, scores.get(class_id, 0.0))
            return scores
        except RetrievalUnavailable:
            raise
        except Exception:
            raise RetrievalUnavailable("Pencarian semantic gagal") from None
