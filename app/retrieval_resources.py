"""Chroma asli hanya dibaca untuk snapshot; client tidak pernah membuka path asli."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .catalog import Catalog
from .config import EMBEDDING_MODEL, EXPECTED_CHROMA_DOCUMENT_COUNT, Settings
from .semantic import RetrievalUnavailable, SemanticSearch
from .resources import ResourceValidationError


def _close_client(client):
    close = getattr(client, "close", None)
    if callable(close):
        close()
    elif getattr(client, "_system", None) is not None:
        # Chroma 1.x lama belum mempunyai close(); snapshot selalu client terisolasi.
        client._system.stop()


@dataclass
class RetrievalResources:
    semantic: SemanticSearch
    snapshot: TemporaryDirectory
    client: object

    def close(self):
        try:
            _close_client(self.client)
        finally:
            self.snapshot.cleanup()


def load_retrieval(settings: Settings, catalog: Catalog) -> RetrievalResources:
    if settings.collection_name != "wayang_sasak":
        raise RetrievalUnavailable("Collection tidak diizinkan")
    source = settings.chroma_dir
    if not (source / "chroma.sqlite3").is_file():
        raise RetrievalUnavailable("ChromaDB asli belum tersedia")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise RetrievalUnavailable("Snapshot tidak menerima symlink")
    snapshot = TemporaryDirectory(prefix="wayang-chroma-", ignore_cleanup_errors=True)
    client = None
    try:
        target = Path(snapshot.name) / "database"
        shutil.copytree(source, target)
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(
            path=str(target), settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False)
        )
        # Sengaja get_collection, tidak get_or_create_collection.
        collection = client.get_collection(settings.collection_name, embedding_function=None)
        if (settings.security.production or settings.security.strict_artifacts) and collection.count() != EXPECTED_CHROMA_DOCUMENT_COUNT:
            raise RetrievalUnavailable("Jumlah dokumen tidak sesuai baseline")
        # Model source is application-owned, never supplied by a request.
        model_path = Path(EMBEDDING_MODEL)
        is_local = model_path.is_dir()
        looks_local = (model_path.is_absolute() or str(EMBEDDING_MODEL).startswith((".", "~"))
                       or "\\" in str(EMBEDDING_MODEL) or model_path.exists())
        if looks_local and not is_local:
            raise ResourceValidationError("Model embedding lokal tidak tersedia")
        try:
            embedder = SentenceTransformer(
                str(model_path.resolve()) if is_local else EMBEDDING_MODEL,
                local_files_only=is_local or settings.security.production or settings.embedding_local_only,
                trust_remote_code=False,
                revision=settings.embedding_revision,
            )
        except Exception:
            raise ResourceValidationError("Model embedding tidak tersedia atau tidak valid") from None
        semantic = SemanticSearch(collection, embedder, catalog, settings.normalize_embeddings)
        if (settings.security.production or settings.security.strict_artifacts) and semantic.metric != "cosine":
            raise RetrievalUnavailable("Metric tidak sesuai baseline")
        return RetrievalResources(semantic, snapshot, client)
    except Exception as exc:
        try:
            _close_client(client)
        finally:
            snapshot.cleanup()
        if isinstance(exc, ResourceValidationError):
            raise
        raise RetrievalUnavailable("Resource retrieval gagal dimuat") from None
