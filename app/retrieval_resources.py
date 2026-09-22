"""Chroma asli hanya dibaca untuk snapshot; client tidak pernah membuka path asli."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .catalog import Catalog
from .config import EMBEDDING_MODEL, Settings
from .semantic import RetrievalUnavailable, SemanticSearch


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
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        semantic = SemanticSearch(collection, embedder, catalog, settings.normalize_embeddings)
        return RetrievalResources(semantic, snapshot, client)
    except Exception:
        try:
            _close_client(client)
        finally:
            snapshot.cleanup()
        raise RetrievalUnavailable("Resource retrieval gagal dimuat") from None
