import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.retrieval_resources import load_retrieval
from app.semantic import RetrievalUnavailable
from evaluation.mock_search import mock_service
from test_retrieval import FakeCollection


def test_missing_chroma_does_not_create_directory(settings, tmp_path):
    path = tmp_path / "absent"
    with pytest.raises(RetrievalUnavailable):
        load_retrieval(replace(settings, chroma_dir=path), mock_service().catalog)
    assert not path.exists()


def test_persistent_client_only_receives_snapshot(settings, tmp_path, monkeypatch):
    source = tmp_path / "original"
    source.mkdir()
    original = source / "chroma.sqlite3"
    original.write_bytes(b"original read-only fixture")
    calls = []

    class Client:
        def __init__(self, *, path, settings):
            from pathlib import Path
            assert Path(path).resolve() != source.resolve()
            (Path(path) / "chroma.sqlite3").write_bytes(b"client migration only on snapshot")
            calls.append("client")
        def get_collection(self, name, *, embedding_function):
            assert name == "wayang_sasak" and embedding_function is None
            calls.append("get")
            return FakeCollection()
        def close(self):
            calls.append("close")

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=Client))
    monkeypatch.setitem(sys.modules, "chromadb.config", SimpleNamespace(Settings=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *_: object()))
    resources = load_retrieval(replace(settings, chroma_dir=source), mock_service().catalog)
    resources.close()
    assert original.read_bytes() == b"original read-only fixture"
    assert calls == ["client", "get", "close"]
