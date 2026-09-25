import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import create_app
from app.config import EXPECTED_CHROMA_DOCUMENT_COUNT, EMBEDDING_MODEL, SecuritySettings, Settings
from app.resources import ResourceValidationError
from app.retrieval_resources import load_retrieval
from app.schemas import SearchRequest
from app.chat_schemas import ChatRequest
from evaluation.mock_search import mock_service


@pytest.fixture
def embedding_runtime(settings, tmp_path, monkeypatch):
    source = tmp_path / 'chroma'
    source.mkdir()
    (source / 'chroma.sqlite3').write_bytes(b'fixture')
    calls, closed = [], []
    collection = SimpleNamespace(count=lambda: EXPECTED_CHROMA_DOCUMENT_COUNT, metadata={'hnsw:space': 'cosine'},
        get=lambda **k: {'ids': [str(i) for i in range(EXPECTED_CHROMA_DOCUMENT_COUNT)], 'metadatas': [{'label': '0'}] * EXPECTED_CHROMA_DOCUMENT_COUNT})
    client = SimpleNamespace(get_collection=lambda *a, **k: collection,
                             close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, 'chromadb', SimpleNamespace(PersistentClient=lambda **k: client))
    monkeypatch.setitem(sys.modules, 'chromadb.config', SimpleNamespace(Settings=lambda **k: k))
    def loader(model, **kwargs):
        calls.append((model, kwargs))
        return object()
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=loader))
    return replace(settings, chroma_dir=source), calls, closed


@pytest.mark.parametrize('production,configured', [(False, False), (False, True), (True, False), (True, True)])
def test_embedding_effective_policy(embedding_runtime, production, configured):
    settings, calls, closed = embedding_runtime
    settings = replace(settings, security=SecuritySettings(production=production, service_token='fixture' * 8),
                       embedding_local_only=configured, embedding_revision='operator-revision')
    resources = load_retrieval(settings, mock_service().catalog)
    resources.close()
    assert calls == [(EMBEDDING_MODEL, dict(local_files_only=production or configured,
                                          trust_remote_code=False, revision='operator-revision'))]
    assert closed == [True]


def test_embedding_environment(monkeypatch):
    assert Settings.from_env().embedding_revision is None
    assert Settings.from_env().embedding_local_only is False
    monkeypatch.setenv('WAYANG_EMBEDDING_LOCAL_ONLY', 'true')
    monkeypatch.setenv('WAYANG_EMBEDDING_REVISION', ' operator-commit ')
    assert Settings.from_env().embedding_local_only is True
    assert Settings.from_env().embedding_revision == 'operator-commit'


@pytest.mark.parametrize('exists', [True, False])
def test_local_directory_never_falls_back_to_hub(embedding_runtime, monkeypatch, tmp_path, exists):
    settings, calls, closed = embedding_runtime
    path = tmp_path / 'local-model'
    if exists:
        path.mkdir()
    monkeypatch.setattr('app.retrieval_resources.EMBEDDING_MODEL', str(path))
    if exists:
        load_retrieval(settings, mock_service().catalog).close()
        assert calls[0][0] == str(path.resolve())
        assert calls[0][1]['local_files_only'] is True
        assert calls[0][1]['trust_remote_code'] is False
    else:
        with pytest.raises(ResourceValidationError):
            load_retrieval(settings, mock_service().catalog)
        assert calls == []
    assert closed == [True]


def test_production_cache_miss_aborts_startup_safely(embedding_runtime, monkeypatch, fake_resources, caplog):
    settings, calls, closed = embedding_runtime
    settings = replace(settings, security=SecuritySettings(production=True, service_token='fixture' * 8))
    secret = 'private/path https://hub.invalid token=synthetic-sensitive-value'
    def unavailable(model, **kwargs):
        calls.append(kwargs)
        raise OSError(secret)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=unavailable))
    app = create_app(settings, resource_loader=lambda _: fake_resources)
    with pytest.raises(ResourceValidationError) as error:
        with TestClient(app):
            pytest.fail('Production startup must fail')
    assert len(calls) == 1 and calls[0]['local_files_only'] is True
    assert error.value.__suppress_context__
    assert secret not in str(error.value) + caplog.text
    assert 'private/path' not in str(error.value) + caplog.text
    assert not app.state.ready and app.state.resources is None
    assert closed == [True]


@pytest.mark.parametrize('schema,field', [(s, f) for s in (SearchRequest, ChatRequest)
    for f in ('model_identifier', 'model_name_or_path', 'local_files_only', 'revision', 'trust_remote_code', 'cache_folder')])
def test_requests_cannot_configure_embedding(schema, field):
    payload = {'query' if schema is SearchRequest else 'message': 'Wayang Sasak'}
    with pytest.raises(ValidationError):
        schema.model_validate({**payload, field: 'attacker'})


def test_query_text_cannot_configure_loader(embedding_runtime):
    settings, calls, _ = embedding_runtime
    SearchRequest(query='Wayang Sasak model_name_or_path=attacker revision=evil local_files_only=false')
    load_retrieval(settings, mock_service().catalog).close()
    assert calls == [(EMBEDDING_MODEL, dict(local_files_only=False, trust_remote_code=False, revision=None))]
