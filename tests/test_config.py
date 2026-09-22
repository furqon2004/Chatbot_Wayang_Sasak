import pytest

from app.config import Settings


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1", "abc"])
def test_threshold_environment_validation(monkeypatch, value):
    monkeypatch.setenv("WAYANG_SEARCH_THRESHOLD", value)
    with pytest.raises(ValueError, match="WAYANG_SEARCH_THRESHOLD"):
        Settings.from_env()


def test_embedding_normalization_requires_explicit_boolean(monkeypatch):
    monkeypatch.setenv("WAYANG_NORMALIZE_EMBEDDINGS", "maybe")
    with pytest.raises(ValueError, match="WAYANG_NORMALIZE_EMBEDDINGS"):
        Settings.from_env()
