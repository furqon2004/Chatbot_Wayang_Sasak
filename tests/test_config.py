import pytest

from app.config import Settings


def test_default_calibrated_threshold(monkeypatch, settings):
    monkeypatch.delenv("WAYANG_SEARCH_THRESHOLD", raising=False)
    assert settings.search_threshold == 0.5590755322575569
    assert Settings.from_env().search_threshold == 0.5590755322575569


def test_threshold_environment_override(monkeypatch):
    monkeypatch.setenv("WAYANG_SEARCH_THRESHOLD", "0.61")
    assert Settings.from_env().search_threshold == 0.61


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1", "abc"])
def test_threshold_environment_validation(monkeypatch, value):
    monkeypatch.setenv("WAYANG_SEARCH_THRESHOLD", value)
    with pytest.raises(ValueError, match="WAYANG_SEARCH_THRESHOLD"):
        Settings.from_env()


def test_embedding_normalization_requires_explicit_boolean(monkeypatch):
    monkeypatch.setenv("WAYANG_NORMALIZE_EMBEDDINGS", "maybe")
    with pytest.raises(ValueError, match="WAYANG_NORMALIZE_EMBEDDINGS"):
        Settings.from_env()


@pytest.mark.parametrize("name,value", [
    ("GEMINI_MODEL", "invalid/private/path"), ("GEMINI_TIMEOUT_SECONDS", "nan"),
    ("GEMINI_TIMEOUT_SECONDS", "0"), ("GEMINI_TIMEOUT_SECONDS", "61"),
    ("GEMINI_MAX_ATTEMPTS", "4"),
])
def test_gemini_invalid_settings_fail_without_echoing_value(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


def test_default_gemini_model(settings):
    assert settings.gemini_model == "gemini-3.6-flash"
    assert Settings.from_env().gemini_model == "gemini-3.6-flash"


def test_gemini_model_environment_override(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-operator-model")
    assert Settings.from_env().gemini_model == "gemini-operator-model"
