"""UI regressions using mocked HTTP; no model or real user content required."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from ui_client import BackendError, request_api, send_message


@pytest.fixture(autouse=True)
def isolated_ui_config(monkeypatch, tmp_path):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)


def test_image_takes_priority_and_never_sends_text():
    upload = BytesIO(b"test-image")
    upload.name, upload.size = "tokoh.png", 10
    with patch("ui_client.request_api", return_value={"status": "uncertain"}) as api:
        kind, _ = send_message("question", upload)
    assert kind == "image"
    api.assert_called_once_with("/predict-image", image=(b"test-image", "image/png"))


def test_text_routes_to_chat_and_empty_input_does_not_send():
    with patch("ui_client.request_api", return_value={}) as api:
        send_message(" Siapa Jayengrana? ")
        api.assert_called_once_with("/chat", payload={"message": "Siapa Jayengrana?"})
        api.reset_mock()
        with pytest.raises(BackendError):
            send_message(" ")
        api.assert_not_called()


def test_http_errors_do_not_expose_response_body():
    response = httpx.Response(500, text="PRIVATE RESPONSE", request=httpx.Request("POST", "http://test/chat"))
    with patch("httpx.Client.post", return_value=response):
        with pytest.raises(BackendError) as error:
            request_api("/chat", payload={"message": "question"})
    assert "PRIVATE" not in str(error.value)


def app_test():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    return streamlit_testing.AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=15)


def test_native_ui_chat_reset_and_offline():
    at = app_test()
    with patch("ui_client.request_api", return_value={"status": "ok"}), patch(
        "ui_client.send_message", return_value=("chat", {"answer": "Jawaban pengujian"})
    ):
        at.run()
        assert not at.exception
        next(w for w in at.text_input if w.label == "Pertanyaan tentang Wayang Sasak").input("Siapa Jayengrana?")
        next(b for b in at.button if b.label == "KIRIM").click().run()
        assert not at.exception
        assert len(at.session_state["messages"]) == 2
        assert any(t.value == "Jawaban pengujian" for t in at.text)
        next(b for b in at.button if b.label == "Percakapan Baru").click().run()
        assert at.session_state["messages"] == []
    with patch("ui_client.request_api", side_effect=BackendError("Offline")):
        at.run()
        assert not at.exception
        assert any("OFFLINE" in m.value for m in at.markdown)


@pytest.mark.parametrize("status", ["accepted", "uncertain"])
def test_prediction_rendering_including_null_identity(status):
    at = app_test()
    data = {"status": status, "confidence": 0.8, "margin": 0.3,
            "prediksi": {"nama": "Jayengrana"} if status == "accepted" else None,
            "profil": {"deskripsi": "Profil tokoh"} if status == "accepted" else None,
            "message": "Gunakan gambar lebih jelas.",
            "top_3": [{"nama": name, "probability": score} for name, score in
                      [("Jayengrana", 0.8), ("Tokoh B", 0.15), ("Tokoh C", 0.05)]]}
    at.session_state["messages"] = [{"role": "assistant", "kind": "image", "data": data}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
    assert not at.exception
    assert len(at.metric) == (2 if status == "accepted" else 0)
    assert len(at.warning) == (1 if status == "uncertain" else 0)
    assert any("Tokoh C" in item.value for item in at.text)
