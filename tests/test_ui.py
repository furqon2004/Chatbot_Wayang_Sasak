"""UI regressions using mocked HTTP; no model or real user content required."""
from io import BytesIO
from copy import deepcopy
import builtins
import io
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

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
        next(w for w in at.text_input if w.label == "Pesan").input("Siapa Jayengrana?")
        next(b for b in at.button if b.label == "Kirim").click().run()
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
            "profil": {"deskripsi": "Deskripsi pendek", "deskripsi_lengkap": "Deskripsi lengkap pilihan",
                       "ciri_ciri": "Mahkota dan busana khas", "filosofi": "Nilai keberanian"}
                      if status == "accepted" else None,
            "message": "Gunakan gambar lebih jelas.",
            "decision": {"accepted": status == "accepted", "reasons": [] if status == "accepted" else ["low_confidence"]},
            "top_3": [{"nama": name, "probability": score} for name, score in
                      [("Jayengrana", 0.8), ("Tokoh B", 0.15), ("Tokoh C", 0.05)]]}
    at.session_state["messages"] = [{"role": "assistant", "kind": "image", "data": data}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
    assert not at.exception
    assert not at.metric
    assert len(at.warning) == (1 if status == "uncertain" else 0)
    rendered = "\n".join(item.value for kind in (at.text, at.markdown, at.caption) for item in kind)
    assert "Top-3" not in rendered
    if status == "accepted":
        assert "Tokoh B" not in rendered and "Tokoh C" not in rendered
        assert "Kemungkinan tokoh:" not in rendered and "Dugaan terdekat:" not in rendered
        assert "Gambar yang kamu kirim adalah tokoh Wayang Sasak **Jayengrana**." in rendered
        assert "Ciri-ciri Jayengrana:" in rendered and "Mahkota dan busana khas" in rendered
        assert "Tentang Jayengrana:" in rendered
        assert rendered.count("Deskripsi lengkap pilihan") == 1
        assert "Deskripsi pendek" not in rendered
        assert "Filosofi:" in rendered and "Nilai keberanian" in rendered
        assert "Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?" in rendered
        assert not at.expander
        assert "Confidence:" in rendered and "Margin:" in rendered
    else:
        assert "Wayang AI belum cukup yakin untuk menetapkan satu tokoh." in at.warning[0].value
        assert "Dugaan terdekat: Jayengrana (80.0%)." in rendered
        assert "Kemungkinan tokoh:" in rendered
        assert "2. Tokoh B (15.0%)" in rendered and "3. Tokoh C (5.0%)" in rendered
        assert "Daftar ini adalah kandidat, bukan hasil identifikasi final." in rendered
        assert "Apakah tokoh pada gambar termasuk salah satu kandidat tersebut?" in rendered
        assert "Gambar yang kamu kirim adalah tokoh" not in rendered
        assert "Tentang Jayengrana:" not in rendered
        assert at.session_state["messages"][0]["data"]["prediksi"] is None
        assert at.session_state["messages"][0]["data"]["profil"] is None
        assert all(word in rendered for word in ("terang", "tajam", "satu tokoh", "seluruh tubuh"))
        assert at.expander[0].label == "Detail keyakinan model"
        assert not at.expander[0].proto.expanded
        assert [item.value for item in at.expander[0].text] == [
            "Confidence: 0.8", "Margin: 0.3", "Alasan rejection: low_confidence"]


@pytest.mark.parametrize("candidate_count", [0, 1, 3, 4])
def test_uncertain_candidates_are_limited_and_response_is_not_mutated(candidate_count):
    at = app_test()
    data = {"status": "uncertain", "prediksi": None, "profil": None,
            "confidence": 0.71, "margin": 0.12,
            "decision": {"accepted": False, "reasons": ["low_confidence", "low_margin"]},
            "top_3": [{"nama": f"Kandidat {i}", "probability": 0.71 / (i + 1)}
                      for i in range(candidate_count)]}
    original = deepcopy(data)
    at.session_state["messages"] = [{"role": "assistant", "kind": "image", "data": data}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
        at.run()
    assert not at.exception
    assert at.session_state["messages"][0]["data"] == original
    texts = [item.value for item in at.text]
    for i in range(min(candidate_count, 3)):
        assert f"{i + 1}. Kandidat {i} ({0.71 / (i + 1):.1%})" in texts
    assert not any("Kandidat 3" in text for text in texts)
    assert "Alasan rejection: low_confidence, low_margin" in texts
    if not candidate_count:
        assert "Daftar kandidat belum tersedia pada respons ini." in texts


def test_identical_descriptions_are_only_rendered_once():
    at = app_test()
    at.session_state["messages"] = [{"role": "assistant", "kind": "image", "data": {
        "status": "accepted", "prediksi": {"nama": "Jayengrana"},
        "profil": {"deskripsi": "Deskripsi identik", "deskripsi_lengkap": "Deskripsi identik"},
        "confidence": 0.9, "margin": 0.3}}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
    assert not at.exception
    texts = [item.value for item in at.text]
    assert texts.count("Deskripsi identik") == 1
    assert "Ciri-ciri Jayengrana:" not in texts and "Filosofi:" not in texts


def test_description_fallback_and_optional_philosophy():
    at = app_test()
    at.session_state["messages"] = [{"role": "assistant", "kind": "image", "data": {
        "status": "accepted", "prediksi": {"nama": "Jayengrana"},
        "profil": {"deskripsi_lengkap": "", "deskripsi": "Deskripsi fallback", "ciri_ciri": "Ciri khas"},
        "confidence": 0.912345, "margin": 0.234567}}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
    assert not at.exception
    texts = [item.value for item in at.text]
    assert texts.count("Deskripsi fallback") == 1
    assert "Filosofi:" not in texts
    assert "Confidence: 0.912345" not in texts and "Margin: 0.234567" not in texts


@pytest.mark.parametrize("status", ["accepted", "uncertain", "error"])
def test_uploaded_image_bytes_stay_in_session_without_disk_writes(status):
    upload = BytesIO()
    Image.new("RGB", (80, 120), "gold").save(upload, format="PNG")
    content = upload.getvalue()
    upload.name, upload.size = "tokoh.png", len(content)
    data = {"status": status, "prediksi": {"nama": "Jayengrana"} if status == "accepted" else None,
            "profil": None, "top_3": [{"nama": "Hidden candidate", "probability": 0.1}]}
    at = app_test()
    with patch("ui_client.request_api", return_value={"status": "ok"}), patch(
        "ui_client.send_message", return_value=("image", data),
        side_effect=BackendError("Layanan belum tersedia") if status == "error" else None
    ), patch("streamlit.file_uploader", return_value=upload):
        at.run()
        real_open, real_io_open = builtins.open, io.open

        def readonly(opener):
            def guarded(file, mode="r", *args, **kwargs):
                assert not any(flag in mode for flag in "wax+"), "Upload flow attempted a disk write"
                return opener(file, mode, *args, **kwargs)
            return guarded

        with patch("builtins.open", side_effect=readonly(real_open)), patch("io.open", side_effect=readonly(real_io_open)):
            next(b for b in at.button if b.label == "Kirim").click().run()
            assert not at.exception
            history = at.session_state["messages"]
            assert type(history[0]["image_bytes"]) is bytes
            assert history[0]["image_bytes"] == content
            assert len(at.get("image")) == 1
            assert at.get("image")[0].proto.imgs[0].caption == "Gambar yang kamu kirim"
            at.run()
            assert not at.exception
            assert len(at.session_state["messages"]) == 2
            assert len(at.get("image")) == 1
            if status == "error":
                assert at.session_state["messages"][1]["kind"] == "error"
            next(b for b in at.button if b.label == "Percakapan Baru").click().run()
            assert not at.exception
            assert at.session_state["messages"] == []
            assert not at.get("image")


def test_invalid_image_preview_does_not_break_conversation():
    at = app_test()
    at.session_state["messages"] = [
        {"role": "user", "text": "", "image_bytes": b"not an image"},
        {"role": "assistant", "kind": "error", "data": "Gambar tidak valid"}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
    assert not at.exception
    assert len(at.warning) == 2


def test_catalog_is_in_main_chat_and_click_opens_character():
    at = app_test()
    with patch("ui_client.request_api", return_value={"catalog": [{"id": 0, "nama": "Jayangrana"}]}), patch(
        "ui_client.send_message", return_value=("chat", {"mode": "answer", "answer": "Penjelasan Jayangrana"})
    ) as send:
        at.run()
        assert not at.sidebar.text_input
        next(b for b in at.button if b.label == "Lihat katalog").click().run()
        assert not at.exception
        assert not any(b.label == "Jayangrana" for b in at.sidebar.button)
        next(b for b in at.button if b.label == "Jayangrana").click().run()
        assert not at.exception
        send.assert_called_once_with("Jelaskan Jayangrana", None, character_id=0)
        assert any(t.value == "Penjelasan Jayangrana" for t in at.text)


@pytest.mark.parametrize("with_suggestions", [True, False])
def test_catalog_button_renders_twenty_profileless_cards_without_semantic_search(fake_resources, with_suggestions):
    from app.catalog import Catalog
    from app.retrieval import RetrievalService
    from unittest.mock import Mock

    semantic = Mock()
    semantic.scores.side_effect = AssertionError("Catalog must not use semantic search")
    service = RetrievalService(Catalog(fake_resources.labels, fake_resources.characters), semantic)
    at = app_test()

    def respond(path, **kwargs):
        if path == "/health":
            return {"status": "ok"}
        assert path == "/search"
        assert kwargs == {"payload": {"query": "daftar tokoh", "top_k": 3}}
        response = service.search(**kwargs["payload"]).model_dump()
        assert response["mode"] == "catalog"
        assert len(response["catalog"]) == 20
        assert all("profil" not in item for item in response["catalog"])
        if not with_suggestions:
            response.pop("suggestions")
        return response

    with patch("ui_client.request_api", side_effect=respond) as api:
        at.run()
        next(b for b in at.button if b.label == "Lihat katalog").click().run()
    assert not at.exception
    api.assert_any_call("/search", payload={"query": "daftar tokoh", "top_k": 3})
    semantic.scores.assert_not_called()
    labels = [button.label for button in at.button]
    for item in service.catalog.items:
        assert labels.count(item.nama) == 1
        assert item.nama not in [button.label for button in at.sidebar.button]
    assert not at.sidebar.text_input


@pytest.mark.parametrize("status", [401, 403, 404, 422, 429, 500, 502, 503, 504, 418])
def test_catalog_http_failure_is_safe_and_clears_stale_results(status):
    at = app_test()
    at.session_state["search_result"] = {"mode": "catalog", "catalog": [{"nama": "Stale result"}]}
    response = httpx.Response(status, text="PRIVATE_BODY_TRACEBACK", headers={"X-Private": "PRIVATE_HEADER"},
                              request=httpx.Request("POST", "http://test/search"))
    with patch("httpx.Client.get", return_value=httpx.Response(200, json={"status": "ok"},
               request=httpx.Request("GET", "http://test/health"))), patch("httpx.Client.post", return_value=response):
        at.run()
        next(b for b in at.button if b.label == "Lihat katalog").click().run()
    assert not at.exception
    assert not at.sidebar.warning
    assert len(at.warning) == 1
    warning = at.warning[0].value
    assert "PRIVATE" not in warning and "Traceback" not in warning
    assert "search_result" not in at.session_state
    assert not any(item.value == "Stale result" for item in at.sidebar.text)
    if status >= 500 or status == 418:
        assert f"HTTP {status}" in warning


@pytest.mark.parametrize("body", ["not-json", "[]", "null"])
def test_invalid_json_has_distinct_safe_error(body):
    response = httpx.Response(200, text=body, request=httpx.Request("POST", "http://test/search"))
    with patch("httpx.Client.post", return_value=response), pytest.raises(BackendError) as error:
        request_api("/search", payload={"query": "daftar tokoh", "top_k": 3})
    assert str(error.value) == "Format respons backend tidak valid. Silakan coba lagi."


@pytest.mark.parametrize("exception,expected", [
    (httpx.ConnectError("PRIVATE CONNECTION"), "Koneksi ke backend gagal."),
    (httpx.ReadTimeout("PRIVATE TIMEOUT"), "Waktu tunggu backend habis."),
])
def test_transport_errors_are_distinct_and_safe(exception, expected):
    with patch("httpx.Client.post", side_effect=exception), pytest.raises(BackendError) as error:
        request_api("/search", payload={"query": "daftar tokoh", "top_k": 3})
    assert str(error.value).startswith(expected)
    assert "PRIVATE" not in str(error.value)


def test_frontend_is_utf8_without_mojibake():
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_bytes().decode("utf-8")
    for character in ("·", "◈", "◇", "◆", "…"):
        for encoding in ("latin-1", "cp1252"):
            assert character.encode("utf-8").decode(encoding) not in source
    assert "\ufffd" not in source
    assert "Top-3 kandidat" not in source


def test_composer_combines_text_and_attachment_and_locks_upload_when_typing():
    at = app_test()
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
        assert not at.exception and not at.tabs
        assert len(at.text_input) == 1
        assert len(at.get("file_uploader")) == 1
        assert sum(b.label == "Kirim" for b in at.button) == 1
        at.text_input(key="text_question_0").input("Jelaskan Umar Maya").run()
        assert at.get("file_uploader")[0].proto.disabled
        next(b for b in at.button if b.label == "Hapus input").click().run()
        assert not at.get("file_uploader")[0].proto.disabled
        assert at.text_input[0].value == ""


def test_empty_composer_never_sends_or_adds_history():
    at = app_test()
    with patch("ui_client.request_api", return_value={"status": "ok"}), patch("ui_client.send_message") as send:
        at.run()
        next(b for b in at.button if b.label == "Kirim").click().run()
    assert not at.exception
    send.assert_not_called()
    assert at.session_state["messages"] == []
    assert len(at.warning) == 1


def test_unified_composer_clears_upload_before_next_text_message():
    upload = BytesIO()
    Image.new("RGB", (80, 120), "gold").save(upload, format="PNG")
    upload.name, upload.size = "tokoh.png", len(upload.getvalue())
    at = app_test()
    with patch("ui_client.request_api", return_value={"status": "ok"}), patch(
        "ui_client.send_message", side_effect=[("image", {"status": "uncertain", "prediksi": None}), ("chat", {"answer": "Penjelasan"})]
    ) as send:
        with patch("streamlit.file_uploader", return_value=upload):
            at.run()
            next(b for b in at.button if b.label == "Kirim").click().run()
        assert not at.exception
        send.assert_called_once_with("", upload)
        at.run()
        assert at.text_input[0].value == "" and not at.text_input[0].disabled
        at.text_input[0].input("Jelaskan Umar Maya")
        next(b for b in at.button if b.label == "Kirim").click().run()
        send.assert_called_with("Jelaskan Umar Maya", None)
        assert not at.exception
        history = at.session_state["messages"]
        assert history[0]["image_bytes"] == upload.getvalue()
        assert "image_bytes" not in history[2]
        assert len(at.get("image")) == 1


def test_composer_rejects_simultaneous_text_and_image():
    upload = BytesIO(b"image")
    upload.name, upload.size = "tokoh.png", 5
    at = app_test()
    with patch("ui_client.request_api", return_value={"status": "ok"}), patch("ui_client.send_message") as send:
        at.run()
        at.text_input[0].input("Jelaskan Umar Maya")
        with patch("streamlit.file_uploader", return_value=upload):
            next(b for b in at.button if b.label == "Kirim").click().run()
        assert not at.exception
        assert any("Pilih satu jenis pesan" in w.value for w in at.warning)
        assert at.session_state["messages"] == []
        send.assert_not_called()


def test_selected_attachment_locks_text_and_clear_unlocks_it():
    upload = BytesIO(b"image")
    upload.name, upload.size = "tokoh.png", 5
    at = app_test()
    at.session_state["image_upload_0"] = upload
    with patch("ui_client.request_api", return_value={"status": "ok"}), patch(
        "streamlit.file_uploader", side_effect=lambda *args, **kwargs: upload if kwargs["key"] == "image_upload_0" else None
    ):
        at.run()
        assert not at.exception and at.text_input[0].disabled
        next(b for b in at.button if b.label == "Hapus input").click().run()
        assert not at.exception and not at.text_input[0].disabled
        assert at.text_input[0].value == ""


def test_text_character_has_one_image_and_yes_opens_clickable_choices(tmp_path, monkeypatch):
    root = tmp_path / "dataset_asli"
    image = root / "umar_maya" / "1.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (40, 60), "gold").save(image)
    monkeypatch.setenv("WAYANG_IMAGE_ROOT", str(root))
    at = app_test()
    reply = {"mode": "answer", "answer": "Umar Maya adalah penasihat.",
             "character": {"id": 1, "nama": "Umar Maya", "gambar_utama": "dataset_asli/umar_maya/1.png"},
             "follow_up": "Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?"}
    choices = {"mode": "suggestions", "answer": "Pilih tokoh", "choices": [
        {"id": i, "nama": name} for i, name in [(0, "Jayangrana"), (2, "Maktal"), (19, "Gunungan")]]}
    with patch("ui_client.request_api", return_value={"status": "ok"}), patch(
        "ui_client.send_message", side_effect=[("chat", reply), ("chat", choices), ("chat", reply)]
    ) as send:
        at.run()
        at.text_input(key="text_question_0").input("Umar Maya")
        next(b for b in at.button if b.label == "Kirim").click().run()
        assert not at.exception
        assert len(at.get("image")) == 1
        assert at.session_state["seen_ids"] == [1]
        at.text_input(key="text_question_1").input("Ya")
        next(b for b in at.button if b.label == "Kirim").click().run()
        assert not at.exception
        send.assert_called_with("Ya", None, awaiting_more=True, seen_ids=[1])
        assert len(at.get("image")) == 1
        for name in ["Jayangrana", "Maktal", "Gunungan"]:
            assert sum(b.label == name for b in at.button) == 1
        next(b for b in at.button if b.label == "Maktal").click().run()
        send.assert_called_with("Jelaskan Maktal", None, character_id=2)
        assert not at.exception
        next(b for b in at.button if b.label == "Percakapan Baru").click().run()
        assert not at.session_state["seen_ids"] and not at.session_state["awaiting_more"]


def test_reference_image_rejects_traversal_and_missing_files(tmp_path, monkeypatch):
    from ui_media import character_image
    monkeypatch.setenv("WAYANG_IMAGE_ROOT", str(tmp_path))
    assert character_image("dataset_asli/../private.png") is None
    assert character_image("C:/private.png") is None
    assert character_image("dataset_asli/missing.png") is None
    assert character_image("https://example.com/image.png?token=secret") is None


def test_web_answer_shows_citations_without_unrelated_character_image():
    at = app_test()
    at.session_state["messages"] = [{"role": "assistant", "kind": "chat", "data": {
        "mode": "web_answer", "answer": "Ringkasan sejarah [1]", "sources": [],
        "web_sources": [{"title": "Museum", "url": "https://example.org/museum"}],
        "search_suggestions_html": "<div>Google Search</div>",
        "scope_notice": "Aplikasi memuat 20 tokoh, bukan jumlah universal.",
        "follow_up": "Ingin mengenal tokoh lainnya?"}}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
    assert not at.exception and not at.get("image")
    assert len(at.get("link_button")) == 1
    assert at.get("link_button")[0].proto.url == "https://example.org/museum"
    assert any(t.value == "Ringkasan sejarah [1]" for t in at.text)


def test_web_failure_renders_catalog_without_provider_details():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.chat import ChatService
    from app.gemini import GeminiFailure
    from evaluation.mock_search import mock_service
    reply = asyncio.run(ChatService(mock_service(), SimpleNamespace(
        search=AsyncMock(side_effect=GeminiFailure("rate_limit")))).chat("Sejarah Wayang Sasak"))
    at = app_test()
    at.session_state["messages"] = [{"role": "assistant", "kind": "chat", "data": reply.model_dump()}]
    with patch("ui_client.request_api", return_value={"status": "ok"}):
        at.run()
    assert not at.exception
    rendered = "\n".join(item.value for kind in (at.text, at.markdown, at.caption) for item in kind)
    assert "Informasi internet belum dapat diverifikasi" in rendered
    assert "Gemini" not in rendered and "kuota" not in rendered.casefold()
    for item in reply.catalog:
        assert any(item.nama in button.label for button in at.button)
