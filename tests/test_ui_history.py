from ui_history import snapshot, restore


def test_refresh_roundtrip_preserves_context_and_image():
    state = {"messages": [{"role": "user", "kind": "image", "image_bytes": b"fixture"}],
             "seen_ids": [0], "awaiting_more": True,
             "conversation_context": {"subject_ids": [0], "domain": "wayang_sasak"},
             "last_question": 1000}
    assert restore(snapshot(state), now=1299) == state


def test_expiry_is_based_on_last_question_not_refresh():
    state = {"messages": [], "last_question": 1000}
    saved = snapshot(state)
    assert restore(saved, now=1299)["last_question"] == 1000
    assert restore(saved, now=1300) is None
    assert restore(saved, now=999) is None


def test_invalid_browser_data_is_not_restored():
    assert restore({"last_question": "invalid"}) is None
    assert restore({"last_question": 1000, "messages": [None]}, now=1001) is None
    assert restore({"last_question": 1000, "messages": [
        {"role": "user", "image_base64": "invalid!"}]}, now=1001) is None


def test_new_chat_clears_all_server_history(monkeypatch):
    import ui_history
    state = {"messages": [{"role": "user", "text": "private"}],
             "conversation_context": {"subject_ids": [0]}, "last_question": 1000}
    monkeypatch.setattr(ui_history.st, "session_state", state)
    ui_history.clear_history()
    assert state == {"history_reset": True}
