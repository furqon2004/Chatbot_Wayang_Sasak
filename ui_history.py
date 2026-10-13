"""Ephemeral per-tab history. No disk/server cache, cookies, or query tokens."""
import base64
import re
import time
import uuid
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

TTL = 300
FIELDS = ("messages", "seen_ids", "awaiting_more", "conversation_context", "recommendation_selections")
bridge = components.declare_component("wayang_tab_history", path=str(Path(__file__).parent / "ui_history_component"))


def snapshot(state):
    messages = []
    for message in state.get("messages", []):
        row = dict(message)
        raw = row.pop("image_bytes", None)
        if raw is not None:
            row["image_base64"] = base64.b64encode(raw).decode("ascii")
        messages.append(row)
    return {**{key: state.get(key) for key in FIELDS},
            "recommendation_selections": state.get("recommendation_selections", {}), "messages": messages,
            "last_question": state.get("last_question", 0)}


def restore(payload, now=None):
    now = time.time() if now is None else now
    if not isinstance(payload, dict):
        return None
    stamp = payload.get("last_question")
    if type(stamp) not in (int, float) or not 0 <= now - stamp < TTL:
        return None
    if not isinstance(payload.get("messages"), list):
        return None
    selections = payload.get("recommendation_selections") or {}
    if (not isinstance(selections, dict) or any(
            not isinstance(key, str) or not re.fullmatch(r"message_\d+", key)
            or type(value) is not int or not 0 <= value < 20
            for key, value in selections.items())):
        return None
    result = {key: payload.get(key) for key in FIELDS}
    result["recommendation_selections"] = selections
    messages = []
    for message in payload["messages"]:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            return None
        row = dict(message)
        raw = row.pop("image_base64", None)
        if raw is not None:
            try:
                row["image_bytes"] = base64.b64decode(raw, validate=True)
            except (ValueError, TypeError):
                return None
        messages.append(row)
    result.update(messages=messages, last_question=stamp)
    return result


def clear_history():
    st.session_state.clear()
    st.session_state["history_reset"] = True


def touch():
    st.session_state["last_question"] = time.time()


def sync_history():
    state = st.session_state
    state.setdefault("history_boot", str(uuid.uuid4()))
    if state.get("last_question") and time.time() - state.last_question >= TTL:
        clear_history()
        st.rerun()
    value = bridge(boot=state.history_boot, reset=state.get("history_reset", False),
        snapshot=snapshot(state) if state.get("history_ready") else None,
        ttl=TTL, default=None, key="tab_history_bridge")
    if isinstance(value, dict) and value.get("boot") == state.history_boot:
        if value.get("expired"):
            clear_history()
            st.rerun()
        if not state.get("history_ready"):
            restored = restore(value.get("snapshot"))
            if restored and not state.get("history_reset") and not state.get("messages"):
                state.update(restored)
            state["history_ready"] = True
            state.pop("history_reset", None)
            st.rerun()
        if value.get("storage_error"):
            st.warning("Penyimpanan sementara tab tidak tersedia atau penuh; refresh mungkin menghapus percakapan.")
