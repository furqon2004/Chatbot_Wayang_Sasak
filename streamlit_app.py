"""Native Streamlit presentation. User content lives only in session memory."""
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import re
import uuid
import streamlit as st
import streamlit.components.v1 as components

from ui_client import BackendError, request_api, send_message, web_answer_markdown
from ui_media import character_image
from ui_history import sync_history, clear_history, touch

load_dotenv()
st.set_page_config(page_title="Wayang AI · Wayang Sasak", page_icon="◈",
                   layout="centered", initial_sidebar_state="collapsed")


def queue_action(action, *, clear_text=False, clear_image=False):
    if st.session_state.get("processing") or st.session_state.get("pending_action"):
        return False
    action["message_id"] = uuid.uuid4().hex
    st.session_state["pending_action"] = action
    st.session_state["processing"] = True
    if clear_text:
        st.session_state.text_generation += 1
    if clear_image:
        st.session_state.image_generation += 1
    return True


def profile(value):
    description = value.get("deskripsi_lengkap") or value.get("deskripsi")
    for content, title in ((description, "Tentang tokoh"),
                           (value.get("filosofi"), "Filosofi"),
                           (value.get("ciri_ciri"), "Ciri-ciri")):
        if content:
            st.caption(title)
            st.text(content)
    if value.get("kata_kunci"):
        st.text(" · ".join(value["kata_kunci"]))


def catalog(items, key, recommendation=False):
    if not items:
        return
    selected = st.session_state.setdefault("recommendation_selections", {}).get(key)
    with st.container(key=f"catalog_grid_{key}"):
        for start in range(0, len(items), 3):
            row = items[start:start + 3]
            columns = st.columns(3)
            for column, (index, item) in zip(columns, enumerate(row, start)):
                name = item.get("nama", item.get("label", "Tokoh"))
                with column:
                    with st.container(border=True, key=f"catalog_card_{key}_{index}"):
                        source = character_image(item.get("gambar_utama"))
                        with st.container(key=f"catalog_image_{key}_{index}", horizontal_alignment="center"):
                            if source:
                                try:
                                    st.image(source, width="content")
                                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                                    st.markdown("🎭")
                            else:
                                st.markdown("🎭")
                        if recommendation and selected is not None:
                            st.text(f"{name} · Dipilih" if item.get("id") == selected else name)
                        elif st.button(name, key=f"{key}_choice_{index}", width="stretch",
                                       disabled=st.session_state.processing):
                            if recommendation:
                                st.session_state["recommendation_selections"][key] = item.get("id")
                            queue_action({"kind": "question", "question": "Jelaskan " + name,
                                          "character_id": item.get("id")})
                            st.rerun()


def reference_image(item):
    source = character_image(item.get("gambar_utama"))
    if source:
        try:
            st.image(source, width=420, caption=item.get("nama", "Tokoh Wayang Sasak"))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.caption("Gambar tokoh belum dapat ditampilkan.")
    else:
        st.caption("Gambar tokoh belum tersedia. File gambar dari dataset perlu disediakan.")


def render(kind, data, key="response"):
    if kind == "error":
        st.warning(data, icon=":material/info:")
    elif kind == "image":
        identity = data.get("prediksi")
        if data.get("status") == "accepted" and isinstance(identity, dict):
            name = identity.get("nama") or identity.get("label") or "Tokoh"
            # Escape Markdown metacharacters in backend text before bolding it.
            escaped_name = re.sub(r"([\\`*_{}\[\]()<>#+.!|~-])", r"\\\1", name)
            st.markdown(f"Gambar yang kamu kirim adalah tokoh Wayang Sasak **{escaped_name}**.")
            details = data.get("profil") or {}
            st.caption(f"Label: {identity.get('label') or name}")
            st.text(f"Confidence: {data.get('confidence', 0):.1%} ? Margin: {data.get('margin', 0):.1%}")
            if details.get("ciri_ciri"):
                st.text(f"Ciri-ciri {name}:")
                st.text(details["ciri_ciri"])
            st.text(f"Tentang {name}:")
            st.text(details.get("deskripsi_lengkap") or details.get("deskripsi")
                    or "Deskripsi belum tersedia.")
            if details.get("filosofi"):
                st.text("Filosofi:")
                st.text(details["filosofi"])
            st.text("Apakah Anda ingin tahu tokoh Wayang Sasak yang lain?")
            if isinstance(identity.get("id"), int) and st.button(
                "Jelaskan lebih lanjut", key=f"{key}_explain", disabled=st.session_state.processing
            ):
                queue_action({"kind": "question", "question": "Jelaskan " + name,
                              "character_id": identity["id"], "explain_more": True})
                st.rerun()
        elif data.get("status") == "uncertain":
            st.warning("Wayang AI belum cukup yakin untuk menetapkan satu tokoh.", icon=":material/info:")
            candidates = (data.get("top_3") or [])[:3]
            if candidates:
                nearest = candidates[0]
                st.text(f"Dugaan terdekat: {nearest.get('nama') or nearest.get('label') or 'Tokoh'} "
                        f"({nearest['probability']:.1%}).")
                st.text("Kemungkinan tokoh:")
                for number, candidate in enumerate(candidates, start=1):
                    st.text(f"{number}. {candidate.get('nama') or candidate.get('label') or 'Tokoh'} "
                            f"({candidate['probability']:.1%})")
                st.caption("Daftar ini adalah kandidat, bukan hasil identifikasi final.")
                st.text("Apakah tokoh pada gambar termasuk salah satu kandidat tersebut?")
            else:
                st.text("Daftar kandidat belum tersedia pada respons ini.")
            st.text("Unggah gambar yang lebih terang dan tajam, menampilkan satu tokoh dengan seluruh tubuh terlihat.")
            with st.expander("Detail keyakinan model", expanded=False):
                st.text(f"Confidence: {data.get('confidence', 'Tidak tersedia')}")
                st.text(f"Margin: {data.get('margin', 'Tidak tersedia')}")
                reasons = (data.get("decision") or {}).get("reasons") or []
                st.text("Alasan rejection: " + (", ".join(reasons) if reasons else "Tidak tersedia"))
        else:
            st.warning("Respons pengenalan gambar tidak valid. Silakan coba lagi.", icon=":material/info:")
    elif kind == "chat":
        if data.get("scope_notice"):
            st.caption(data["scope_notice"])
        opening = data.get("opening")
        if opening:
            st.text(opening)
        if len(data.get("subjects", [])) == 2:
            st.caption("Tokoh: " + " dan ".join(subject["nama"] for subject in data["subjects"]))
        if data.get("character"):
            reference_image(data["character"])
        answer = data.get("answer", "Belum ada jawaban.")
        if opening and answer.startswith(opening + "\n\n"):
            answer = answer[len(opening) + 2:]
        if data.get("web_sources"):
            st.markdown(web_answer_markdown(answer, data["web_sources"]))
        else:
            st.markdown(answer)
        if data.get("web_sources"):
            st.caption("Sumber internet")
            for number, source in enumerate(data["web_sources"], start=1):
                st.link_button(f"[{number}] {source['title']}", source["url"])
        if data.get("mode") == "answer" and data.get("llm_status") in {
            "rate_limit", "unavailable", "timeout", "invalid_key", "model_unavailable", "internal", "invalid_response"
        }:
            st.caption("Jawaban profil menggunakan basis pengetahuan lokal.")
        choices = data.get("choices") or []
        catalog(choices or data.get("catalog") or [], key, recommendation=bool(choices))
        if data.get("follow_up"):
            st.text(data["follow_up"])
        if data.get("sources"):
            with st.expander("Sumber jawaban"):
                for number, source in enumerate(data["sources"], start=1):
                    st.text((f"[{number}] " if data.get("narrative") else "") + f"{source.get('document', 'Basis pengetahuan')} · {source.get('label') or 'Katalog Wayang Sasak'}")
    elif kind == "search":
        if data.get("message"):
            st.text(data["message"])
        catalog(data.get("results") or data.get("catalog") or [], key)


sync_history()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("image_generation", 0)
st.session_state.setdefault("text_generation", 0)
st.session_state.setdefault("processing", False)
st.session_state.setdefault("seen_ids", [])
st.session_state.setdefault("awaiting_more", False)
st.session_state.setdefault("conversation_context", None)
st.session_state.setdefault("recommendation_selections", {})

with st.sidebar:
    st.subheader("Wayang AI")
    if st.button("Percakapan Baru", icon=":material/add_comment:", width="stretch",
                 disabled=st.session_state.processing):
        clear_history()
        st.rerun()
    st.caption("Percakapan sementara di tab ini, tetap ada saat refresh. Dihapus setelah 5 menit tanpa pertanyaan atau saat Percakapan Baru.")


@st.fragment(run_every="30s")
def header():
    import time
    if (st.session_state.get("last_question")
            and time.time() - st.session_state.last_question >= 300):
        clear_history()
        st.rerun(scope="app")
    with st.container(border=True, horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center"):
        st.markdown("### :yellow[◈] WAYANG AI", width="content")
        try:
            request_api("/health")
            st.badge("ONLINE", color="green", icon=":material/check_circle:")
        except BackendError:
            st.badge("OFFLINE", color="gray", icon=":material/cloud_off:")


header()
st.html("""
<style>
[data-testid="stMainBlockContainer"] { max-width: 52rem; padding-bottom: 10rem; }
[data-testid="stChatMessage"] { max-width: 52rem; margin-inline: auto; padding-block: .3rem; }
[class*="st-key-chat_user_bubble_"] {
  margin-left: auto; width: fit-content; max-width: 82%;
  padding: .65rem .9rem; border-radius: 1.1rem;
  background: #382752; color: #f5f1e8;
}
[class*="st-key-chat_assistant_bubble_"] {
  max-width: 100%; padding: .45rem .55rem;
  border-radius: 1rem; background: rgba(35, 25, 55, .55);
}
[class*="st-key-catalog_grid_"] [data-testid="stHorizontalBlock"] {
  align-items: stretch; gap: .75rem;
}
[class*="st-key-catalog_grid_"] [data-testid="column"] {
  min-width: 0 !important;
}
[class*="st-key-catalog_card_"] { width: 100%; height: 19rem; box-sizing: border-box; }
[class*="st-key-catalog_card_"] [data-testid="stVerticalBlock"] { height: 100%; }
[class*="st-key-catalog_image_"] {
  height: 12rem; min-height: 12rem; display: flex; align-items: center;
  justify-content: center; overflow: hidden; border-radius: .5rem;
}
[class*="st-key-catalog_image_"] [data-testid="stImage"] {
  height: 100%; width: 100%; display: flex; align-items: center; justify-content: center;
}
[class*="st-key-catalog_image_"] img {
  height: 10.5rem !important;
  max-width: calc(100% - 1.5rem) !important;
  width: auto !important;
  object-fit: contain; object-position: center;
}
[class*="st-key-catalog_image_"] p { margin: 0; font-size: 2.5rem; }
[class*="st-key-catalog_card_"] button {
  width: 100%; height: 3rem; min-height: 3rem; box-sizing: border-box;
  white-space: normal; word-break: normal; overflow-wrap: break-word; text-align: center;
}
@media (max-width: 768px) {
  [class*="st-key-chat_user_bubble_"] { max-width: 92%; }
}
</style>
""")


def render_user_message(message, index):
    with st.chat_message("user"):
        with st.container(key=f"chat_user_bubble_{index}"):
            if message.get("text"):
                st.text(message["text"])
            if message.get("image_bytes"):
                try:
                    st.image(message["image_bytes"], width=420, caption="Gambar yang kamu kirim")
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    st.warning("Pratinjau gambar tidak dapat ditampilkan. Gunakan gambar JPG, PNG, atau WebP yang valid.")


message_count = len(st.session_state.messages)
if message_count and st.session_state.get("last_scrolled_message_count") != message_count:
    st.session_state["last_scrolled_message_count"] = message_count
    components.html("""
    <script>
    const doc = window.parent.document;
    const scroll = () => {
      const main = doc.querySelector('[data-testid="stMain"]');
      if (!main) return;
      const latest = main.querySelector('[data-testid="stChatMessage"]:last-of-type');
      if (latest) latest.scrollIntoView({block: 'end', behavior: 'smooth'});
      main.scrollTo({top: main.scrollHeight, behavior: 'smooth'});
    };
    setTimeout(scroll, 150);
    setTimeout(scroll, 600);
    </script>
    """, height=0)
if not st.session_state.messages:
    with st.container(horizontal_alignment="center", gap="medium"):
        st.markdown("# :yellow[◇ ◆ ◇]", width="content")
        st.markdown("## SELAMAT DATANG DI WAYANG AI", width="content")
        st.markdown("Halo! Aku Wayang AI, temanmu untuk mengenal Wayang Sasak. Mau tahu tentang cerita, sejarah, atau tokoh-tokohnya? Tanyakan saja, aku siap membantu.",
                    width="content")
        st.caption("Lampirkan gambar untuk mengenali tokoh, atau tulis pertanyaan Anda.")
else:
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            render_user_message(message, index)
        else:
            with st.chat_message("assistant"):
                with st.container(key=f"chat_assistant_bubble_{index}"):
                    render(message["kind"], message["data"], key=f"message_{index}")


def remember_response(kind, data):
    if not isinstance(data, dict):
        st.session_state.awaiting_more = False
        return
    if kind == "chat":
        st.session_state.conversation_context = data.get("conversation_context")
        for subject in data.get("subjects", []):
            if subject["id"] not in st.session_state.seen_ids:
                st.session_state.seen_ids.append(subject["id"])
    elif kind == "image":
        identity = data.get("prediksi") if data.get("status") == "accepted" else None
        st.session_state.conversation_context = ({"subject_ids": [identity["id"]], "evidence_ids": [], "intent": "profile"}
            if identity and isinstance(identity.get("id"), int) else None)
    character = data.get("character") if kind == "chat" else data.get("prediksi")
    if character and isinstance(character.get("id"), int):
        if character["id"] not in st.session_state.seen_ids:
            st.session_state.seen_ids.append(character["id"])
    st.session_state.awaiting_more = bool(data.get("follow_up")) or (
        kind == "image" and data.get("status") == "accepted")


def ask_question(question, message_id, character_id=None, explain_more=False):
    touch()
    st.session_state.messages.append({"role": "user", "kind": "text", "text": question,
                                      "message_id": message_id})
    with st.chat_message("user"):
        with st.container(key=f"chat_user_bubble_live_{message_id}"):
            st.text(question)
    with st.chat_message("assistant"):
        with st.container(key=f"chat_assistant_bubble_pending_{message_id}"):
            with st.spinner("Sedang memahami pertanyaan dan mencari informasi Wayang Sasak…"):
                try:
                    if explain_more:
                        kind, data = send_message(question, None, character_id=character_id,
                                                  explain_more=True, message_id=message_id)
                    elif character_id is not None:
                        kind, data = send_message(question, None, character_id=character_id,
                                                  message_id=message_id)
                    else:
                        options = {"message_id": message_id}
                        if st.session_state.conversation_context is not None:
                            options["conversation_context"] = st.session_state.conversation_context
                        if st.session_state.awaiting_more:
                            options.update(awaiting_more=True, seen_ids=st.session_state.seen_ids)
                        kind, data = send_message(question, None, **options)
                except BackendError as exc:
                    kind, data = "error", str(exc)
                except Exception:
                    kind, data = "error", "Permintaan gagal diproses. Silakan coba lagi."
    remember_response(kind, data)
    st.session_state.messages.append({"role": "assistant", "kind": kind, "data": data})


if st.session_state.awaiting_more:
    yes, no = st.columns(2)
    if yes.button("Ya, kenalkan tokoh lain", width="stretch", disabled=st.session_state.processing):
        if queue_action({"kind": "question", "question": "Ya"}):
            st.rerun()
    if no.button("Cukup dulu", width="stretch", disabled=st.session_state.processing):
        if queue_action({"kind": "question", "question": "Tidak dulu"}):
            st.rerun()

st.html("""
<style>
.stMainBlockContainer { padding-bottom: 15rem; }
.st-key-chat_composer {
    position: fixed; bottom: 1rem; z-index: 50;
    width: calc(100% - 4rem); max-width: 46rem;
    background: var(--background-color, #0e1117);
    border-radius: 1rem; box-shadow: 0 -8px 24px rgba(0,0,0,.12);
}
@media (min-width: 769px) {
    [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="true"])
    .st-key-chat_composer { width: calc(100% - 25rem); }
}
@media (max-width: 768px) {
    .st-key-chat_composer { left: 1rem; width: calc(100% - 2rem); bottom: .5rem; }
}
</style>
""")
with st.container(border=True, key="chat_composer"):
    attachment_col, message_col, send_col = st.columns([1.6, 6, 1.4], vertical_alignment="bottom")
    image_key = f"image_upload_{st.session_state.image_generation}"
    text_key = f"text_question_{st.session_state.text_generation}"
    current_image = st.session_state.get(image_key)
    with message_col:
        text = st.text_input(
            "Pesan", placeholder="Tulis pesan atau lampirkan satu gambar...",
            max_chars=1000, key=text_key, disabled=st.session_state.processing or current_image is not None,
            label_visibility="collapsed")
        if st.session_state.processing:
            st.caption("Sedang memproses pesan. Input dinonaktifkan sementara.")
    with attachment_col:
        with st.popover("Lampiran", icon=":material/attach_file:"):
            st.caption("Upload Foto")
            uploaded = st.file_uploader(
                "Unggah gambar tokoh Wayang Sasak", type=["jpg", "jpeg", "png", "webp"],
                key=image_key, disabled=st.session_state.processing or bool(text.strip()),
                help="Maksimal 10 MB. Kosongkan teks untuk memilih gambar.")
            st.divider()
            if st.button("Jelajahi Katalog", icon=":material/menu_book:",
                         key="browse_catalog", disabled=st.session_state.processing or bool(text.strip()) or current_image is not None,
                         width="stretch"):
                queue_action({"kind": "catalog"})
                st.rerun()
    with send_col:
        submitted = st.button("Kirim", type="primary", icon=":material/arrow_upward:",
                              disabled=st.session_state.processing)
    if uploaded is not None:
        st.caption("Gambar siap dikirim. Hapus lampiran untuk menulis teks.")
if st.session_state.processing and (action := st.session_state.pop("pending_action", None)):
    try:
        if action["kind"] == "question":
            ask_question(action["question"], action["message_id"],
                         action.get("character_id"), action.get("explain_more", False))
        elif action["kind"] == "image":
            touch()
            message_id = action["message_id"]
            user_message = {"role": "user", "kind": "image", "text": "",
                            "image_bytes": action["image_bytes"], "message_id": message_id}
            st.session_state.messages.append(user_message)
            render_user_message(user_message, f"live_{message_id}")
            with st.chat_message("assistant"):
                with st.container(key=f"chat_assistant_bubble_pending_{message_id}"):
                    with st.spinner("Sedang menganalisis gambar Wayang Sasak…"):
                        image = BytesIO(action["image_bytes"])
                        image.name = action["name"]
                        image.size = len(action["image_bytes"])
                        try:
                            kind, data = send_message("", image, message_id=message_id)
                        except BackendError as exc:
                            kind, data = "error", str(exc)
                        except Exception:
                            kind, data = "error", "Permintaan gagal diproses. Silakan coba lagi."
            remember_response(kind, data)
            st.session_state.messages.append({"role": "assistant", "kind": kind, "data": data})
        elif action["kind"] == "catalog":
            touch()
            st.session_state.pop("search_result", None)
            st.session_state.messages.append({"role": "user", "kind": "text", "text": "Jelajahi Katalog"})
            with st.chat_message("user"):
                with st.container(key="chat_user_bubble_catalog_request"):
                    st.text("Jelajahi Katalog")
            with st.chat_message("assistant"):
                with st.container(key="chat_assistant_bubble_catalog_loading"):
                    with st.spinner("Sedang memuat katalog tokoh…"):
                        try:
                            data = request_api("/search", payload={"query": "daftar tokoh", "top_k": 3})
                            kind = "search"
                        except BackendError as exc:
                            kind, data = "error", str(exc)
                        except Exception:
                            kind, data = "error", "Katalog gagal dimuat. Silakan coba lagi."
            st.session_state.messages.append({"role": "assistant", "kind": kind, "data": data})
            st.session_state.awaiting_more = False
            st.session_state.conversation_context = None
    except Exception:
        st.session_state.messages.append({"role": "assistant", "kind": "error",
            "data": "Permintaan gagal diproses. Silakan coba lagi."})
    finally:
        st.session_state.processing = False
    st.rerun()

if submitted and not st.session_state.processing:
    if text.strip() and uploaded is not None:
        st.warning("Pilih satu jenis pesan: teks atau gambar. Hapus salah satunya dahulu.")
    elif uploaded is not None:
        if not 0 < uploaded.size <= 10 * 1024 * 1024:
            st.warning("Gunakan gambar tidak kosong dengan ukuran maksimal 10 MB.")
        elif queue_action({"kind": "image", "image_bytes": uploaded.getvalue(), "name": uploaded.name},
                          clear_image=True):
            st.rerun()
    elif text.strip() and any(c.isalnum() for c in text):
        if queue_action({"kind": "question", "question": text.strip()}, clear_text=True):
            st.rerun()


# Bind once per rerun; native input blur syncs its value before clicking Send.
components.html("""
<script>
const doc = window.parent.document;
const parent = window.parent;
if (window.parent.wayangEnterHandler) {
  doc.removeEventListener('keydown', window.parent.wayangEnterHandler, true);
}
if (parent.wayangSubmitClickHandler) {
  doc.removeEventListener('click', parent.wayangSubmitClickHandler, true);
}
const resetAfterClear = () => {
  const input = doc.querySelector('.st-key-chat_composer input[type="text"]');
  if (input && !input.value.trim()) parent.wayangSubmitPending = false;
};
resetAfterClear();
parent.wayangSubmitClickHandler = (event) => {
  const button = event.target.closest('.st-key-chat_composer button');
  if (!button || !button.textContent.includes('Kirim')) return;
  if (parent.wayangSubmitPending && event.detail !== 0) {
    event.preventDefault();
    event.stopImmediatePropagation();
    return;
  }
  parent.wayangSubmitPending = true;
};
doc.addEventListener('click', parent.wayangSubmitClickHandler, true);
parent.wayangEnterHandler = (event) => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
  if (!event.target.closest('.st-key-chat_composer') || event.target.tagName !== 'INPUT') return;
  if (event.target.disabled || !event.target.value.trim()) return;
  if (parent.wayangSubmitPending) return;
  event.preventDefault();
  parent.wayangSubmitPending = true;
  const main = doc.querySelector('[data-testid="stMain"]');
  if (main) main.scrollTo({top: main.scrollHeight, behavior: 'smooth'});
  event.target.blur();
  setTimeout(() => {
    const button = [...doc.querySelectorAll('.st-key-chat_composer button')]
      .find(item => item.textContent.includes('Kirim'));
    if (button && !button.disabled) button.click();
    else parent.wayangSubmitPending = false;
  }, 150);
};
doc.addEventListener('keydown', parent.wayangEnterHandler, true);
</script>
""", height=0)
