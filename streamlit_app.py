"""Native Streamlit presentation. User content lives only in session memory."""
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
import re
import streamlit as st
import streamlit.components.v1 as components

from ui_client import BackendError, request_api, send_message, web_answer_markdown
from ui_media import character_image
from ui_history import sync_history, clear_history, touch

load_dotenv()
st.set_page_config(page_title="Wayang AI · Wayang Sasak", page_icon="◈",
                   layout="centered", initial_sidebar_state="collapsed")


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


def catalog(items, key):
    for index, item in enumerate(items):
        if st.button(item.get("nama", item.get("label", "Tokoh")),
                     key=f"{key}_choice_{index}", width="stretch"):
            st.session_state["pending_question"] = "Jelaskan " + item.get("nama", item.get("label", "Tokoh"))
            st.session_state["pending_character_id"] = item.get("id")
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
            if isinstance(identity.get("id"), int) and st.button("Jelaskan lebih lanjut", key=f"{key}_explain"):
                st.session_state["pending_question"] = "Jelaskan " + name
                st.session_state["pending_character_id"] = identity["id"]
                st.session_state["pending_explain_more"] = True
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
            st.text(answer)
        if data.get("web_sources"):
            st.caption("Sumber internet")
            for number, source in enumerate(data["web_sources"], start=1):
                st.link_button(f"[{number}] {source['title']}", source["url"])
        if data.get("mode") == "answer" and data.get("llm_status") in {
            "rate_limit", "unavailable", "timeout", "invalid_key", "model_unavailable", "internal", "invalid_response"
        }:
            st.caption("Jawaban profil menggunakan basis pengetahuan lokal.")
        if data.get("scope_notice"):
            st.text(data["scope_notice"])
        catalog(data.get("choices") or data.get("catalog") or [], key)
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
st.session_state.setdefault("seen_ids", [])
st.session_state.setdefault("awaiting_more", False)
st.session_state.setdefault("conversation_context", None)

with st.sidebar:
    st.subheader("Jelajahi Wayang Sasak")
    if st.button("Percakapan Baru", icon=":material/add_comment:", width="stretch"):
        clear_history()
        st.rerun()
    show_catalog = st.button("Lihat katalog", icon=":material/menu_book:", width="stretch")
    if show_catalog:
        touch()
        st.session_state.pop("search_result", None)
        st.session_state.messages.append({"role": "user", "kind": "text", "text": "Lihat katalog"})
        try:
            data = request_api("/search", payload={"query": "daftar tokoh", "top_k": 3})
            st.session_state.messages.append({"role": "assistant", "kind": "search", "data": data})
        except BackendError as exc:
            st.session_state.messages.append({"role": "assistant", "kind": "error", "data": str(exc)})
        st.session_state["awaiting_more"] = False
        st.session_state["conversation_context"] = None
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
message_count = len(st.session_state.messages)
if message_count and st.session_state.get("last_scrolled_message_count") != message_count:
    st.session_state["last_scrolled_message_count"] = message_count
    components.html("""
    <script>
    const doc = window.parent.document;
    const scroll = () => {
      const main = doc.querySelector('[data-testid="stMain"]');
      if (main) main.scrollTo({top: main.scrollHeight, behavior: 'smooth'});
    };
    setTimeout(scroll, 150);
    setTimeout(scroll, 600);
    </script>
    """, height=0)
if not st.session_state.messages:
    with st.container(horizontal_alignment="center", gap="medium"):
        st.markdown("# :yellow[◇ ◆ ◇]", width="content")
        st.markdown("## SELAMAT DATANG DI WAYANG AI", width="content")
        st.markdown("Kenali tokoh dan temukan cerita, ciri khas, serta filosofi Wayang Sasak.",
                    width="content")
        st.caption("Lampirkan gambar untuk mengenali tokoh, atau tulis pertanyaan Anda.")
else:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["text"])
                if message.get("image_bytes"):
                    try:
                        st.image(message["image_bytes"], width=420,
                                 caption="Gambar yang kamu kirim")
                    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                        st.warning("Pratinjau gambar tidak dapat ditampilkan. Gunakan gambar JPG, PNG, atau WebP yang valid.")
            else:
                render(message["kind"], message["data"], key=f"message_{index}")


def remember_response(kind, data):
    if not isinstance(data, dict):
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


def ask_question(question, character_id=None, explain_more=False):
    touch()
    st.session_state.messages.append({"role": "user", "kind": "text", "text": question})
    with st.spinner("Wayang AI sedang menjawab…"):
        try:
            if explain_more:
                kind, data = send_message(question, None, character_id=character_id, explain_more=True)
            elif character_id is not None:
                kind, data = send_message(question, None, character_id=character_id)
            else:
                options = {}
                if st.session_state.conversation_context is not None:
                    options["conversation_context"] = st.session_state.conversation_context
                if st.session_state.awaiting_more:
                    options.update(awaiting_more=True, seen_ids=st.session_state.seen_ids)
                kind, data = send_message(question, None, **options)
        except BackendError as exc:
            kind, data = "error", str(exc)
    remember_response(kind, data)
    st.session_state.messages.append({"role": "assistant", "kind": kind, "data": data})


if st.session_state.awaiting_more:
    yes, no = st.columns(2)
    if yes.button("Ya, kenalkan tokoh lain", width="stretch"):
        st.session_state["pending_question"] = "Ya"
    if no.button("Cukup dulu", width="stretch"):
        st.session_state["pending_question"] = "Tidak dulu"

if "pending_question" in st.session_state:
    ask_question(st.session_state.pop("pending_question"), st.session_state.pop("pending_character_id", None),
                 st.session_state.pop("pending_explain_more", False))
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
            max_chars=1000, key=text_key, disabled=current_image is not None,
            label_visibility="collapsed")
    with attachment_col:
        with st.popover("Lampiran", icon=":material/attach_file:"):
            uploaded = st.file_uploader(
                "Unggah gambar tokoh Wayang Sasak", type=["jpg", "jpeg", "png", "webp"],
                key=image_key, disabled=bool(text.strip()),
                help="Maksimal 10 MB. Kosongkan teks untuk memilih gambar.")
    with send_col:
        submitted = st.button("Kirim", type="primary", icon=":material/arrow_upward:")
    if uploaded is not None:
        st.caption("Gambar siap dikirim. Hapus lampiran untuk menulis teks.")
    if text.strip() or uploaded is not None:
        if st.button("Hapus input", icon=":material/close:"):
            st.session_state.image_generation += 1
            st.session_state.text_generation += 1
            st.rerun()

if submitted and text.strip() and uploaded is not None:
    st.warning("Pilih satu jenis pesan: teks atau gambar. Hapus salah satunya dahulu.")
elif submitted and uploaded is not None:
    if not 0 < uploaded.size <= 10 * 1024 * 1024:
        st.warning("Gunakan gambar tidak kosong dengan ukuran maksimal 10 MB.")
    else:
        touch()
        st.session_state.messages.append({"role": "user", "kind": "image", "text": "",
                                          "image_bytes": uploaded.getvalue()})
        with st.spinner("Wayang AI sedang mengenali tokoh…"):
            try:
                kind, data = send_message("", uploaded)
            except BackendError as exc:
                kind, data = "error", str(exc)
        remember_response(kind, data)
        st.session_state.messages.append({"role": "assistant", "kind": kind, "data": data})
        st.session_state.image_generation += 1
        st.session_state.text_generation += 1
        st.rerun()

elif submitted:
    if not any(c.isalnum() for c in text):
        st.warning("Tulis pesan atau lampirkan gambar terlebih dahulu.")
    else:
        ask_question(text.strip())
        st.session_state.image_generation += 1
        st.session_state.text_generation += 1
        st.rerun()


# Bind once per rerun; native input blur syncs its value before clicking Send.
components.html("""
<script>
const doc = window.parent.document;
if (window.parent.wayangEnterHandler) {
  doc.removeEventListener('keydown', window.parent.wayangEnterHandler, true);
}
window.parent.wayangEnterHandler = (event) => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
  if (!event.target.closest('.st-key-chat_composer') || event.target.tagName !== 'INPUT') return;
  if (event.target.disabled || !event.target.value.trim()) return;
  event.preventDefault();
  const main = doc.querySelector('[data-testid="stMain"]');
  if (main) main.scrollTo({top: main.scrollHeight, behavior: 'smooth'});
  event.target.blur();
  setTimeout(() => {
    const button = [...doc.querySelectorAll('.st-key-chat_composer button')]
      .find(item => item.textContent.includes('Kirim'));
    if (button && !button.disabled) button.click();
  }, 150);
};
doc.addEventListener('keydown', window.parent.wayangEnterHandler, true);
</script>
""", height=0)
