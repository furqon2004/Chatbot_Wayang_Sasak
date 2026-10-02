"""Native Streamlit presentation. User content lives only in session memory."""
from dotenv import load_dotenv
import streamlit as st

from ui_client import BackendError, request_api, send_message

load_dotenv()
st.set_page_config(page_title="Wayang AI · Wayang Sasak", page_icon="◈",
                   layout="centered", initial_sidebar_state="collapsed")


def profile(value):
    for key, title in (("deskripsi", "Profil"), ("deskripsi_lengkap", "Tentang tokoh"),
                       ("filosofi", "Filosofi"), ("ciri_ciri", "Ciri-ciri")):
        if value.get(key):
            st.caption(title)
            st.text(value[key])
    if value.get("kata_kunci"):
        st.text(" · ".join(value["kata_kunci"]))


def catalog(items):
    for item in items:
        with st.container(border=True):
            st.text(item.get("nama", item.get("label", "Tokoh")))
            if item.get("profil"):
                profile(item["profil"])


def render(kind, data):
    if kind == "error":
        st.warning(data, icon=":material/info:")
    elif kind == "image":
        identity = data.get("prediksi")
        if data.get("status") == "accepted" and isinstance(identity, dict):
            st.success("Tokoh berhasil dikenali", icon=":material/check_circle:")
            st.text(identity.get("nama", identity.get("label", "Tokoh")))
            left, right = st.columns(2)
            left.metric("Confidence", f"{data.get('confidence', 0):.1%}")
            right.metric("Margin", f"{data.get('margin', 0):.1%}")
            profile(data.get("profil") or {})
        else:
            st.warning("Identitas tokoh belum dapat dipastikan", icon=":material/info:")
            st.text(data.get("message") or "Coba gambar lebih terang, tajam, dan menampilkan satu tokoh secara utuh.")
            st.caption("Kandidat berikut bukan identifikasi final.")
        st.markdown("**Top-3 kandidat**")
        for candidate in data.get("top_3", [])[:3]:
            st.text(f"{candidate.get('nama', candidate.get('label', 'Tokoh'))} · {candidate.get('probability', 0):.1%}")
    elif kind == "chat":
        st.text(data.get("answer", "Belum ada jawaban."))
        if data.get("scope_notice"):
            st.text(data["scope_notice"])
        catalog(data.get("catalog") or [])
        if data.get("sources"):
            with st.expander("Sumber jawaban"):
                for source in data["sources"]:
                    st.text(f"{source.get('document', 'Basis pengetahuan')} · {source.get('label') or 'Katalog Wayang Sasak'}")
    elif kind == "search":
        if data.get("message"):
            st.text(data["message"])
        catalog(data.get("results") or data.get("catalog") or [])
    if isinstance(data, dict):
        for suggestion in data.get("suggestions", []):
            st.text(suggestion)


st.session_state.setdefault("messages", [])
st.session_state.setdefault("composer_generation", 0)

with st.sidebar:
    st.subheader("Jelajahi Wayang Sasak")
    if st.button("Percakapan Baru", icon=":material/add_comment:", width="stretch"):
        st.session_state.clear()
        st.rerun()
    with st.form("search"):
        query = st.text_input("Cari tokoh atau ciri-ciri", max_chars=1000,
                              placeholder="Contoh: Jayengrana")
        search = st.form_submit_button("Cari", width="stretch")
    show_catalog = st.button("Lihat katalog", icon=":material/menu_book:", width="stretch")
    if search or show_catalog:
        if search and not any(c.isalnum() for c in query):
            st.warning("Masukkan nama atau ciri-ciri tokoh.")
        else:
            with st.spinner("Mencari tokoh…"):
                try:
                    st.session_state["search_result"] = request_api("/search", payload={
                        "query": "daftar semua tokoh" if show_catalog else query, "top_k": 3})
                except BackendError as exc:
                    st.session_state.pop("search_result", None)
                    st.warning(str(exc))
    if "search_result" in st.session_state:
        render("search", st.session_state["search_result"])
    st.caption("Riwayat hanya tersedia selama sesi ini.")


@st.fragment(run_every="30s")
def header():
    with st.container(border=True, horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center"):
        st.markdown("### :yellow[◈] WAYANG AI", width="content")
        try:
            request_api("/health")
            st.badge("ONLINE", color="green", icon=":material/check_circle:")
        except BackendError:
            st.badge("OFFLINE", color="gray", icon=":material/cloud_off:")


header()
if not st.session_state.messages:
    with st.container(horizontal_alignment="center", gap="medium"):
        st.markdown("# :yellow[◇ ◆ ◇]", width="content")
        st.markdown("## SELAMAT DATANG DI WAYANG AI", width="content")
        st.markdown("Kenali tokoh dan temukan cerita, ciri khas, serta filosofi Wayang Sasak.",
                    width="content")
        st.caption("Lampirkan gambar untuk mengenali tokoh, atau tulis pertanyaan Anda.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["text"])
                if message.get("has_image"):
                    st.caption("Gambar dilampirkan untuk pengenalan tokoh.")
            else:
                render(message["kind"], message["data"])

with st.form(f"composer_{st.session_state.composer_generation}"):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        with st.popover("Lampiran", icon=":material/attach_file:"):
            uploaded = st.file_uploader("Lampirkan gambar tokoh", type=["jpg", "jpeg", "png", "webp"],
                                        help="Maksimal 10 MB. Satu gambar per pesan.")
        text = st.text_input("Pertanyaan tentang Wayang Sasak", placeholder="Tanyakan tentang Wayang Sasak…",
                             max_chars=1000)
        submitted = st.form_submit_button("KIRIM", type="primary")
    st.caption("Jika gambar dilampirkan, pesan ini digunakan untuk mengenali tokoh. Pertanyaan lanjutan dapat dikirim tanpa gambar.")

if submitted:
    if uploaded is None and not any(c.isalnum() for c in text):
        st.warning("Tulis pertanyaan atau lampirkan gambar terlebih dahulu.")
    else:
        st.session_state.messages.append({"role": "user", "text": text.strip(), "has_image": uploaded is not None})
        with st.spinner("Wayang AI sedang memproses…"):
            try:
                kind, data = send_message(text, uploaded)
            except BackendError as exc:
                kind, data = "error", str(exc)
        st.session_state.messages.append({"role": "assistant", "kind": kind, "data": data})
        st.session_state.composer_generation += 1
        st.rerun()
