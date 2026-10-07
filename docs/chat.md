# Tahap 3: chat dengan evidence tervalidasi

**Status: perbaikan kompatibilitas berdasarkan validasi Colab pengguna.**
Pengguna membuktikan akses Gemini 3.6 Flash dan structured output dengan
`response_json_schema`. Pengguna kini melaporkan smoke Gemini asli **5/5 lulus**,
termasuk filosofi Gunungan dengan `llm_used=true` dan `llm_status=used`.
Baseline aktif 48 chunk mencakup `gunungan__filosofi`; pembaruan pengetahuan
dan artefak Drive disengaja oleh proyek. Lihat [provenance dan baseline](retrieval.md#baseline-artefak-aktif--pembaruan-gunungan-25-september-2026).
Test lokal memakai mock tanpa membaca key atau memanggil API asli.

`POST /chat` menerima `{"message":"Jelaskan Jayangrana"}`. Message 1–1000
karakter dan harus mengandung huruf/angka. Konteks lanjutan menerima `awaiting_more`
dan `seen_ids` (maksimal 20 ID resmi); riwayat percakapan bebas tidak dikirim.
Field lain ditolak agar kontrak jelas. Service menggunakan instance retrieval yang sama
dengan `/search`.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Coba berikan saya contoh tokoh Wayang Saksak?"}'
```

Response berisi mode, answer, sources, suggestions, llm_used, llm_status, dan
catalog (20 ID/nama JSON) untuk mode catalog/fallback_catalog. Source answer
berisi document `wayang_db.json`, id, label, dan evidence_ids yang benar-benar
dirender. Mode katalog memiliki source document katalog, tanpa source palsu.

## Profil tokoh lokal: grounding yang dapat diperiksa

Gemini memilih dan mengurutkan ID evidence, **bukan menulis fakta bebas**.
Respons terstruktur divalidasi dengan Pydantic: field ekstra, ID tak dikenal,
ID duplikat, dan output malformed ditolak. Backend merender nama asli dan
kutipan JSON yang dipilih. Ini sengaja merupakan jawaban ekstraktif: susunan
kalimat lebih terbatas daripada parafrase generatif, tetapi setiap pernyataan
berasal dari evidence yang dapat dicocokkan programatik. System instruction
saja tidak dijadikan jaminan kebenaran fakta.

Context berisi maksimal 6000 karakter JSON evidence dari 3 hasil retrieval,
termasuk ID/label/nama dan kutipan publik. Tidak ada embedding, daftar gambar,
path internal, atau profil training dikirim. Query dikirim sebagai data di
pesan pengguna; aturan tetap berada di system instruction. Pada jalur profil tokoh ini, tidak ada tools,
web search, atau pengetahuan luar dipakai untuk jawaban.

Mode catalog/fallback_catalog tidak memanggil Gemini. Nama di teks dan field
terstruktur semuanya dirender dari katalog JSON, sehingga model tidak bisa
menambah, mengganti, atau mengurangi nama. Instruksi jelas untuk mengarang tokoh
dirutekan deterministik ke katalog. Untuk query lain, pemilihan evidence LLM
tetap dibatasi pada ID yang diberikan. Bila evidence_ids kosong, sistem
menyatakan informasi belum tersedia dan menawarkan katalog.

Jaminan ini membatasi **asal fakta**, bukan membuktikan semua query relevan
terjawab: ranking, kelengkapan JSON, atau evidence yang dipilih masih perlu
evaluasi dengan data asli. Timeout/error LLM menggunakan kutipan terbaik dari
retrieval sebagai fallback dan menandai llm_used=false.

## SDK, model, dan konfigurasi

SDK resmi: `google-genai>=1.30,<2`. Default `GEMINI_MODEL=gemini-3.6-flash`.
Operator tetap dapat override melalui `GEMINI_MODEL`; adapter memakai Settings.
Tidak ada penggantian model otomatis atau request berbayar saat startup.
Schema dikirim melalui `response_json_schema=EvidenceSelection.model_json_schema()`
dengan `additionalProperties=false`; validasi Pydantic lokal dan allowlist ID
wajib tetap berlaku. HTTP 400 umum menjadi `invalid_request`, HTTP 404 menjadi
`model_unavailable`, dan reason `API_KEY_INVALID` tetap `invalid_key`.

Thinking memakai default provider tanpa parameter khusus model. Batas output
1024 token dipertahankan berdasarkan keberhasilan eksperimen Colab pengguna;
output terpotong tetap ditolak dan memakai fallback aman. Lihat
[dokumentasi thinking resmi](https://ai.google.dev/gemini-api/docs/generate-content/thinking).

- `GEMINI_API_KEY`: hanya environment/Colab Secrets, tidak dicetak/disimpan repo.
- `GEMINI_MODEL`: identifier model yang tersedia untuk akun.
- `GEMINI_TIMEOUT_SECONDS=15`: deadline total pemanggilan LLM termasuk retry,
  rentang >0 sampai 60 detik.
- `GEMINI_MAX_ATTEMPTS=2`: maksimum total attempt, rentang 1–3. Backoff 0.5 lalu
  1 detik untuk error sementara 500/502/503/504 atau connection error. Timeout,
  429/quota, key invalid, dan input/model invalid tidak di-retry. Retry bawaan
  SDK dinonaktifkan agar jumlah attempt tidak berlipat.

Key kosong tetap mengaktifkan chat deterministik. `/health` menambahkan
gemini_configured dan gemini_client_ready; flag itu hanya validasi konfigurasi
lokal, bukan bukti key/quota/model berhasil digunakan. Client dibuat satu kali
di lifespan dan ditutup pada shutdown.

422 untuk input invalid, 503 untuk retrieval unavailable (katalog tetap dapat
berjalan dari JSON), 500 untuk internal error. Kegagalan Gemini menghasilkan
HTTP 200 dengan fallback berguna; periksa llm_status: used, not_needed,
not_configured, timeout, rate_limit, invalid_key, invalid_request, model_unavailable, unavailable,
invalid_response, insufficient_evidence, atau internal. Log hanya durasi dan
kategori aman. Raw exception, prompt, context, dan key tidak dikirim kembali.

## Validasi Colab

Pasang requirements-dev, mount Drive, pastikan smoke search lulus dengan
threshold/normalisasi embedding yang sesuai. Simpan key melalui Colab Secrets
dengan nama GEMINI_API_KEY dan izinkan akses notebook, atau environment.
Skrip memuat Secrets hanya bila environment belum tersedia.

```bash
WAYANG_ENABLE_CHAT=true GEMINI_MODEL=gemini-3.6-flash python scripts/colab_chat_smoke_test.py
```

Skrip ini opt-in dan dapat memakai quota/biaya API. Test default pytest tidak
menggunakan key/API sungguhan. Smoke hanya mencetak mode/status/jumlah source,
tidak menulis response atau key ke file. Kelulusan mengharuskan kedua kasus in-scope
memakai Gemini sungguhan dengan ID dari context yang benar-benar dikirim; fallback saja bukan kelulusan
integrasi. JSON dan Chroma sumber di-hash sebelum/sesudah. Model klasifikasi
hanya dibaca oleh loader Tahap 1.

Rate limit di gateway, authentication, dan evaluasi beban belum merupakan
deployment produksi. Batas message/context/timeout sudah diterapkan; sebelum
membuka layanan publik, tambahkan pembatasan request/quota sesuai pengguna.

Sumber resmi: [SDK yang didukung](https://ai.google.dev/gemini-api/docs/libraries),
[Gemini 3.6 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash),
[Google GenAI SDK Python](https://googleapis.github.io/python-genai/).


## Pertanyaan budaya umum dan sumber internet

Sejarah, asal-usul, jalan cerita, lakon, dan jumlah tokoh diarahkan ke jalur
`web_answer`. Gemini memakai Google Search dan backend memeriksa metadata sumber
serta dukungan kutipan. UI menampilkan nomor referensi, tautan sumber, dan widget
Search Suggestions dari provider. Instruksi di luar domain tetap disaring sebelum
panggilan web. Jawaban web tidak menambahkan tokoh ke katalog atau mengubah JSON.

Saat kuota Google Search atau layanannya bermasalah, aplikasi mencoba membaca
sumber budaya publik dari daftar tetap di `app/web_sources.py`; Gemini merangkum
teks yang berhasil diambil dengan ID sumber tervalidasi. Ini pembacaan sumber web
terpilih, bukan mesin pencarian umum. Keterkaitan ID dan URL dapat diperiksa,
sedangkan kesesuaian setiap parafrase tetap memiliki keterbatasan model generatif.
Jika sumber tidak mendukung angka total atau cerita tertentu, jawaban menyatakan
keterbatasannya. Batas 20 tokoh adalah cakupan profil/pengenalan aplikasi, bukan
jumlah seluruh tokoh tradisional. Pertanyaan jumlah di aplikasi dijawab langsung
berdasarkan katalog tanpa panggilan Gemini.

`web_sources` membawa judul dan URL HTTPS; `sources` tetap dikhususkan untuk JSON
lokal. HTML hanya berasal dari metadata `search_entry_point` Google dan dirender
melalui sanitasi `st.html`, bukan HTML jawaban model. Kuota/error internet tidak
ditampilkan sebagai jawaban terverifikasi; `llm_status` menyatakan kegagalannya.
Circuit breaker pencarian dipisahkan dari chat biasa. Timeout web total 60 detik;
`GEMINI_SEARCH_TIMEOUT_SECONDS` mengatur timeout model di jalur tersebut.
