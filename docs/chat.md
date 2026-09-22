# Tahap 3: chat dengan evidence tervalidasi

**Status: PERSIAPAN — DITUNDA MENUNGGU API KEY atas permintaan pengguna.**
Kode di bawah sudah tersedia dan diuji dengan mock sebelum permintaan penundaan.
Belum ada request Gemini asli; Tahap 3 belum dinyatakan selesai. Biarkan key
kosong untuk memakai fallback. Perintah smoke Colab di bawah disiapkan untuk
dijalankan setelah pengguna siap mengaktifkan Gemini.

`POST /chat` menerima `{"message":"Jelaskan Jayangrana"}`. Message 1–1000
karakter dan harus mengandung huruf/angka. History belum didukung; field ekstra
ditolak agar kontrak jelas. Service menggunakan instance retrieval yang sama
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

## Grounding yang dapat diperiksa

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
pesan pengguna; aturan tetap berada di system instruction. Tidak ada tools,
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

SDK resmi: `google-genai>=1.30,<2`. Default `GEMINI_MODEL=gemini-2.5-flash`
diverifikasi terhadap halaman resmi pada 22 September 2026. Google masih
menyediakan identifier tersebut, tetapi akses keluarga 2.5 dibatasi ke pengguna
lama. Model bisa diubah dengan environment sesuai akses akun; backend tidak
mengganti model otomatis dan startup tidak melakukan request berbayar.

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
not_configured, timeout, rate_limit, invalid_key, model_unavailable, unavailable,
invalid_response, insufficient_evidence, atau internal. Log hanya durasi dan
kategori aman. Raw exception, prompt, context, dan key tidak dikirim kembali.

## Validasi Colab

Pasang requirements-dev, mount Drive, pastikan smoke search lulus dengan
threshold/normalisasi embedding yang sesuai. Simpan key melalui Colab Secrets
dengan nama GEMINI_API_KEY dan izinkan akses notebook, atau environment.
Skrip memuat Secrets hanya bila environment belum tersedia.

```bash
python scripts/colab_chat_smoke_test.py
```

Skrip ini opt-in dan dapat memakai quota/biaya API. Test default pytest tidak
menggunakan key/API sungguhan. Smoke hanya mencetak mode/status/jumlah source,
tidak menulis response atau key ke file. Kelulusan mengharuskan setidaknya satu
request Gemini sungguhan memakai evidence valid; fallback saja bukan kelulusan
integrasi. JSON dan Chroma sumber di-hash sebelum/sesudah. Model klasifikasi
hanya dibaca oleh loader Tahap 1.

Rate limit di gateway, authentication, dan evaluasi beban belum merupakan
deployment produksi. Batas message/context/timeout sudah diterapkan; sebelum
membuka layanan publik, tambahkan pembatasan request/quota sesuai pengguna.

Sumber resmi: [SDK yang didukung](https://ai.google.dev/gemini-api/docs/libraries),
[Gemini 2.5 Flash dan akses](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash),
[Google GenAI SDK Python](https://googleapis.github.io/python-genai/).
