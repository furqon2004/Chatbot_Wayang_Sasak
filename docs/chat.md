# Chat dan pencarian OpenAI

Backend memakai OpenAI untuk narasi lokal dan pencarian internet, dengan satu
`LLM_API_KEY`. Default `LLM_PROVIDER=openai` dan `LLM_MODEL=gpt-4.1-mini`.
Tidak ada dependensi atau adapter Gemini aktif. Konfigurasi lama `GEMINI_*`
tidak digunakan. Pengaturan dibaca proses backend; restart setelah pembaruan kode.

`POST /chat` memeriksa domain Wayang Sasak sebelum retrieval/provider.
Katalog dan discovery peran berjalan lokal. Konteks lanjutan divalidasi backend;
konteks lama tidak mengizinkan topik baru di luar domain.

Narasi lokal menggunakan `local-narrative-v2`, evidence JSON/retrieval asli,
structured output dan sitasi yang dirender backend. Lihat
[narasi interaktif dan batas faktual](interactive-narrative.md).
ID valid dan guard heuristik bukan bukti kesetiaan faktual.

Pencarian memakai SDK OpenAI Responses dengan tool `web_search` wajib.
Backend mensyaratkan hasil pencarian selesai dan anotasi `url_citation` yang valid;
nomor sitasi dipetakan ke sumber HTTPS dan ditampilkan sebagai tautan di UI.
Jawaban tanpa sumber ditolak. Jika layanan atau evidence tidak tersedia,
aplikasi memberi fallback jujur dan katalog lokal. Tidak ada jawaban pengganti
dari ingatan model tanpa evidence. Jumlah 20 hanya cakupan katalog aplikasi.

Konfigurasi aktif (lihat `.env.example`, jangan bagikan key):

- `WAYANG_ENABLE_CHAT=true` mengaktifkan chat.
- `LLM_API_KEY` dipakai kedua jalur; tidak membutuhkan key pencarian terpisah.
- `LLM_TIMEOUT_SECONDS=45`, `LLM_MAX_ATTEMPTS=2` untuk narasi.
- `WEB_TIMEOUT_SECONDS=45`, `WEB_MAX_ATTEMPTS=2` untuk pencarian.
- SDK `max_retries=0`; retry aplikasi dibatasi 1?2 attempt.
- Cache lokal TTL 900 detik/32 entri; kegagalan tidak disimpan.

Smoke browser lokal opsional, dijalankan pengguna dari PowerShell:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\local_browser_smoke.py --enable-search
```

Skrip ini memakai koneksi langsung khusus smoke, TLS aktif, retry mati,
dan maksimal tiga panggilan SDK gabungan narasi/pencarian. Ikuti petunjuk
terminal untuk UI. Tanpa flag tersebut pencarian tidak diaktifkan dalam smoke.
Pengujian otomatis memakai mock dan tidak membuktikan akses/kuota akun nyata.

Untuk membuka UI yang tersambung ke backend smoke tersebut, jalankan terminal
PowerShell kedua dari direktori proyek (environment ini hanya berlaku pada proses):

```powershell
$env:WAYANG_API_URL="http://127.0.0.1:8011"
.\.venv\Scripts\python.exe -B -m streamlit run .\streamlit_app.py --server.port 8511
```

Buka `http://localhost:8511`, mulai percakapan baru, lalu tanyakan
“Apa itu Wayang Sasak?”. Bila evidence lokal kurang, backend mencoba web search.
Periksa tautan sitasi pada jawaban. “Siapa tokoh utamanya?” dapat dilanjutkan
dari konteks domain tersebut; pertanyaan pemrograman tetap ditolak lokal.
Hentikan pemeriksaan bila provider gagal. Batas tiga panggilan SDK dihitung
bersama untuk narasi dan web; memuat ulang UI tidak mereset batas backend.

Referensi kontrak: [OpenAI web search](https://developers.openai.com/api/docs/guides/tools-web-search).


## Pemakaian lokal biasa (bukan smoke)

Jangan memakai `local_browser_smoke.py` sebagai server percakapan sehari-hari:
skrip itu sengaja berhenti setelah kegagalan pertama atau tiga panggilan SDK total.
Kegagalan narasi pada smoke juga menghentikan pencarian web berikutnya.

Hentikan backend smoke dengan Ctrl+C, kemudian jalankan:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\local_chat.py --direct
```

Launcher ini memakai port 8011 yang sama, mengaktifkan narasi dan web, serta
membaca konfigurasi lokal hanya saat pengguna menjalankannya. `--direct` memilih
HTTPX trust_env=False untuk proses tersebut; TLS tetap aktif. Tanpa flag itu,
HTTPX mengikuti environment. Tidak mengubah `.env`, proxy sistem, atau default aplikasi.
Jangan gunakan koneksi langsung bila bertentangan dengan kebijakan jaringan.

Tidak ada batas tiga panggilan per sesi pada launcher aplikasi ini; permintaan
dapat memakai kuota API. SDK tanpa retry, satu attempt aplikasi per operasi.
Cache, pembatasan kapasitas dan circuit breaker tetap berlaku. Kegagalan narasi
tidak menghentikan web, dan kegagalan web tidak menghentikan narasi.
Pengujian mock mencakup kedua arah; ini bukan bukti konektivitas akun nyata.
