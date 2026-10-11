> Catatan migrasi OpenAI: uraian provider/konfigurasi lama di bawah merupakan
> catatan historis. Implementasi aktif memakai OpenAI untuk narasi dan web;
> lihat [konfigurasi chat](chat.md). Hasil audit lama bukan validasi migrasi terbaru.

# Catatan validasi milestone

**Pembaruan 25 September 2026:** pengguna melaporkan smoke retrieval asli
lulus pada baseline **48 chunk** dan smoke Gemini asli **5/5 lulus**, termasuk
filosofi Gunungan (`llm_used=true`, `llm_status=used`). JSON/Drive diperbarui
secara sengaja berdasarkan data/narasumber proyek; chunk baru
`gunungan__filosofi`. Lihat [baseline dan provenance](retrieval.md).
Validasi lokal pembaruan baseline: **359 test lulus**, 3 warning baseline;
compileall, pip check, secret scan (0 temuan), dan git diff --check lulus.
Tes mencakup batas 47/48/49 untuk strict dan production, metric cosine,
serta evidence filosofi Gunungan yang harus berasal dari context.

Bagian A?C di bawah adalah riwayat tahap sebelumnya; status menunggu di sana
tidak menggantikan laporan Colab terbaru. Tidak ada eksekusi API asli lokal.

Baseline Git: `main`, `d23d70b`. `todolist.md` sudah berubah saat pekerjaan
dimulai dan dipertahankan sebagai perubahan pengguna, tidak dimasukkan ke commit.
Baseline yang benar-benar dijalankan lokal: **23 passed**, 3 warning.
Riwayat validasi artefak asli pada todolist diterima sebagai informasi pengguna;
angka hasil tersebut tidak diklaim sebagai hasil eksekusi ulang sesi ini.

## A — Audit regresi Tahap 1

Status: **IMPLEMENTASI LOKAL SELESAI — VALIDASI COLAB MENUNGGU**.

- Mempertahankan pipeline inferensi, temperature divide, ranking sebelum
  pembulatan, fallback CPU, dan load sekali saat lifespan.
- Serializer bersama `app/public.py` menggunakan allowlist profil. Respons
  `/predict-image` sengaja tidak lagi mengirim field training/internal.
- Parser `app/resources.py` menerima nested mapping legacy berdasarkan isi.
  Bentuk JSON asli perlu diperiksa lagi di Colab; tidak ada artefak lokal.
- File lain: `app/inference.py`, `tests/test_public.py`, `README.md`.
- Unit test lokal: **33 passed**, 3 warning (dua deprecation dependency,
  satu warning Pillow yang sengaja dipicu test). Compileall dan pip check lulus.
- Smoke model asli: **BELUM DIJALANKAN**. Jalankan skrip Colab yang sudah ada
  dengan gambar acuan yang sama dan `--reference-json`, toleransi `1e-5`.
  Parser resource berubah; validasi ini tetap diperlukan.
- Contoh hasil mock yang diuji: prediksi fixture ID 19 (`gunungan`), profil
  tanpa `embedding`; angka confidence mock bukan hasil model asli.
- Model, dataset, JSON asli, dan ChromaDB tidak disentuh.

## B — Hybrid retrieval tanpa Gemini

Status: **IMPLEMENTASI LOKAL SELESAI — VALIDASI COLAB MENUNGGU**.

- Menambahkan `app/catalog.py`, `query_normalizer.py`, `lexical.py`,
  `semantic.py`, `retrieval.py`, `retrieval_resources.py`, `schemas.py`,
  `search_routes.py`, dan `evaluation.py`. Lifespan/config dan dependency
  diperluas; pipeline numerik klasifikasi tetap sama.
- `/search` menyediakan answer/catalog/fallback_catalog. `/health` menambah
  retrieval_loaded, collection_name, document_count. Katalog tetap tersedia
  dari JSON ketika semantic belum siap; query biasa menghasilkan 503.
- Chroma hanya dibuka dari snapshot sementara. Metadata/metric diverifikasi;
  chunk digroup per tokoh, bobot semantic/lexical tetap 55/45.
- `evaluation/search_cases.json`, fixture sintetis, dan `scripts/evaluate_search.py`
  menghasilkan threshold **0.457** dari gap **0.220–0.694**. **14/14** kasus
  evaluasi mock lulus. Threshold ini belum terkalibrasi dengan embedding asli.
- Unit test seluruh milestone A+B: **81 passed**, 3 warning yang sama seperti
  baseline. Compileall, pip check, dan impor dependency retrieval lulus.
  Environment lokal: Python 3.10, ChromaDB 1.5.9, SentenceTransformers 3.4.1,
  RapidFuzz 3.14.5. Versi Colab belum diuji dalam sesi ini.
- Test baru: catalog, config, query_normalizer, retrieval, retrieval_resources,
  search_api; test API lama hanya diperbarui untuk tambahan field health dan
  isolasi path mock. Seluruh pemeriksaan numerik Tahap 1 dipertahankan.
- Contoh **MOCK**, hasil eksekusi fixture:
  [katalog search 20 tokoh](examples/search-catalog-mock.json).
- Smoke Chroma asli: **BELUM DIJALANKAN**. Disiapkan
  `scripts/colab_search_smoke_test.py`, termasuk kalibrasi opsional dan hash
  JSON/Chroma sumber sebelum/sesudah. Format metadata, metric, normalisasi
  embedding, dan kompatibilitas snapshot perlu dipastikan di Colab.
- Dokumentasi: `docs/retrieval.md`, README, `.env.example`.
- Model, dataset, wayang_db.json asli, dan ChromaDB sumber tidak diubah.
  Sesuai instruksi pengguna, lanjut C setelah gerbang lokal B lulus; klaim
  kelulusan artefak asli tetap menunggu smoke Colab.

## C — Persiapan chat; aktivasi Gemini ditunda

Status: **PERSIAPAN TAHAP 3 — MENUNGGU API KEY / PERMINTAAN LANJUTAN PENGGUNA**.
Tidak dinyatakan sebagai milestone selesai. Saat pengguna meminta penundaan,
kode integrasi dan test mock sudah dibuat; pekerjaan itu disimpan tanpa
mengaktifkan atau memvalidasi API Gemini asli.

- Modul yang disiapkan: `app/chat.py`, `chat_routes.py`, `chat_schemas.py`,
  `gemini.py`; config/lifespan/environment/dependency diperluas. SDK lokal
  google-genai **1.75.0**, default model saat ini gemini-3.6-flash setelah perbaikan kompatibilitas
  berdasarkan bukti Colab pengguna; smoke aplikasi penuh perlu dijalankan ulang.
- `/chat` menggunakan retrieval yang sama. Katalog dirender langsung dari
  JSON. Pada jalur LLM, hanya pemilihan ID evidence yang diterima; fakta/nama
  output dirender dari kutipan asli. Tanpa key, fallback tetap bekerja.
- `/health` menambah gemini_configured dan gemini_client_ready tanpa key/prompt.
- Test tambahan: chat API, adapter SDK, sumber smoke, konfigurasi, timeout,
  retry terbatas, quota/key/model error, malformed, prompt injection, sources,
  dan regresi bahwa endpoint lain tidak memanggil Gemini.
- Test akhir lokal: **133 passed**, 3 warning baseline. Compileall dan pip
  check lulus. Transport HTTP mock memeriksa SDK resmi tanpa jaringan berbayar.
- Evaluation fixture diperluas untuk query chat wajib; threshold tidak berubah.
- Contoh **MOCK**, dieksekusi tanpa key:
  [chat katalog](examples/chat-catalog-mock.json),
  [chat jawaban fallback](examples/chat-answer-mock.json).
- Smoke Gemini asli: **BELUM DIJALANKAN / DITUNDA**. Skrip
  `scripts/colab_chat_smoke_test.py` sudah disiapkan untuk environment/Colab
  Secrets; tidak menyimpan key atau respons sensitif ke file.
- Dokumentasi: `docs/chat.md`, README, `.env.example`. `.env` lokal tetap
  diabaikan Git dan tidak ada key asli yang ditulis ke repository.
- Sisa pekerjaan setelah pengguna siap: konfirmasi akses model pada akun,
  smoke search/kalibrasi dengan Chroma asli, smoke model gambar acuan, lalu
  smoke Gemini asli dan perbaikan bila ada ketidakcocokan integrasi.
- Model, dataset, wayang_db.json asli, dan ChromaDB sumber tidak diubah.
  Tidak membuat frontend/Streamlit atau deployment.
