# Catatan validasi milestone

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
