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
