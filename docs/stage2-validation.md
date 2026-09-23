# Validasi Tahap 2 — 23 September 2026

**IMPLEMENTASI LOKAL SELESAI — VALIDASI COLAB MENUNGGU**

Workspace berada di branch `main`. Pada awal sesi, implementasi Tahap 2,
modul persiapan Tahap 3, dan test keduanya sudah tersedia. README.md dan
todolist.md sudah memiliki perubahan pengguna. README tidak diedit sesi ini;
todolist hanya dikoreksi pada baseline 25 menjadi 23. Tidak ada commit/push.

Audit Tahap 1 mempertahankan loader model, processor, classifier, dan JSON
sekali per lifespan; `.eval()`, `torch.inference_mode()`, normalisasi L2,
softmax(logits / temperature), CPU fallback, mapping nested legacy, serta
ranking sebelum pembulatan. `app/resources.py` dan `app/inference.py` tidak diubah.

Perbaikan sesi ini: aplikasi default tidak mendaftarkan `/chat`, tidak memuat
client Gemini dan tidak membaca GEMINI_API_KEY; katalog mensyaratkan nama dari
JSON; lexical menambahkan fuzzy keyword; smoke Colab memverifikasi seluruh
isi katalog, top-1, deduplikasi, evidence, dan formula 55/45 pada respons HTTP.
Kode Tahap 3 lama tetap disimpan, dengan DI eksplisit untuk test lama.

## Hasil lokal (mock)

Interpreter: `.venv/Scripts/python.exe`. Python sistem tidak memiliki pytest.

| Pemeriksaan | Hasil |
| --- | --- |
| Regresi Tahap 1: test_api, test_resources, test_colab_smoke, test_public | 33 lulus |
| Seluruh `pytest -q`, termasuk test mock lama Tahap 3 | 140 lulus, 3 warning |
| `compileall app tests scripts evaluation main.py` | Lulus |
| `pip check` | No broken requirements found |
| `scripts/evaluate_search.py --mock` | 16/16 kasus lulus |

Warning: dua deprecation dependency TestClient dan satu warning Pillow yang
sengaja dipicu test batas resolusi. Tidak ada panggilan Gemini asli.

Query **“Coba berikan saya contoh tokoh Wayang Saksak?”** diuji melalui HTTP
dengan mock collection/embedder: mode `catalog`, tepat 20 ID unik berurutan
0–19, nama sama dengan fixture JSON (termasuk Darun Dia), tanpa encode semantic.
Ini bukti perilaku lokal, bukan validasi isi wayang_db.json asli.

Threshold 0.457 masih hasil kalibrasi fixture: maksimum negatif 0.220,
minimum positif 0.694. Bobot tetap 55% semantic + 45% lexical.

## Validasi Colab yang masih diperlukan

Model, JSON asli, dan ChromaDB tidak tersedia di workspace Windows.
Smoke artefak asli **belum dijalankan**; format metadata, metric, normalisasi
embedding, ranking dan threshold produksi belum dapat dipastikan.
Tidak ada model/dataset/wayang_db.json/ChromaDB asli yang diubah.

Jalankan dari root proyek di Colab setelah Drive tersedia:

```bash
python scripts/colab_search_smoke_test.py --calibrate
# Set WAYANG_SEARCH_THRESHOLD sesuai hasil kalibrasi, lalu:
python scripts/colab_search_smoke_test.py
python scripts/colab_smoke_test.py --image /content/path/gambar-acuan.jpg --tolerance 1e-5
```

Skrip search memakai snapshot sementara agar PersistentClient tidak menulis
ke database sumber, serta membandingkan hash JSON/Chroma sebelum dan sesudah.
Hentikan penulis database lain saat snapshot diambil. Skrip akan gagal jika
metric/metadata tidak dikenali, count bukan 47, ranking tidak cocok, atau skor
positif/negatif tidak terpisah saat kalibrasi; kelulusan tidak dipaksakan.
