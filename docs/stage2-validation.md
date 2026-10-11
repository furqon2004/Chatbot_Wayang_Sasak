> Catatan migrasi OpenAI: uraian provider/konfigurasi lama di bawah merupakan
> catatan historis. Implementasi aktif memakai OpenAI untuk narasi dan web;
> lihat [konfigurasi chat](chat.md). Hasil audit lama bukan validasi migrasi terbaru.

﻿# Validasi Tahap 2 — 23 September 2026

**TAHAP 2 SELESAI — VALIDASI ARTEFAK ASLI COLAB LULUS SETELAH KALIBRASI**

Branch: `feat/stage2-search`. Hasil artefak asli di bawah berasal dari
validasi Google Colab yang dilaporkan pengguna; artefak tersebut tidak tersedia
untuk dijalankan ulang di workspace Windows. Finalisasi lokal tidak melakukan
commit/push.

Baseline aktif kini **48 chunk**, setelah pembaruan Gunungan secara sengaja
oleh proyek. Pengguna melaporkan smoke retrieval baru lulus dengan
`--expected-count 48`; angka kalibrasi di bawah tetap catatan historis, bukan
pengukuran ulang atas artefak baru. Lihat [baseline dan provenance](retrieval.md).

## Hasil artefak asli Google Colab

| Parameter / pemeriksaan | Hasil |
| --- | --- |
| Chroma metric | `cosine` |
| `document_count` saat validasi historis Tahap 2 | 47 (baseline aktif kini 48) |
| Threshold kalibrasi | `0.5590755322575569` |
| `negative_max` | `0.49093165576457987` |
| `positive_min` | `0.6272194087505341` |
| `positive_count` | 8 |
| `negative_count` | 4 |
| Evaluation setelah kalibrasi | 15/15 case lulus |
| Hash `wayang_db.json` dan ChromaDB sebelum/sesudah | Tidak berubah |

Threshold merupakan midpoint antara maksimum negatif dan minimum positif.
Nilai ini masih **initial calibrated threshold**: query calibration dan
evaluation belum menggunakan holdout terpisah. Kelulusan 15/15 belum menjadi
bukti generalisasi pada query baru. Validasi lanjutan perlu holdout independen
yang tidak digunakan untuk memilih threshold, termasuk query ambigu dan
variasi bahasa pengguna.

## Konfigurasi dan batas perubahan

`app/config.py` menetapkan `DEFAULT_SEARCH_THRESHOLD = 0.5590755322575569`
sebagai default `Settings.search_threshold`. `Settings.from_env()` tetap
menerima override melalui `WAYANG_SEARCH_THRESHOLD`; `.env.example` memuat
nilai default tersebut tanpa rahasia. Retrieval service dan smoke script
menggunakan konfigurasi, tanpa literal threshold baru di keduanya.

Bobot hybrid tetap **55% semantic + 45% lexical**. Model, processor,
classifier, hasil inferensi gambar, JSON, ChromaDB, dan dataset tidak diubah.
Aplikasi default tidak mendaftarkan `/chat`; finalisasi ini hanya mencatat
kebutuhan keamanan Tahap 3 di `todolist.md`, tanpa implementasi Gemini/chat.

Profil publik `ciri_ciri` tetap string: input string dipertahankan, list
dinormalisasi menjadi string, null/missing menjadi string kosong.
`kata_kunci` tetap list. Smoke memilih teks ciri maksimal 200 karakter dan
mencetak report JSON sebelum pemeriksaan; kegagalan dicetak sebagai ringkasan
expected/actual serta skor bila tersedia, lalu keluar non-zero.

## Hasil pemeriksaan lokal finalisasi

Interpreter: `.venv/Scripts/python.exe`.

| Pemeriksaan | Hasil |
| --- | --- |
| Seluruh `pytest`, termasuk test mock lama Tahap 3 | 165 lulus, 3 warning |
| `compileall app tests scripts evaluation main.py` | Lulus |
| `pip check` | No broken requirements found |

Regression test mencakup default Settings, override environment, query
`sesuatu` menjadi `fallback_catalog`, serta query relevan terlemah pada fixture
tetap `answer`. Replay batas skor Colab juga memeriksa bahwa `negative_max`
ditolak dan `positive_min` diterima; replay ini menggunakan mock ranking,
bukan reproduksi embedding atau identitas query minimum asli.
Kalibrasi fixture sintetis tetap diuji terpisah dari default artefak asli.

Warning berasal dari dua deprecation dependency TestClient dan satu warning
Pillow yang sengaja dipicu test batas resolusi. Tidak ada panggilan Gemini asli.

## Menjalankan ulang di Colab

Jalankan dari root proyek setelah Drive tersedia. Pastikan environment tidak
menyimpan override threshold lama:

```bash
export WAYANG_SEARCH_THRESHOLD=0.5590755322575569
python scripts/colab_search_smoke_test.py
# Hanya bila hendak mengukur ulang kalibrasi:
python scripts/colab_search_smoke_test.py --calibrate
```

Smoke memakai snapshot sementara agar PersistentClient tidak menulis ke
ChromaDB sumber dan membandingkan hash JSON/Chroma sebelum dan sesudah.
Hentikan penulis database lain saat snapshot diambil. Kelulusan evaluasi
tidak dipaksakan dengan mengubah expected case atau bobot ranking.
