# Tahap 2: kontrak dan validasi retrieval

Aplikasi default hanya mengaktifkan `/health`, `/predict-image`, dan `/search`.
`/chat` tidak terdaftar dan `Settings.from_env()` tidak membaca GEMINI_API_KEY.
Modul/test Tahap 3 yang telah ada dipertahankan; jalur dependency injection
eksplisitnya hanya dipakai test lama. Tidak ada panggilan Gemini asli.

`POST /search` menerima `{"query":"Apa fungsi Gunungan?","top_k":3}`.
Query 1–1000 karakter, minimal satu huruf/angka; top_k integer 1–5.
Respons Pydantic memiliki discriminator `mode`:

- `catalog`: permintaan daftar/contoh/pilihan; 20 ID 0–19 langsung dari JSON.
- `answer`: hasil unik per tokoh, score, semantic_score, lexical_score, evidence,
  evidence_ids, dan profil publik yang sama dengan prediksi gambar.
- `fallback_catalog`: bukti di bawah threshold/permintaan mengarang tokoh;
  pesan keterbatasan dan 20 pilihan JSON, tanpa jawaban umum rekaan.

Contoh request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Coba berikan saya contoh tokoh Wayang Saksak?"}'
```

422: input invalid; 503: retrieval belum siap/gagal query; 500: error internal
yang tidak diekspos. Katalog hanya membutuhkan JSON sehingga masih dapat
diakses jika Chroma tidak siap. `/health` tetap melaporkan kesiapan klasifikasi
seperti Tahap 1 dan menambahkan `retrieval_loaded`, `collection_name`,
`document_count`. HTTP 200 health bukan jaminan fitur retrieval siap;
konsumen search perlu memeriksa `retrieval_loaded`.

## Resource dan sumber data

Embedder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` dimuat
satu kali pada lifespan, bersama client dan collection. Cache model pertama
kali dapat membutuhkan unduhan internet. `WAYANG_NORMALIZE_EMBEDDINGS=false`
adalah default encode SentenceTransformer; cocokkan dengan kode pembuat
collection asli sebelum mengklaim hasil integrasi tervalidasi.

`WAYANG_CHROMA_DIR` default menunjuk root Drive yang diberikan pengguna,
`WAYANG_COLLECTION=wayang_sasak`. Direktori wajib berisi `chroma.sqlite3`.
Karena PersistentClient dapat menulis/migrasi saat dibuka, sumber disalin ke
direktori sementara; client hanya membuka snapshot tersebut. Tidak ada
create/add/upsert/delete/reset collection, dan tidak membangun ulang embedding.
Hentikan penulis database lain ketika mengambil snapshot agar file SQLite/HNSW
konsisten. Snapshot memerlukan ruang disk tambahan sebesar database dan bisa
dimigrasi oleh versi client; kompatibilitas database lama tetap perlu smoke test.

Startup memeriksa count > 0, metric aktual (configuration atau legacy
`hnsw:space`), dan metadata semua chunk. Field ID/nama yang dikenali:
`class_id`, `label_id`, `tokoh_id`, `label`, `id`, `slug`, `class_name`,
`nama`, `name`, `nama_tokoh`, `tokoh`. ID/nama harus cocok unik dengan JSON;
metadata tidak dikenal/bertentangan menonaktifkan retrieval dengan error aman.
Field aktual artefak asli belum tersedia lokal dan tidak diklaim sudah diaudit.

## Ranking

Metric wajib terverifikasi; tidak mengasumsikan cosine ketika metadata kosong.
Konversi skor ke [0,1]:

- cosine: `clamp(1-distance, 0, 1)`; similarity negatif menjadi nol.
- squared L2: `1/(1+distance)`.
- inner product: `sigmoid(1-distance)` karena dot product tidak selalu [0,1].

Skor ini ukuran ranking, bukan probabilitas. Query mengambil min(count,
max(20, 8*top_k)) chunk lalu max semantic score per tokoh. Lexical memakai
coverage token query, bukan panjang/frekuensi deskripsi: exact nama/label/alias
= 1; fuzzy nama (ratio >= 88) = 0.9*ratio/100; coverage keyword *0.85;
coverage deskripsi/filosofi/ciri *0.65. Fuzzy keyword memakai token minimal
5 karakter, ratio >= 88, dan coverage berbobot similarity *0.75 (lebih rendah
dari keyword exact). Nilai terbesar menjadi lexical_score.
Test mencakup Jayagrana/Gununggan dan string acak; ambang fuzzy ini aturan
terbatas untuk nama, bukan ambang relevansi semantic.

`score = 0.55*semantic_score + 0.45*lexical_score`. Ranking dilakukan sebelum
serialisasi dan setiap tokoh hanya muncul sekali. Evidence publik adalah
kutipan terpilih dari field publik JSON milik tokoh hasil retrieval (maksimal
3 kutipan x 500 karakter), bukan dump chunk/metadata Chroma. ID evidence
`profile:<id>:<field>:<index>` menunjuk sumber JSON yang bisa ditelusuri.

## Kalibrasi dan keterbatasan

```bash
python scripts/evaluate_search.py --mock
python scripts/colab_search_smoke_test.py --calibrate
```

Evaluation set tersimpan dalam `evaluation/search_cases.json`. Contoh ciri dan
filosofi sintetis diberi `mock_only` dan tidak dijalankan pada data asli.
Smoke Colab menambahkan query ciri dari JSON aktual, tanpa menebak fakta.
Kalibrator memastikan top-1 positif benar lalu mengambil midpoint antara
skor negatif tertinggi dan skor positif terendah. Jika tidak ada gap, proses
gagal; threshold tidak dipaksakan. Bobot 55/45 tidak diubah.

Threshold default hanya berdasarkan fixture skor mock untuk menguji mekanisme.
Nilai terpilih **0.457**: midpoint antara maksimum negatif **0.220** dan
minimum positif **0.694**. Evaluation set yang tersedia memuat 16 kasus
(9 positif, 4 negatif, 2 katalog, 1 invalid).
Ini bukan validasi relevansi embedding asli. Hasil kalibrasi Colab dicetak ke terminal;
set `WAYANG_SEARCH_THRESHOLD` ke nilai tersebut dan jalankan smoke lagi tanpa
`--calibrate` untuk memeriksa konfigurasi deployment. Skrip tidak menyimpan
atau menimpa artefak. Set kecil ini belum membuktikan kualitas generalisasi;
perlu perluasan query evaluasi sebelum deployment publik.

Dasar API: [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure),
[Chroma client](https://docs.trychroma.com/reference/python/client),
[SentenceTransformer encode](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
