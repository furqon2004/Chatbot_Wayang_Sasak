# Backend Wayang Sasak

Backend FastAPI ini memuat `CLIPVisionModelWithProjection`, image processor,
linear classifier, label mapping, dan basis pengetahuan tepat satu kali saat
startup. Prediksi gambar memiliki [decision guard dan kontrak uncertain](docs/vision-confidence-guard.md)
dengan threshold confidence 0.85 dan margin 0.20; profil hanya diberikan saat accepted.
Kontrak `GET /health` dipertahankan;
Tahap 2 menambahkan `POST /search`. Persiapan Tahap 3 (`POST /chat`) sudah
tersedia dengan narasi OpenAI, pencarian web OpenAI, dan fallback lokal.
Lihat konfigurasi aktif pada dokumentasi chat.

Dokumentasi tambahan: [retrieval dan kalibrasi](docs/retrieval.md),
[chat, grounding, dan OpenAI](docs/chat.md),
[laporan milestone](docs/milestones.md).

## Struktur

```text
app/
  api.py          # lifespan dan endpoint
  config.py       # path serta batas upload
  inference.py    # pipeline inferensi
  resources.py    # validasi dan pemuatan artefak
scripts/
  colab_smoke_test.py
tests/
main.py
```

## Instalasi dan menjalankan server

### Setelah clone di Windows (PowerShell)

Gunakan Python 3.10 atau lebih baru dan environment terpisah:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Edit .env: arahkan model dan database ke artefak asli jika menjalankan lokal.
.\.venv\Scripts\python.exe -m uvicorn main:app --env-file .env --host 127.0.0.1 --port 8000
```

`.venv` adalah folder dependensi Python; `.env` adalah file konfigurasi.
Keduanya tidak disimpan di Git. `.env.example` menyediakan template tanpa rahasia.
File `.env` dibaca melalui opsi Uvicorn `--env-file`, bukan otomatis oleh
`Settings.from_env()`. Environment sistem yang sudah disetel tetap diprioritaskan.
Prediksi gambar dan search tidak membutuhkan API key. Chat memakai
`LLM_API_KEY` jika tersedia, atau fallback deterministik jika kosong.

Model dan database tidak ikut clone. Tanpa artefak asli, startup akan gagal
dengan pesan artefak yang hilang; menyalin `.env` saja tidak menyediakan model.
Untuk unit test dengan model tiruan, artefak asli dan `.env` tidak diperlukan:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Gunakan satu worker sebagai awal agar model tidak diduplikasi ke RAM/VRAM.
Rentang versi dependency belum merupakan lockfile; performa dan kompatibilitas
deployment perlu divalidasi pada environment Colab yang digunakan.

### Google Colab

Di Google Colab, mount Drive lebih dulu, pindah ke folder source code ini, lalu:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Default produksi tidak berubah:

- model: `/content/drive/MyDrive/skripsi_wayang/model_clip_b32_terbaik`
- knowledge base: `/content/drive/MyDrive/skripsi_wayang/wayang_db.json`

Override opsional untuk test/deployment lain:

```bash
export WAYANG_MODEL_DIR=/path/ke/model_clip_b32_terbaik
export WAYANG_DB_PATH=/path/ke/wayang_db.json
export WAYANG_MAX_UPLOAD_BYTES=10485760
```

### Temperature scaling

Implementasi memakai rumus berikut dari `Untitled33.ipynb`. Status awal di
`todolist.md` menyatakan rumus ini telah diverifikasi di Colab sebelumnya:

```python
logits_raw = classifier(image_embeddings)
logits_scaled = logits_raw / temperature
probabilities = torch.softmax(logits_scaled, dim=-1)
```

Temperature selalu dibaca dari `model_config.json` (nilai model aktif saat
dokumentasi ini ditulis: `0.21597230434417725`). Backend tidak memiliki mode
perkalian dan tidak memakai `SUPCON_TEMPERATURE=0.10`, karena nilai tersebut
hanya digunakan oleh supervised contrastive loss saat training.

## Pemakaian API

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict-image \
  -F "file=@/path/ke/gambar-wayang.jpg"
```

`/health` mengembalikan status kesiapan, arsitektur, device, jumlah kelas, dan
status knowledge base tanpa menampilkan path lokal. `/predict-image`
mengembalikan:

- `prediksi`: ID, label, nama, confidence top-1, dan margin top-1/top-2;
- `top_3`: tiga ID/label/nama beserta probabilitas desimal 0–1;
- `profil`: profil tokoh top-1 dari `wayang_db.json`.

Profil publik hanya berisi `deskripsi`, `deskripsi_lengkap`, `filosofi`,
`ciri_ciri`, `kata_kunci`, dan `gambar_utama` bila aman tersedia. Field training,
embedding, metadata internal, serta daftar semua gambar tidak lagi diekspos.
Gambar publik menerima path relatif `dataset_asli/...` atau HTTPS tanpa token;
path absolut lokal/Drive tidak dikirim. Parser mapping mendukung object nested
`id2label`/`label2id` (termasuk arah legacy terbalik), `id_to_label`/`label_to_id`,
dan `idx_to_class`/`class_to_idx`, dengan urutan resmi tetap divalidasi.

`confidence` adalah probabilitas keluaran model, bukan kepastian objektif.
Angka dibulatkan menjadi enam desimal hanya pada respons API.

Error utama:

| HTTP | Kondisi |
|---:|---|
| 400 | upload kosong, rusak, atau bukan data gambar valid |
| 413 | ukuran melewati `WAYANG_MAX_UPLOAD_BYTES` atau resolusi gambar terlalu besar |
| 415 | MIME/format bukan JPEG, PNG, atau WebP |
| 422 | field multipart `file` tidak diberikan |
| 500 | kegagalan inferensi internal (tanpa traceback pada respons) |
| 503 | resource belum siap |

## Pengujian lokal

```bash
pip install -r requirements-dev.txt
pytest
```

Unit test menggunakan model tiruan dan temporary artifact. Test mencakup
startup/load satu kali, health, PNG/JPEG, RGB, top-3 dan probabilitas fixture,
upload invalid/kosong/terlalu besar, artefak hilang, checkpoint, pembagian temperature,
error internal, dan fallback CPU. Kelulusan unit test bukan bukti kecocokan
dengan model asli.

## Smoke test integrasi Google Colab

Setelah Drive di-mount, jalankan validasi artefak asli dengan sebuah gambar:

Pasang `requirements-dev.txt` untuk smoke test karena skrip memakai `httpx`
melalui `TestClient`. Skrip ini membaca environment sistem, bukan file `.env`.

```bash
python scripts/colab_smoke_test.py \
  --image /content/path/gambar-acuan.jpg \
  --tolerance 1e-5
```

Skrip menghitung pembanding independen dengan urutan operasi notebook. Untuk
perbandingan tambahan dengan output `Untitled33.ipynb`, simpan JSON referensi
dengan field `prediksi.id`, `prediksi.confidence`, `prediksi.margin`, serta
`top_3[].id` dan `top_3[].probability`, lalu tambahkan:

```bash
--reference-json /content/path/hasil-evaluasi.json
```

Skrip memakai artefak asli, memeriksa `/health`, lalu membandingkan ID top-1,
urutan dan probabilitas top-3, confidence, serta margin. Hasil aktual baru
dicetak saat dijalankan; dokumentasi ini tidak mengarang output model.

Notebook `Untitled33.ipynb` tidak disertakan dalam repository. Validasi Colab
sebelumnya diterima sebagai baseline pengguna, tetapi eksekusi ulang sesi ini
tetap memerlukan artefak asli. Gunakan `--reference-json` dengan keluaran notebook
untuk memverifikasi kecocokan dengan evaluasi asli; pembanding langsung saja
hanya memeriksa konsistensi dua implementasi dengan rumus yang sama.

## Validasi tambahan

```bash
python -m compileall app tests scripts
python -m pip check
python scripts/evaluate_search.py --mock
python scripts/colab_search_smoke_test.py
python scripts/colab_chat_smoke_test.py
```

## Di luar lingkup

Training, rebuild embedding, dan deployment produksi tidak termasuk.
Dataset, model, JSON, dan ChromaDB sumber tidak ditulis atau diubah.

## Antarmuka Wayang AI

Instal dependensi UI dan jalankan dari root repository:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Backend harus dijalankan pada terminal terpisah. Agar `/chat` tersedia:

```powershell
$env:WAYANG_ENABLE_CHAT = "true"
.venv\Scripts\python.exe -m uvicorn main:app --env-file .env
```

UI membaca `.env` tanpa menimpa environment sistem. `WAYANG_API_URL` default
`http://127.0.0.1:8000`; sesuaikan jika backend berada di host lain.
`WAYANG_SERVICE_TOKEN` dikirim hanya dari server UI ke backend jika dikonfigurasi.
UI tidak membutuhkan akses model atau kunci OpenAI.

Header memeriksa `/health` setiap 30 detik. Input teks dan lampiran berada dalam
satu area dengan tombol Kirim. Teks yang terisi mengunci unggahan; gambar yang
dipilih mengunci teks. Hapus input untuk berganti jenis pesan. Gambar dikirim
sebagai multipart ke `/predict-image`, sedangkan teks dikirim ke `/chat`.
Setelah dikirim, kedua input dikosongkan; kiriman yang berisi keduanya ditolak.
Tombol katalog di sidebar membuka pilihan tokoh di percakapan utama melalui `/search`.
Klik nama tokoh untuk meminta penjelasan melalui `/chat`; pencarian dilakukan melalui input chat.
Riwayat dan hasil pencarian hanya disimpan di session state, tanpa cache global
atau penulisan ke disk. Byte gambar disimpan dalam memori sesi agar unggahan
tetap tampil di riwayat. Percakapan Baru menghapus riwayat, konteks, dan lampiran.
Client tidak mencatat token, pertanyaan, gambar, atau isi respons ke log.
Parser multipart `/predict-image` juga memakai memori tanpa rollover ke file
sementara; batas ukuran request dari middleware tetap berlaku.

Layout memakai komponen native Streamlit dengan pembungkusan baris pada layar
kecil, label kontrol, dan fokus keyboard bawaan. Tidak ada HTML/CSS injection.
Tema mengikuti [konfigurasi native Streamlit](https://docs.streamlit.io/develop/api-reference/configuration/config.toml).
Emas ornamen `#D5A437`; tombol memakai emas gelap `#87651F` untuk menjaga
kontras teks putih. Penempatan, transisi, dan surface mengikuti batas komponen
native; tidak meniru merek atau aset pihak lain.

Pengujian UI (HTTP dimock, tanpa model):

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ui.py -q
```


### Gambar tokoh dan percakapan lanjutan

Jawaban teks menampilkan satu `gambar_utama` dari tokoh yang dijelaskan.
Sediakan folder `dataset_asli` di sebelah `wayang_db.json`, di root proyek,
atau atur `WAYANG_IMAGE_ROOT` ke folder `dataset_asli` yang sebenarnya.
JSON hanya menyimpan referensi; tautan folder Google Drive bukan URL gambar langsung.
Jika file belum tersedia, UI menampilkan keterangan tanpa mengganti gambar tokoh.

Unggahan gambar tetap ditampilkan sekali dalam pesan pengguna. Setelah prediksi
accepted, UI meminta penjelasan `/chat` untuk tokoh yang dikenali; saat layanan
chat gagal, profil dari hasil prediksi tetap ditampilkan. Hasil uncertain tetap
meminta gambar yang lebih jelas dan tidak diklaim sebagai identifikasi final.

LLM memilih evidence penjelasan dan pertanyaan lanjutan yang tervalidasi.
Jawaban `ya` saat ada tawaran lanjutan mengirim `awaiting_more` dan `seen_ids`;
backend memilih tiga tokoh acak berbeda dari katalog resmi, menghindari tokoh
yang telah dibahas selama masih tersedia setidaknya tiga tokoh baru. Tombol pilihan
membuka penjelasan di chat. Pengacakan dilakukan aplikasi, tanpa memerlukan LLM.
Konteks ini hanya berlaku dalam sesi UI dan dihapus oleh Percakapan Baru.


### Sejarah, cerita, dan jumlah tokoh dari internet

Pertanyaan Wayang Sasak tentang sejarah, cerita, atau jumlah tokoh dapat memakai
OpenAI Responses dengan tool `web_search`, melalui `LLM_API_KEY` yang sama.
Backend memeriksa hasil pencarian dan anotasi URL; UI menampilkan sitasi yang
bisa diklik dan daftar sumber. Lihat [konfigurasi chat](docs/chat.md).

Angka 20 merupakan cakupan katalog aplikasi, bukan jumlah seluruh tokoh tradisional.
Jika sumber tidak mendukung angka pasti, aplikasi menyampaikan keterbatasan.
Kutipan bukan verifikasi kebenaran budaya independen. Profil lokal tidak ditimpa web.
