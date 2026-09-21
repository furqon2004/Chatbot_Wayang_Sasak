TODO — Backend FastAPI Tahap 1: Prediksi Gambar Wayang Sasak

Tujuan tahap ini

Bangun backend FastAPI dasar untuk memuat model CLIP ViT-B/32 + linear classifier satu kali saat aplikasi dimulai, menyediakan endpoint pemeriksaan sistem, dan menerima gambar untuk memprediksi satu dari 20 tokoh Wayang Sasak.

Tahap ini hanya mencakup:

GET /health

POST /predict-image

integrasi model gambar dengan wayang_db.json

pengujian dasar kedua endpoint

Jangan mengerjakan hybrid retrieval, endpoint /search, Gemini, endpoint /chat, atau Streamlit pada tahap ini.

Cara menjalankan tugas ini di Codex

Workspace Codex kemungkinan hanya berisi file todolist.md. Itu normal dan bukan berarti artefak proyek hilang. Artefak model berada di Google Drive dan baru tersedia ketika kode dijalankan di Google Colab.

Codex harus mengikuti aturan berikut:

Buat source code backend lengkap di workspace/repository saat ini agar dapat diunduh atau dipindahkan ke Google Colab.

Jangan mencari, menyalin, membuat ulang, atau mengganti model asli di workspace lokal.

Jangan mengubah path produksi hanya karena /content/drive/... tidak tersedia di lingkungan Codex.

Kode produksi harus menggunakan default root /content/drive/MyDrive/skripsi_wayang, tetapi boleh menerima override melalui environment variable untuk pengujian lokal.

Unit test lokal boleh memakai dependency injection, temporary directory, dan model tiruan untuk menguji kontrak API serta error handling.

Test dengan model tiruan bukan bukti bahwa inferensi model asli sudah benar. Pisahkan laporan unit test lokal dari validasi integrasi Colab.

Jangan mengarang hasil /health, /predict-image, confidence, top-3, atau waktu inferensi model asli.

Siapkan satu skrip atau notebook smoke test yang akan dijalankan pengguna di Google Colab setelah Google Drive di-mount.

Jika implementasi bergantung pada informasi yang tidak tersedia—khususnya rumus temperature dari training/evaluasi—berhenti pada bagian tersebut, tandai sebagai blocker, dan minta pengguna memberikan potongan kode yang relevan. Jangan memilih rumus berdasarkan konvensi umum.

Deliverable tahap ini tetap berupa kode backend dan test. Keterbatasan akses Codex ke Google Drive tidak boleh dijadikan alasan untuk hanya menulis rencana, tetapi juga tidak boleh ditutupi dengan asumsi atau hasil pengujian palsu.

Konteks proyek yang wajib dipertahankan

Root Google Drive: /content/drive/MyDrive/skripsi_wayang

Folder model: /content/drive/MyDrive/skripsi_wayang/model_clip_b32_terbaik

File basis pengetahuan: /content/drive/MyDrive/skripsi_wayang/wayang_db.json

Base model: openai/clip-vit-base-patch32

Arsitektur: CLIPVisionModelWithProjection + linear classifier

Ukuran input model: 224 × 224

Dimensi image embedding: 512

Jumlah kelas: 20

Normalisasi embedding: L2 normalization

Temperature dari model_config.json: 0.21597230434417725

Perangkat: gunakan CUDA jika tersedia, jika tidak gunakan CPU

Model dan data harus dimuat satu kali saat startup, bukan setiap request

Jangan mengubah, menghapus, melatih ulang, atau membangun ulang dataset, model, JSON, maupun ChromaDB

Nama folder dataset_tryning memang salah eja dan tidak boleh diganti

Struktur artefak model

model_clip_b32_terbaik/
├── vision_encoder/
│   ├── config.json
│   └── model.safetensors
├── image_processor/
│   └── preprocessor_config.json
├── classifier_head.pt
├── label_mapping.json
└── model_config.json

Urutan label resmi

0  jayangrana
1  umar_maya
2  maktal
3  taptanus
4  samtanus
5  selandir
6  baktak
7  umar_madi
8  smakun
9  hermus
10 darun_die
11 ibnu_jarah
12 kius_nendar
13 munigarim
14 nursiwan
15 lurah
16 kembung
17 gulanggi
18 samardikaran
19 gunungan

Urutan ini wajib konsisten dengan label_mapping.json, output classifier, wayang_db.json, dan respons API. Jangan membuat mapping baru berdasarkan urutan folder.

Checklist implementasi

1. Persiapan proyek backend

Periksa isi root proyek dan pertahankan file milik pengguna yang sudah ada.

Buat struktur backend yang sederhana dan mudah dilanjutkan pada tahap berikutnya.

Buat requirements.txt dengan dependency yang benar-benar diperlukan.

Gunakan environment variable untuk konfigurasi path apabila sesuai, tetapi sediakan default path Google Drive di atas.

Jangan memasukkan API key atau rahasia apa pun ke source code.

Dependency minimum yang kemungkinan diperlukan:

fastapi
uvicorn[standard]
python-multipart
torch
transformers
pillow

2. Konfigurasi dan validasi artefak

Definisikan path model dan wayang_db.json dari satu konfigurasi terpusat.

Pastikan seluruh artefak model wajib tersedia sebelum server dinyatakan sehat.

Baca model_config.json; jangan hard-code temperature jika nilai tersedia dari file.

Baca label_mapping.json dan validasi bahwa mapping berisi tepat 20 kelas dengan ID 0–19.

Baca wayang_db.json dan validasi bahwa terdapat tepat 20 tokoh.

Sesuaikan pembacaan JSON dengan format aktual: field label dapat berisi string angka seperti "0" sampai "19".

Buat indeks data tokoh berdasarkan ID agar hasil classifier dapat dipetakan tanpa pencarian ambigu.

Jika artefak tidak lengkap atau tidak konsisten, gagalkan startup dengan pesan error yang jelas.

3. Pemuatan model satu kali

Gunakan mekanisme lifespan/startup FastAPI untuk memuat semua resource satu kali.

Muat CLIPVisionModelWithProjection dari folder vision_encoder dengan local_files_only=True.

Muat CLIPImageProcessor dari folder image_processor dengan local_files_only=True.

Muat classifier_head.pt secara aman ke device aktif.

Tangani format checkpoint aktual yang berisi bobot (20, 512) dan bias (20,).

Bentuk torch.nn.Linear(512, 20) bila checkpoint berupa state_dict, lalu muat bobotnya.

Pindahkan vision encoder dan classifier ke device yang sama.

Aktifkan mode evaluasi dengan .eval().

Matikan gradient pada inferensi menggunakan torch.inference_mode().

Simpan model, processor, mapping, konfigurasi, dan data JSON pada state aplikasi yang dapat digunakan endpoint.

Jangan memuat ulang model di dalam fungsi endpoint.

4. Pipeline inferensi gambar

Terima file upload melalui UploadFile.

Validasi tipe file gambar yang diperbolehkan, misalnya JPEG, PNG, dan WebP.

Tetapkan batas ukuran upload yang wajar dan tolak file terlalu besar.

Baca gambar dengan Pillow dan ubah ke RGB.

Tolak file rusak atau file yang bukan gambar dengan HTTP 400 atau 415 yang sesuai.

Gunakan CLIPImageProcessor tersimpan untuk resize/crop/normalisasi sesuai konfigurasi model.

Pindahkan pixel_values ke device model.

Ambil image_embeds berdimensi 512 dari vision encoder.

Terapkan L2 normalization pada embedding dengan torch.nn.functional.normalize(..., dim=-1).

Masukkan embedding ke linear classifier.

Terapkan temperature hanya setelah rumusnya diverifikasi dari kode training/evaluasi model aktif.

Hitung probabilitas menggunakan softmax.

Ambil top-3 prediksi.

Hitung confidence top-1 dan margin antara probabilitas top-1 dan top-2.

Pastikan seluruh nilai probabilitas yang dikirim sebagai JSON sudah dikonversi menjadi tipe Python biasa.

Ambil profil tokoh top-1 dari wayang_db.json berdasarkan ID kelas.

Aturan wajib: nilai temperature 0.21597230434417725 tidak memberi tahu arah operasinya. Jangan memakai logits / temperature hanya karena itu konvensi umum, dan jangan pula memakai perkalian tanpa bukti. Temukan ekspresi aktual pada kode training/evaluasi, misalnya softmax(logits / temperature) atau bentuk lain. Jika kode tersebut tidak tersedia, buat fungsi temperature yang eksplisit dan terkonfigurasi, tandai integrasi belum tervalidasi, lalu minta pengguna memberikan kode evaluasi. Endpoint tidak boleh dinyatakan cocok dengan model asli sebelum hal ini diverifikasi.

5. Endpoint GET /health

Buat endpoint GET /health.

Respons minimal menampilkan status aplikasi, status model, nama arsitektur, device, jumlah kelas, dan ketersediaan basis pengetahuan.

Jangan membocorkan path sensitif atau stack trace pada respons publik.

Jika model belum siap, gunakan status HTTP yang mencerminkan bahwa layanan belum siap.

Contoh bentuk respons:

{
  "status": "ok",
  "model_loaded": true,
  "model_name": "CLIP ViT-B/32 + linear classifier",
  "device": "cuda",
  "num_classes": 20,
  "knowledge_base_loaded": true
}

6. Endpoint POST /predict-image

Buat endpoint POST /predict-image dengan parameter file gambar.

Berikan status HTTP dan pesan yang jelas untuk upload kosong, format salah, gambar rusak, dan kegagalan inferensi.

Respons sukses harus memuat prediksi top-1, top-3, confidence, margin, serta profil tokoh top-1.

Gunakan skala probabilitas yang konsisten; disarankan angka desimal 0–1, dengan pembulatan hanya pada lapisan respons.

Hindari klaim bahwa confidence adalah kepastian objektif.

Jangan meminta Gemini atau ChromaDB untuk endpoint ini.

Contoh bentuk respons:

{
  "prediksi": {
    "id": 0,
    "label": "jayangrana",
    "nama": "Jayangrana",
    "confidence": 0.87,
    "margin": 0.31
  },
  "top_3": [
    {
      "id": 0,
      "label": "jayangrana",
      "nama": "Jayangrana",
      "probability": 0.87
    }
  ],
  "profil": {
    "deskripsi": "...",
    "deskripsi_lengkap": "...",
    "filosofi": "...",
    "ciri_ciri": [],
    "kata_kunci": []
  }
}

Format akhir boleh disempurnakan, tetapi nama field harus konsisten dan didokumentasikan.

7. Keandalan dan keamanan dasar

Tambahkan logging ringkas untuk startup, device, waktu inferensi, dan error tanpa mencetak data sensitif.

Jangan mengembalikan traceback internal kepada pengguna API.

Pisahkan error input pengguna dari error internal server.

Tambahkan CORS hanya jika benar-benar diperlukan; jangan gunakan konfigurasi terbuka tanpa alasan.

Pastikan file upload diproses di memori atau file sementara yang aman dan tidak dibiarkan menumpuk.

Jangan melakukan operasi tulis pada model, dataset, JSON, atau ChromaDB.

8. Pengujian

Tambahkan unit test lokal untuk startup/resource loading menggunakan temporary artifact atau model tiruan.

Uji kontrak GET /health; HTTP 200 hanya boleh diharapkan ketika resource test dinyatakan siap.

Uji kontrak POST /predict-image secara lokal menggunakan model tiruan dan sedikitnya satu gambar valid.

Buat smoke test integrasi terpisah untuk Google Colab yang menggunakan artefak asli dari Google Drive.

Pada validasi Colab, bandingkan hasil endpoint dengan hasil skrip evaluasi untuk gambar yang sama: ID kelas, urutan top-3, confidence, dan margin harus konsisten dalam toleransi numerik kecil.

Uji gambar PNG dan JPEG.

Uji upload file teks yang menyamar sebagai gambar.

Uji file gambar rusak.

Uji request tanpa file.

Uji perilaku ketika salah satu artefak model tidak tersedia menggunakan mock/path sementara; jangan memindahkan atau menghapus artefak asli.

Pastikan dua request berurutan tidak memuat ulang model.

Pastikan inferensi berjalan pada CPU ketika CUDA tidak tersedia.

Dalam laporan tes, beri label yang jelas antara LULUS — unit test dengan mock dan BELUM DIJALANKAN — integrasi model asli di Colab.

9. Dokumentasi penggunaan

Tambahkan petunjuk instalasi dependency.

Tambahkan perintah menjalankan server, misalnya uvicorn main:app --host 0.0.0.0 --port 8000.

Tambahkan contoh request curl untuk /health dan /predict-image.

Jelaskan struktur respons dan error utama.

Catat bahwa tahap retrieval, Gemini, dan Streamlit belum termasuk.

Kriteria selesai

Tahap ini dinyatakan selesai hanya jika:

Server dapat dimulai tanpa training atau rebuild artefak.

Model, classifier, mapping label, konfigurasi, dan JSON hanya dimuat satu kali.

/health mengonfirmasi model dan knowledge base siap.

/predict-image menerima gambar valid dan menghasilkan top-1 serta top-3 untuk 20 kelas.

Mapping hasil prediksi ke tokoh JSON benar.

Pipeline inference menggunakan RGB, processor CLIP, embedding 512, L2 normalization, classifier, dan temperature yang sama dengan evaluasi.

Hasil backend cocok dengan skrip evaluasi pada gambar acuan.

Kasus input tidak valid menghasilkan respons HTTP yang tepat tanpa crash.

Semua tes yang relevan lulus.

Tidak ada dataset, model, JSON, atau ChromaDB yang berubah.

Jika artefak asli atau kode evaluasi belum tersedia di workspace, Codex boleh menyelesaikan implementasi dan unit test lokal, tetapi tidak boleh mencentang kriteria integrasi model asli, kecocokan temperature, maupun kecocokan hasil backend. Status yang benar adalah: IMPLEMENTASI LOKAL SELESAI — VALIDASI COLAB MASIH MENUNGGU. 