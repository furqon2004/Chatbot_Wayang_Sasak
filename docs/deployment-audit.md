# Audit kesiapan Docker dan VPS — 21 Oktober 2026

## Kesimpulan

Packaging sudah disiapkan untuk satu container berisi FastAPI dan Streamlit,
dengan batas total `1g` RAM dan tanpa swap tambahan. **Kesiapan production dengan
batas tersebut belum terpenuhi.** Pengukuran backend asli di Windows sudah
mencapai 1127,7 MiB RSS setelah smoke HTTP, sebelum proses Streamlit ditambahkan.
Pengukuran Windows bukan pengukuran Linux/cgroup; hasil ini cukup untuk menolak
klaim bahwa aplikasi lengkap sudah terbukti muat dalam 1 GiB.

Audit mencakup source backend/UI, loader model dan retrieval, pengaturan keamanan,
dependency environment lokal, suite test, konfigurasi container, dan artefak asli
yang tersedia. VPS belum diakses dan image Linux belum dibangun. Tidak dilakukan
training, quantization, perubahan threshold, atau modifikasi database/model.

## Temuan berdasarkan prioritas

| Prioritas | Temuan dan bukti | Tindak lanjut |
| --- | --- | --- |
| P1 | Target total 1 GiB tidak memiliki ruang memori yang terbukti cukup. Bobot CLIP 335,1 MiB dan MiniLM 448,8 MiB; backend aktif mencapai 1127,7 MiB RSS di Windows. | Ukur image Linux dalam cgroup 1 GiB sebelum production. Jika OOM, batas itu belum memenuhi kebutuhan model asli. Evaluasi RAM lebih besar atau optimasi model terpisah dengan regresi prediksi/retrieval. |
| P1 | `artifacts/chroma_db_wayang` memiliki 47 dokumen; production menuntut tepat 48 (`app/config.py`, `app/retrieval_resources.py`). Startup production pada folder ini gagal. | Gunakan artefak baseline 48 yang tervalidasi. Folder lokal `chroma_db_wayang_updated_20261005_191158` memiliki 48/cosine dan lolos loader serta smoke asli. Jangan turunkan validasi menjadi 47. |
| P1 | Audit dependency baru gagal: 34 entri advisory, 29 pasangan paket/advisory unik, pada 7 paket. Pemeriksa VEX lama juga gagal karena ada temuan baru. | Tinjau dan remediasi sebelum publikasi. Audit ulang image Linux; hasil environment Windows bukan inventaris final container. Jangan memperluas pengecualian tanpa tinjauan. |
| P1 | Backend membatasi decoder ke JPEG/PNG/WebP, tetapi UI meneruskan byte unggahan ke `st.image` (`streamlit_app.py`) dan memilih MIME berdasarkan ekstensi (`ui_client.py`). Pembatasan backend tidak mencakup seluruh decoding di UI. | Tinjau advisory Pillow terhadap jalur UI dan validasi format/resolusi sebelum decoding preview. Ekstensi file tidak membuktikan format isi. Belum dilakukan eksploitasi atau perubahan alur gambar pada audit ini. |
| P2 | Riwayat UI menyimpan byte gambar dan snapshot base64. TTL 300 detik adalah batas tidak aktif, bukan batas jumlah pesan/byte/session (`ui_history.py`, `streamlit_app.py`). Sesi aktif dan banyak pengguna bisa menambah RAM terus. | Tetapkan anggaran session/history untuk target 1 GiB setelah menyepakati perilaku riwayat. Pengurangan riwayat otomatis belum diterapkan. |
| P2 | Dependency memakai rentang versi; constraints yang ada khusus Windows/Python 3.10. Image menggunakan Linux/Python 3.11. | Buat lock/digest rilis dari build Linux yang sudah lolos validasi. Jangan menganggap versi atau advisory lokal identik dengan image. |
| P2, diperbaiki | CI memasang `requirements-dev.txt`, tetapi sebelumnya file ini tidak menyertakan dependency UI. `tests/test_ui_history.py` mengimpor Streamlit melalui `ui_history.py`. | `requirements-dev.txt` kini menyertakan `requirements-ui.txt`, sehingga instalasi environment test bersih juga memasang dependency UI yang sudah digunakan proyek. Pipeline tetap perlu dijalankan di Linux. |
| P2 | Model aktif best epoch 120 memiliki `temperature: null`, sehingga loader memakai 1.0. Smoke HTTP tidak mengukur kelayakan threshold acceptance pada model ini. | Jalankan regresi gambar referensi dan evaluasi guard model aktif sebelum go-live; jangan mengubah temperature/threshold hanya untuk menghemat RAM. |
| P2 | Model embedding production harus tersedia offline dan cocok dengan embedding Chroma. Cache belum ditransfer/dites di VPS. | Transfer cache terpercaya, pertahankan struktur Hub, verifikasi fingerprint/revision, lalu uji retrieval asli. |
| P2 | Pembatas request berjalan per proses; UI memakai satu token backend dan alamat loopback bersama. Ini bukan kuota per akun pengunjung. | Satu worker dipertahankan. Tinjau rate limit/gateway dan biaya OpenAI sebelum trafik publik. |
| P3 | Docker Desktop tidak ada di dua lokasi instalasi standar Windows yang diperiksa. Executable WSL ada, tetapi `wsl --status` menyatakan fitur belum terpasang. | Build dapat dilakukan langsung di VPS Ubuntu setelah Docker Engine dan Compose plugin tersedia. |

## Hasil pengukuran dan pemeriksaan

Pengukuran memakai CPU, satu thread OMP/MKL, artefak asli, cache embedding lokal,
dan snapshot Chroma sementara. Source Chroma tidak dibuka dengan PersistentClient
secara langsung; loader menyalinnya terlebih dahulu. Gambar uji berupa PNG putih,
bukan evaluasi akurasi model. Mode chat memakai fallback tanpa kunci OpenAI.

| Tahap backend (Windows, Python 3.10.6) | RSS MiB |
| --- | ---: |
| Import runtime | 369,8 |
| Prediksi gambar dengan CLIP | 723,4 |
| CLIP + production retrieval + encode/query | 1105,5 |
| Smoke HTTP selesai | 1127,7 |

Private committed memory pada tahap terakhir: 1847,2 MiB. Angka ini berbeda dari
RSS dan bukan ukuran cgroup Linux. Memori Streamlit, session, concurrency, dan
overhead kernel/container belum termasuk pengukuran backend ini.

- Sebelum penambahan launcher: **963 tes lulus**, 3 warning.
- Setelah seluruh perubahan packaging: **972 tes lulus**, 3 warning;
  compile source berhasil. YAML Compose lolos schema resmi Compose Specification,
  dan template environment diterima oleh `Settings.from_env()` setelah token
  sintetis digunakan untuk validasi.
- Launcher diuji untuk keluarnya salah satu proses (normal/gagal), SIGTERM,
  kegagalan spawn, dan kill/reap proses yang tidak berhenti.
- Smoke production asli dengan folder Chroma 48/cosine: `/health`, `/search`,
  `/chat` fallback, dan `/predict-image` semuanya HTTP 200. Ini tidak membuktikan
  akurasi budaya, ranking holdout, layanan OpenAI langsung, atau kapasitas VPS.
- `pip check`: tidak ada konflik dependency environment lokal.
- Source secret scan bawaan: tidak ada temuan dalam allowlist scanner; `.env`,
  artefak, dan histori Git tidak diperiksa. Ini bukan bukti tidak adanya rahasia
  di seluruh filesystem.
- Bandit `-ll`: 2 Medium B615 pada loader CLIP, 0 High. Kedua pemanggilan memakai
  `local_files_only=True`; keaslian artefak tetap perlu diverifikasi.
- Build, healthcheck Docker, Nginx, TLS, dan batas RAM saat runtime belum dapat
  diuji pada host ini karena Docker tidak tersedia.

## Dependency yang dilaporkan audit online

Tanggal pemeriksaan: 21 Oktober 2026. Duplikat ID di raw report dihitung satu
per pasangan paket/advisory pada kolom berikut. Versi perbaikan adalah data
advisory, bukan upgrade yang sudah diterapkan atau diuji.

| Paket lokal | Advisory unik | Perbaikan yang dilaporkan |
| --- | ---: | --- |
| aiohttp 3.14.3 | 8 | 3.14.4 |
| certifi 2026.7.22 | 1 | 2026.10.19 |
| chromadb 1.5.9 | 4 | Tidak ada versi perbaikan pada hasil audit |
| oauthlib 3.3.1 | 1 | 4.0.0 |
| pillow 12.3.0 | 8 | 13.0.0 |
| sentence-transformers 3.4.1 | 1 | 5.6.0 |
| transformers 4.57.6 | 6 | Berbeda menurut advisory; sebagian tidak menyebut versi perbaikan |

Pillow `>=12.3,<13`, sentence-transformers `<4`, dan transformers `<5` pada
requirements saat ini menghalangi beberapa upgrade yang disebut advisory.
Perubahan major perlu tes kompatibilitas dengan artefak asli. Instalasi image
baru mungkin mengambil patch berbeda untuk dependency lain; tetap audit hasilnya.

Advisory baru sentence-transformers
[GHSA-jhr6-gm9c-rqjv](https://github.com/advisories/GHSA-jhr6-gm9c-rqjv) membahas
bypass `trust_remote_code=False` pada direktori model lokal. Cache mount read-only
dan sumber tepercaya mengurangi kesempatan perubahan setelah deploy; keduanya
tidak memperbaiki paket yang rentan atau membuat cache berbahaya menjadi aman.
Ada perbedaan yang perlu diperhatikan antara data patched-version advisory dan
[penjelasan maintainer PR #3807](https://github.com/huggingface/sentence-transformers/pull/3807):
PR tersebut menyebut warning pada 5.6 dan penegakan trust pada major 6. Jangan
menganggap instalasi 5.6 saja telah membuktikan batas kepercayaan aman.
Pillow mencakup masalah decoder TIFF, misalnya
[advisory resmi Pillow GHSA-q45j-58j2-45rg](https://github.com/python-pillow/Pillow/security/advisories/GHSA-q45j-58j2-45rg), sehingga
jalur preview UI memerlukan tinjauan tersendiri. Ini analisis keterjangkauan
berdasarkan source dan advisory, bukan hasil eksploitasi.

Raw laporan lokal berada di direktori yang diabaikan Git:
`security-scan-results/deployment-pip-audit.json`,
`deployment-bandit.json`, dan `deployment-memory.json`.
Kebijakan VEX yang ada tidak diubah.

## Packaging yang ditambahkan

- Satu service Compose/satu container; FastAPI mendengar loopback port 8000,
  Streamlit port 8501. Batas `mem_limit: 1g` berlaku terhadap keduanya.
- `memswap_limit: 1g` sama dengan batas RAM: tidak ada swap tambahan.
- PyTorch CPU dipasang dari index resmi, tanpa memasang CUDA sebagai kebutuhan VPS.
- Satu worker API, thread native dibatasi, concurrency deployment satu request.
- Launcher meneruskan shutdown dan menghentikan sibling saat satu service keluar.
- Healthcheck memeriksa API dan UI, bukan sekadar keberadaan proses.
- User non-root, capability dilepas, no-new-privileges, log berotasi.
- Artefak/cache/gambar di `/data` dari bind mount read-only. Snapshot Chroma di
  temporary filesystem container yang writable; tidak dijadikan tmpfs RAM.
- Build context dibatasi allowlist `.dockerignore`; `.env`, cache, model,
  database, virtualenv, dan konfigurasi editor tidak dikirim ke build context.
- Port host 8501 diikat ke loopback untuk proxy HTTPS/SSH tunnel. Backend tidak
  dipublikasikan. Contoh Nginx menyediakan forwarding WebSocket untuk Streamlit.

Panduan langkah Ubuntu: [vps-deployment.md](vps-deployment.md).
