# Menjalankan Wayang Sasak di VPS Ubuntu

Baca [audit deployment](deployment-audit.md) terlebih dahulu. Konfigurasi ini
menyatukan API dan Streamlit dalam satu container dengan batas total **1 GiB**.
Backend asli yang diukur di Windows sudah melewati batas itu sebelum UI; image
ini belum dinyatakan siap production pada 1 GiB. Build/instalasi dependency juga
memerlukan RAM dan disk di host; limit Compose hanya membatasi container runtime.

## 1. Siapkan Docker Engine di Ubuntu

Build dapat dijalankan langsung di VPS. Docker Desktop Windows tidak diperlukan
untuk cara ini. Periksa versi Ubuntu dan arsitektur dengan `cat /etc/os-release`
dan `uname -m`; ikuti prasyarat [instalasi resmi Docker Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
Jika Docker sudah ada, periksa versinya sebelum mengubah instalasi yang digunakan
service lain. Untuk instalasi baru dari repository resmi:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "$VERSION_CODENAME")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker version
sudo docker compose version
```

## 2. Transfer source dan artefak

Clone repository/transfer release yang sudah menyertakan file Docker terbaru.
Konfigurasi baru pada workspace belum otomatis tersedia di GitHub sebelum
di-commit dan push. Jangan ikut mengirim `.venv`, `.env` lokal, atau log.
Artefak diabaikan Git dan perlu ditransfer terpisah dari sumber tepercaya:

```text
Chatbot_Wayang_Sasak/
  Dockerfile
  compose.yaml
  .dockerignore
  ...source aplikasi...
  artifacts/
    model_clip_b32_bab4_early_stopping/
      vision_encoder/config.json
      vision_encoder/model.safetensors
      image_processor/preprocessor_config.json
      classifier_head.pt
      label_mapping.json
      model_config.json
    wayang_db.json
    chroma_db_wayang/
      chroma.sqlite3
      ...file index lainnya...
    huggingface/
      hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
        ...blobs, refs, snapshots dan konfigurasi/tokenizer yang lengkap...
    dataset_asli/
      ...gambar referensi yang dirujuk JSON...
```

Salin seluruh direktori Chroma dan cache Hub, bukan hanya SQLite/bobot model.
Pertahankan symlink relatif saat transfer cache jika ada. Pastikan UID 10001 di
container dapat membaca file dan menelusuri direktori; jangan memakai chmod 777.
Verifikasi checksum dengan sumber terpercaya setelah transfer.

## 3. Konfigurasi khusus container

Jalankan dari root project di VPS:

```bash
cp deploy/vps.env.example .env.docker
chmod 600 .env.docker
python3 - <<'PY'
from pathlib import Path
import secrets
path = Path('.env.docker')
text = path.read_text()
assert 'WAYANG_SERVICE_TOKEN=\n' in text
path.write_text(text.replace('WAYANG_SERVICE_TOKEN=\n',
    'WAYANG_SERVICE_TOKEN=' + secrets.token_urlsafe(48) + '\n'))
PY
```

Edit `.env.docker` dengan editor pada VPS. File ini diabaikan Git dan tidak masuk
image. Gunakan token yang dihasilkan; jangan meneruskan token backend ke browser.
Isi `LLM_API_KEY` pada server jika membutuhkan OpenAI/web search. Key kosong
mempertahankan fallback lokal yang sudah ada. Pertahankan mode production,
validasi artefak, serta threshold model/retrieval yang ada.

**Perbaiki pilihan folder Chroma sebelum startup.** Audit lokal menemukan folder
`artifacts/chroma_db_wayang` berisi 47 dokumen dan akan ditolak production.
Folder `artifacts/chroma_db_wayang_updated_20261005_191158` berisi 48/cosine dan
lolos smoke asli. Jika folder tervalidasi itu yang ditransfer dengan nama yang
sama, set di `.env.docker`:

```dotenv
WAYANG_CHROMA_DIR=/data/chroma_db_wayang_updated_20261005_191158
```

Pastikan JSON, metadata, cache embedding, dan revision sesuai baseline yang
dimaksud. Snapshot cache yang terlihat saat audit memiliki commit
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; keberadaannya tidak membuktikan bahwa
commit tersebut yang dipakai saat membuat semua embedding Chroma. Pin
`WAYANG_EMBEDDING_REVISION` setelah asal-usulnya diverifikasi. Production berjalan
offline untuk loader embedding; cache hilang harus diperbaiki sebelum startup.

## 4. Build dan validasi terbatas sebelum publikasi

```bash
sudo docker compose config --quiet
sudo docker compose build
sudo docker compose up -d
sudo docker compose ps
sudo docker compose logs --tail=100 app
sudo docker stats --no-stream
sudo docker inspect "$(sudo docker compose ps -q app)" \
  --format 'memory={{.HostConfig.Memory}} swap-total={{.HostConfig.MemorySwap}}'
```

Kedua nilai limit harus `1073741824` byte. `config --quiet` memvalidasi konfigurasi
tanpa mencetak hasil environment yang berisi key/token. Jangan membagikan output
`docker inspect` lengkap atau resolved Compose config yang memuat environment.

```bash
sudo docker compose exec -T app python -c "from urllib.request import urlopen; print(urlopen('http://127.0.0.1:8000/health', timeout=3).status); print(urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status)"
```

Uji katalog, chat lokal, pertanyaan lanjutan, unggahan accepted/uncertain, dan
gambar referensi dari UI. Jika OpenAI diaktifkan, uji key, biaya, timeout, dan
fallback secara terkontrol. Amati peak RAM saat startup dan request, termasuk
beberapa sesi/gambar besar; satu respons sehat bukan uji kapasitas.

Jika container keluar, periksa status OOM dan log:

```bash
sudo docker inspect "$(sudo docker compose ps -a -q app)" \
  --format 'oom={{.State.OOMKilled}} exit={{.State.ExitCode}}'
sudo docker compose logs --tail=100 app
```

`exit 137` atau `OOMKilled=true` memerlukan tinjauan kapasitas; jangan menganggap
semua kegagalan startup adalah OOM. Chroma 47, cache hilang, token kosong, dan
permission artefak juga dapat menghentikan startup. Hentikan percobaan yang
berulang dengan `sudo docker compose stop`. Memori host/swap tidak menghapus
batas container 1 GiB tanpa swap. Optimasi/quantization atau model yang lebih
kecil perlu validasi hasil inferensi, ranking, dan threshold tersendiri.

## 5. Akses dan HTTPS

UI hanya dipublikasikan di loopback VPS (`127.0.0.1:8501`). Untuk pemeriksaan
privat, buat SSH tunnel dari komputer lokal (ganti user dan host):

```bash
ssh -L 8501:127.0.0.1:8501 USER@HOST
```

Buka `http://localhost:8501`. Setelah kapasitas dan temuan keamanan ditangani,
arahkan DNS domain ke VPS dan pasang Nginx pada host. Contoh konfigurasi tersedia
di `deploy/nginx.conf`; ganti `wayang.example.com` dengan domain asli. File ini
baru aktif setelah disalin dan konfigurasi Nginx divalidasi.

```bash
sudo apt-get install -y nginx
sudo cp deploy/nginx.conf /etc/nginx/sites-available/wayang-sasak
sudo ln -s /etc/nginx/sites-available/wayang-sasak /etc/nginx/sites-enabled/wayang-sasak
sudo nginx -t
sudo systemctl reload nginx
```

Pasang sertifikat melalui [instruksi resmi Certbot untuk Nginx](https://certbot.eff.org/instructions).
Aktifkan HTTPS dan redirect HTTP sesuai domain. Set di `.env.docker`:

```dotenv
STREAMLIT_BROWSER_SERVER_ADDRESS=domain-anda.example
STREAMLIT_BROWSER_SERVER_PORT=443
```

Lalu `sudo docker compose up -d` untuk menerapkan perubahan environment. Tetap
aktifkan proteksi CORS/XSRF Streamlit. Proxy harus mempertahankan upgrade WebSocket,
seperti contoh yang disertakan. Izinkan port SSH yang digunakan serta 80/443 pada
firewall/provider. Port 8000 tetap internal; port 8501 tetap loopback. Perhatikan
[interaksi port Docker dan firewall Ubuntu](https://docs.docker.com/engine/install/ubuntu/#firewall-limitations).

## 6. Syarat go-live

Jika deployment sudah dikelola GitHub Actions, gunakan
`docker compose --env-file .env.image` untuk perintah manual agar image release
tetap sesuai digest yang tercatat. Setup Secrets dan aktivasi pipeline dijelaskan
di [panduan CI/CD](cicd.md).

- Artefak terpercaya dan loader production berhasil; baseline 48/cosine terjaga.
- Image Linux dibangun, diuji, diaudit dependency-nya, lalu dikunci untuk rilis.
- Temuan advisory baru dan jalur decoder UI ditinjau/remediasi.
- Beban representatif terbukti stabil dalam limit total 1 GiB dengan ruang aman;
  jika tidak, ubah rencana kapasitas/optimasi sebelum mempublikasikan.
- Healthcheck, shutdown/restart, TLS/WebSocket, dan proteksi port berhasil diuji.
- Siapkan backup artefak/config dan release image sebelumnya untuk rollback.

IP/host, user SSH, versi Ubuntu, arsitektur, RAM host, dan domain belum diberikan
saat panduan ini ditulis. Belum ada perubahan pada VPS atau deployment publik.
