# CI/CD GitHub Actions untuk repository publik dan VPS Ubuntu

Workflow utama: `.github/workflows/ci-cd.yml`. Pemeriksaan bersama:
`.github/workflows/security.yml`. Script deployment: `scripts/deploy_vps.sh`.

## Alur

| Pemicu | Pemeriksaan | Image | VPS |
| --- | --- | --- | --- |
| Pull request ke `main`, termasuk fork | Tes, compile, scan source, pip check, audit dependency/VEX | Build dan pip check; tidak publish | Tidak deploy |
| Push ke `main` | Pemeriksaan yang sama | Build/publish `ghcr.io/OWNER/REPO:COMMIT_SHA` | Deploy jika variable pengaktif bernilai `true` |
| Run workflow di branch `main` | Pemeriksaan yang sama | Build/publish image commit | Deploy jika input `deploy` dicentang dan variable pengaktif `true` |
| Jadwal security mingguan | Workflow security | Tidak publish | Tidak deploy |

Deployment mengambil image berdasarkan **digest**, menunggu healthcheck API dan
UI, lalu mencatat image yang berhasil di `.env.image` pada VPS. Jika healthcheck
gagal, script memulihkan Compose dan mencoba menjalankan image sebelumnya.
Deployment tetap berstatus gagal saat rollback berhasil. Pada deployment pertama
belum ada release sebelumnya untuk dipulihkan. Pergantian satu container dapat
menimbulkan downtime singkat dan mereset session UI.

**Penghambat yang sudah diketahui:** audit terakhir menemukan advisory baru di
luar kebijakan VEX yang ada, serta backend yang memakai RAM lebih dari 1 GiB di
Windows. CI sengaja mempertahankan kegagalan audit dependency; publish/deploy
tidak berjalan jika pemeriksaan gagal. Tidak ada flag untuk melewati pemeriksaan
tersebut. Penambahan workflow ini belum memperbaiki paket rentan atau membuktikan
bahwa image lengkap muat dalam 1 GiB. Lihat [laporan audit](deployment-audit.md).

## Pengaturan GitHub

Aktifkan Actions pada repository. Buat environment **`production`** melalui
Settings → Environments, batasi deployment ke `main`, dan atur required reviewer
jika diperlukan. Workflow PR memakai runner GitHub, `pull_request`, dan token
read-only; kredensial VPS hanya dipakai job deployment pada `main`.

Tambahkan berikut sebagai **environment secrets** pada `production`:

| Secret | Isi |
| --- | --- |
| `VPS_HOST` | IP IPv4 atau hostname VPS; tanpa `https://` |
| `VPS_USER` | User SSH untuk deployment |
| `VPS_SSH_PRIVATE_KEY` | Private key khusus deployment, format OpenSSH, tanpa passphrase interaktif |
| `VPS_KNOWN_HOSTS` | Entri known_hosts untuk host/port VPS dengan fingerprint yang sudah diverifikasi |

Known-hosts harus sesuai host dan port yang digunakan. Untuk port nonstandar,
format host biasanya `[HOST]:PORT`. Bandingkan fingerprint dengan informasi
server melalui console/provider yang dipercaya sebelum menyimpannya. Workflow
memakai `StrictHostKeyChecking=yes` dan tidak mempercayai hasil scan jaringan
tanpa verifikasi.

Tambahkan **repository variable** berikut melalui Settings → Secrets and
variables → Actions → Variables:

| Variable | Nilai |
| --- | --- |
| `ENABLE_VPS_DEPLOY` | `true` setelah VPS, image registry, kapasitas, dan keamanan siap. Tidak diisi berarti deployment nonaktif. |

Variable berikut dapat ditaruh pada environment `production`:

| Variable | Default | Catatan |
| --- | --- | --- |
| `VPS_PORT` | `22` | Port SSH, rentang 1–65535 |
| `VPS_DEPLOY_PATH` | `/opt/wayang-sasak` | Direktori absolut yang sudah ada, tanpa spasi atau komponen `..` |

`ENABLE_VPS_DEPLOY` harus berupa repository variable karena kondisi job dievaluasi
sebelum environment deployment dimasuki. Gunakan
[pengaturan secrets resmi GitHub](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets)
untuk memasukkan key; jangan menulis nilainya dalam YAML atau repository.

Publish GHCR memakai `GITHUB_TOKEN` otomatis dengan `packages: write` hanya pada
job publish, sehingga tidak perlu membuat secret registry baru; lihat
[autentikasi Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
Action resmi
checkout/setup-python dipin ke commit SHA penuh, mengikuti
[panduan keamanan workflow GitHub](https://docs.github.com/en/actions/reference/security/secure-use).
Job test dan build PR tidak diberi kredensial SSH. Jangan menggantinya dengan
`pull_request_target` untuk mengeksekusi source dari fork.

## Siapkan server satu kali

Ikuti [panduan Ubuntu](vps-deployment.md) untuk Docker Engine/Compose plugin,
artefak asli, environment production, izin baca UID 10001, Nginx, dan HTTPS.
User SSH harus bisa menjalankan Docker secara noninteraktif dan menulis ke
direktori aplikasi. Akses Docker memberikan hak besar atas host; gunakan key
deployment khusus dan lindungi environment/branch production.
Workflow membangun image Linux amd64 pada runner Ubuntu GitHub. Periksa
`uname -m` pada VPS: konfigurasi ini ditujukan untuk `x86_64`; VPS ARM64 memerlukan
penyesuaian build platform sebelum deployment diaktifkan.

Direktori tujuan minimal sebelum job deploy dijalankan:

```text
/opt/wayang-sasak/
  .env.docker          # konfigurasi production dan API key, hanya di VPS
  artifacts/           # model, Chroma 48/cosine, cache embedding, JSON, gambar
```

Workflow mengirim `compose.yaml` sebagai `compose.pending.yaml`, kemudian script
deployment dijalankan melalui SSH. Source Python, model, `.env` lokal, dan API key
tidak ditransfer lewat job deployment. Source aplikasi sudah ada dalam image;
artefak/cache dibaca dari server. Batas total RAM tetap `1g` tanpa swap tambahan.

Default awal `WAYANG_CHROMA_DIR` pada template menunjuk folder kanonis; sesuaikan
ke artefak 48 dokumen yang tervalidasi sesuai panduan. Jangan memakai database 47
dokumen atau menonaktifkan production hanya supaya healthcheck lolos.

## Aktivasi pertama

1. Siapkan server dan environment secrets; biarkan `ENABLE_VPS_DEPLOY` belum diisi.
2. Push konfigurasi CI/CD ke GitHub. Selesaikan kegagalan tes/audit sampai job
   publish berhasil. Tes CI menggunakan mock model; kapasitas artefak asli tetap
   perlu diuji pada VPS.
3. Di halaman package GHCR, ubah visibilitas image menjadi **public** agar VPS
   bisa pull tanpa credential registry. Repository publik tidak otomatis membuat
   package container publik. Jika image sengaja private, login registry secara
   terpisah pada VPS menggunakan credential read-only yang sesuai.
4. Setelah uji kapasitas dan keamanan selesai, set repository variable
   `ENABLE_VPS_DEPLOY=true`.
5. Buka Actions → **CI and VPS deployment** → Run workflow → pilih `main` dan
   centang `deploy`. Setujui environment jika required reviewer diaktifkan.
6. Push selanjutnya ke `main` menjalankan pipeline dan deployment yang sama.

Jadikan hasil checks dan build image PR sebagai required checks pada branch
protection/ruleset `main`, menggunakan nama check yang tampil setelah run pertama.
Jangan menjalankan PR publik pada self-hosted runner VPS production.

## Operasi server setelah dikelola CI/CD

Script menulis `.env.image` hanya setelah image sehat. Untuk perintah manual,
gunakan file ini supaya Compose memilih release yang sama:

```bash
cd /opt/wayang-sasak
docker compose --env-file .env.image ps
docker compose --env-file .env.image logs --tail=100 app
docker compose --env-file .env.image up -d --no-build --pull never
docker stats --no-stream
```

`.compose.previous.yaml` menyimpan Compose sebelum update terakhir. Pull yang
gagal tidak mengganti konfigurasi yang berjalan. Rollback juga dapat gagal jika
artefak, konfigurasi, permission, atau kapasitas host bermasalah; periksa status
container dan log. Script tidak menghapus image lama, model, database, atau
`.env.docker`. Atur retensi image setelah ada kebijakan rollback yang jelas.

Konfigurasi workflow dan script dapat divalidasi lokal; koneksi SSH, publish GHCR,
healthcheck container, dan deployment nyata tetap memerlukan setup GitHub/VPS.
