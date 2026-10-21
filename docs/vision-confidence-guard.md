# Decision guard prediksi gambar

`POST /predict-image` menerima hasil hanya jika **confidence >= min_confidence
AND margin >= min_margin**. Default `WAYANG_MIN_CONFIDENCE=0.85` dan
`WAYANG_MIN_MARGIN=0.20`. Keduanya wajib finite pada interval inklusif [0, 1],
dengan validasi identik di development dan production, termasuk konfigurasi
langsung melalui `Settings`. NaN, infinity, angka negatif, >1, dan teks invalid
ditolak saat startup. Threshold 0 menonaktifkan pembatas terkait; gunakan default
sebagai konfigurasi awal deployment, bukan jaminan kualitas di semua domain.

Formula tetap `softmax(logits / temperature)`, top-3 diurutkan seperti sebelumnya,
confidence = probabilitas top-1, margin = top-1 minus top-2. Guard tidak mengubah
probabilitas, ranking, temperature, atau membatasi confidence maksimum.
Perbandingan memakai float hasil inferensi **sebelum pembulatan**, tanpa epsilon
atau toleransi. Nilai yang tepat sama threshold diterima; float tepat di bawahnya
ditolak. Serialisasi tetap enam desimal: angka yang ditampilkan dapat terlihat
sama threshold meskipun float internal sedikit lebih rendah. Client harus memakai
`status`/`decision`, bukan menghitung ulang keputusan dari angka respons.

## Kontrak HTTP 200

Schema OpenAPI menggunakan union dengan discriminator `status`. Untuk accepted,
`prediksi`, `top_3`, dan `profil` mempertahankan struktur lama; field keputusan,
confidence/margin tingkat atas, `top_3_status`, dan `message` ditambahkan.
`top_3_status: "candidates"` berlaku untuk seluruh top-3: kandidat alternatif
bukan tiga identitas final. `decision.reasons` berisi `low_confidence`,
`low_margin`, keduanya dalam urutan tersebut, atau kosong jika accepted.

Contoh ilustratif accepted (profil dipersingkat):

```json
{
  "status": "accepted",
  "decision": {"accepted": true, "reasons": []},
  "confidence": 0.91,
  "margin": 0.86,
  "prediksi": {"id": 19, "label": "gunungan", "nama": "Gunungan", "confidence": 0.91, "margin": 0.86},
  "top_3_status": "candidates",
  "top_3": [
    {"id": 19, "label": "gunungan", "nama": "Gunungan", "probability": 0.91},
    {"id": 0, "label": "jayangrana", "nama": "Jayangrana", "probability": 0.05},
    {"id": 1, "label": "umar_maya", "nama": "Umar Maya", "probability": 0.02}
  ],
  "profil": {"deskripsi": "Profil tokoh dari katalog"},
  "message": null
}
```

Contoh ilustratif uncertain:

```json
{
  "status": "uncertain",
  "decision": {"accepted": false, "reasons": ["low_confidence", "low_margin"]},
  "confidence": 0.45,
  "margin": 0.10,
  "prediksi": null,
  "profil": null,
  "top_3_status": "candidates",
  "top_3": [
    {"id": 19, "label": "gunungan", "nama": "Gunungan", "probability": 0.45},
    {"id": 0, "label": "jayangrana", "nama": "Jayangrana", "probability": 0.35},
    {"id": 1, "label": "umar_maya", "nama": "Umar Maya", "probability": 0.10}
  ],
  "message": "Gambar belum dapat dikenali dengan yakin. Gunakan foto yang lebih jelas, menampilkan seluruh tubuh wayang, dan berlatar sederhana."
}
```

Perubahan kontrak yang diperlukan: `prediksi` dan `profil` sekarang nullable
untuk uncertain. Client lama wajib memeriksa `status` sebelum mengakses kedua
field tersebut. HTTP 200 hanya berarti inferensi selesai, bukan identitas pasti.
Tampilkan pesan panduan dan label “kandidat belum pasti” untuk semua top-3 pada
uncertain; jangan menyajikan profil tokoh sebagai fakta final. Accepted juga
berarti lolos kebijakan threshold, bukan bukti bahwa model selalu benar.

## Production dan artefak deployment

Tetapkan `WAYANG_MIN_CONFIDENCE=0.85`, `WAYANG_MIN_MARGIN=0.20`,
`WAYANG_ENV=production`, dan `WAYANG_MODEL_DIR` menuju direktori artefak V2
melalui konfigurasi environment deployment. Tetap ikuti pengaturan service token,
allowed hosts, dan validasi artefak pada [security hardening](security-hardening.md).
Restart service setelah perubahan konfigurasi. Jangan memasukkan credential,
file `.env`, atau bobot ke Git. `.env.example` hanya contoh aman.

Model aktif `model_clip_b32_bab4_early_stopping` menggunakan frozen CLIP
ViT-B/32 dan PyTorch Linear(512, 20), dengan best epoch 120. Artefak ini belum
memiliki temperature terkalibrasi (`temperature: null`), sehingga backend memakai
1.0. Loader menyelaraskan urutan kelas artifact ke ID tokoh resmi dan menerapkan
padding putih persegi sesuai metadata preprocessing model. Angka validasi pada
konfigurasi artifact adalah sebelum evaluasi guard, bukan pengukuran rejection/OOD
atau jaminan performa subset accepted.

## Residual risk dan observabilitas

Softmax dapat tetap tinggi pada gambar asing (OOD), bahkan dengan margin besar.
Guard ini bukan detektor OOD sempurna dan tidak membuktikan sistem aman untuk
gambar asing. Deteksi OOD berbasis embedding atau data negatif masih diperlukan,
beserta evaluasi coverage, false acceptance, dan rejection pada data representatif.

Event keputusan hanya memuat accepted/uncertain, kategori alasan, latency, dan
request ID yang dibuat middleware. Tidak ada gambar, nama upload, profil,
probabilitas, logits mentah, path internal, konfigurasi rahasia, atau exception
dalam event tersebut. Middleware keamanan yang ada tetap aktif. Tidak ada endpoint
debug baru; error inferensi tetap generik.
