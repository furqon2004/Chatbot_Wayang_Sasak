> Catatan migrasi OpenAI: uraian provider/konfigurasi lama di bawah merupakan
> catatan historis. Implementasi aktif memakai OpenAI untuk narasi dan web;
> lihat [konfigurasi chat](chat.md). Hasil audit lama bukan validasi migrasi terbaru.

# Audit routing pengantar Wayang Sasak

Baseline: `79c21b6`. Audit 10 Oktober 2026, tanpa API atau perubahan artefak.

## Evidence lokal

- `artifacts/wayang_db.json` memuat 20 profil tokoh: deskripsi, filosofi,
  ciri-ciri, kata kunci, dan gambar. Tidak ditemukan definisi umum Wayang
  Sasak atau penjelasan hubungan tradisi ini dengan cerita Menak.
- Konfigurasi lokal menunjuk ke
  `artifacts/chroma_db_wayang_updated_20261005_191158`: 48 dokumen,
  terdiri dari 20 profil, 20 ciri-ciri, dan 8 filosofi; seluruhnya dipetakan
  ke 20 identitas tokoh. Audit memakai SQLite `mode=ro`, tanpa client Chroma.
  Folder Chroma lama dan backup masing-masing berisi 47 dokumen.
- Penyebutan Menak yang ditemukan adalah julukan **Menak Senduk** pada
  Kembung. Ini bukan evidence hubungan Wayang Sasak dengan cerita Menak.
- Retrieval menggunakan Chroma untuk skor identitas tokoh, lalu mengambil
  kutipan publik dari JSON; dokumen mentah Chroma bukan sumber jawaban.
  Kontrak sumber dan schema pemilihan Gemini saat ini hanya mendukung
  `profile:<id>:<field>:<index>` dari `wayang_db.json`.

## Hasil tiga pertanyaan

| Pertanyaan | Kecukupan lokal | Hasil setelah perbaikan |
| --- | --- | --- |
| Apa itu Wayang Sasak? | Tidak cukup | Fallback lokal, tanpa Search |
| Ceritakan tentang Wayang Sasak. | Tidak cukup | Fallback lokal, tanpa Search |
| Apa hubungan Wayang Sasak dengan cerita Menak? | Tidak cukup; Menak Senduk tidak relevan | Fallback lokal, tanpa Search |

Audit batas skor memakai profil asli dan semantic **mock maksimum 1,0**,
bukan hasil embedding aktual. Dua pertanyaan pertama memiliki lexical 0
untuk semua profil, sehingga skor hybrid maksimum 0,55 masih di bawah
threshold tetap 0,5590755322575569. Pertanyaan Menak dapat memberi skor
tinggi pada profil akibat kata yang tumpang tindih (batas atas Hermus 0,745;
Kembung dan Gunungan 0,6775), tetapi skor bukan bukti kecukupan isi.
Tidak diperlukan API atau embedding aktual untuk membuktikan kekurangan ini.

## Perubahan

- Intent pengantar terbatas dan pertanyaan hubungan Menak mencoba retrieval
  lokal dengan pertanyaan asli. Rewrite pengantar menjadi pertanyaan tokoh
  utama dihapus. Permintaan internet eksplisit, sejarah lebih luas, dan
  alur cerita tetap memakai routing web sebelumnya.
- Hanya kalimat lengkap yang langsung mendefinisikan Wayang Sasak atau
  menyatakan penggunaan cerita Menak, dari deskripsi tervalidasi dan hit
  di atas threshold, boleh masuk context pengantar. Filter sengaja konservatif;
  variasi bahasa lain bisa ditolak dan menghasilkan fallback.
  Nama/julukan tokoh, kata kunci, filosofi, dan ciri visual tidak cukup.
- Gemini tetap hanya memilih ID evidence; backend merender kutipan literal
  beserta sumber JSON sebenarnya, tanpa teks fakta bebas. Jawaban pengantar
  tidak diberi identitas tokoh. Cache RAG yang sama dipakai untuk jawaban valid.
- Evidence kurang/kosong atau retrieval tidak tersedia menghasilkan fallback
  yang menjelaskan kekurangan, tanpa sumber semu atau otomatis memakai Search.
- Model, threshold, dataset, JSON, ChromaDB, guard uncertain, serta tombol
  profil/penjelasan tidak diubah.

## Materi yang perlu disediakan

1. Materi definisi Wayang Sasak, konteks geografis dan bentuk pertunjukannya.
2. Materi yang secara eksplisit menjelaskan hubungan lakon Wayang Sasak dengan
   cerita/Serat Menak; bedakan tradisi, cerita, dan nama atau julukan tokoh.
3. Untuk jawaban sejarah: materi asal-usul/perkembangan beserta batas kepastian
   dan variasi tradisi yang memang didukung sumber.

Sertakan penulis/narasumber, judul, penerbit/lembaga, tahun, halaman/bagian
atau timestamp, kutipan relevan, dan URL bila tersedia. Sumber dapat berupa
buku/kajian, publikasi lembaga kebudayaan, atau transkrip wawancara dalang
yang memiliki identitas dan konteks jelas. Belum ada rujukan bibliografis
umum dalam evidence yang diaudit; jangan membuat atribusi baru.

Penambahan dokumen umum nantinya memerlukan desain ingestion, identitas
evidence, sumber publik, dan evaluasi retrieval khusus dokumen umum. Jangan
menyisipkan sejarah ke profil tokoh hanya untuk melewati filter. Tahap ini
tidak melakukan migrasi itu. Tes evidence cukup menggunakan fixture sintetis
dan mock hasil retrieval; hasilnya tidak menyatakan KB produksi sudah cukup.

## Validasi

`tests/test_general_rag.py` menguji nol Search, kutipan/sumber yang tepat,
cache hit/expiry, perubahan evidence, output Gemini tidak valid, skor di
bawah threshold, retrieval tidak tersedia, serta alur tokoh/katalog.
Tes memakai mock; fixture global suite memblokir jaringan dan mengisolasi
environment dari credential lokal. Tes web diperbarui agar pertanyaan yang
sekarang menjadi intent pengantar tidak lagi diharapkan memanggil Search.

Hasil akhir: `python -m pytest -q` **603 passed**, 3 warnings (dua deprecation
dependensi dan satu peringatan PIL dari tes batas ukuran gambar).
`git diff --check` bersih. Tidak ada API nyata, commit, atau push.
