# Desain: Peningkatan UX, konteks, dan ketahanan Chatbot Wayang Sasak

Tanggal: 2026-10-12

## Tujuan

Meningkatkan pengalaman awal, pemilihan konteks, keandalan pengiriman, dan
perlindungan beban Chatbot Wayang Sasak tanpa mengganti arsitektur FastAPI dan
Streamlit, provider OpenAI, retrieval, knowledge base, atau alur analisis gambar.

## Kondisi repository saat ini

- `streamlit_app.py` sudah menampilkan welcome screen saat history kosong,
  memiliki sidebar dengan tombol percakapan baru dan katalog, serta satu
  composer untuk teks dan gambar.
- Pilihan katalog muncul sebagai tombol dalam respons chat. Popover lampiran
  saat ini hanya menerima gambar.
- Pesan teks dikirim melalui jalur `ask_question`; gambar mempunyai jalur
  terpisah. Input tetap berada di session state dan history tab yang berumur
  lima menit.
- `/chat` menggunakan konteks terstruktur berisi subject/evidence IDs. Backend
  sudah memvalidasi batas dan ID, serta meresolusi nama eksplisit dari pesan
  terbaru. Riwayat pesan lengkap tidak dikirim ke model.
- Retrieval menggabungkan pencarian semantik dan leksikal. Provider aktif
  adalah OpenAI; jawaban lokal dan web sudah memiliki jalur grounding berbeda.
- Middleware membuat request ID, membatasi body, memakai fixed-window limiter
  per alamat peer dan endpoint, serta menolak request saat kapasitas penuh.
  Limiter dan kapasitas masih in-memory per proses. Belum ada FIFO wait queue,
  `Retry-After`, atau deduplikasi idempotensi.
- Deployment dokumentasi memakai satu worker dan menyatakan bahwa koordinasi
  in-memory tidak cukup untuk beberapa worker/replica.
- Schema dan respons saat ini tidak mendefinisikan lock katalog. Tes backend
  menetapkan bahwa nama tokoh eksplisit di pesan berikutnya boleh mengganti
  subjek konteks aktif.

## Pilihan desain

1. **Migrasi UI dan backend besar-besaran.** Mengganti komponen chat dan
   memindahkan request ke sistem job asinkron. Ini memberi posisi antrean dan
   status tahap demi tahap, tetapi mengubah kontrak API, state percakapan, dan
   jalur deployment.
2. **Perubahan bertahap pada alur yang ada (direkomendasikan).** Memperbaiki
   Streamlit dan middleware yang sekarang, menambahkan state kontrak sekecil
   mungkin, antrean tunggu berbatas untuk request sinkron, serta adapter shared
   store opsional untuk deployment multi-worker. Provider dan retrieval tetap.
3. **Hanya perubahan lokal satu proses.** Memakai antrean dan idempotensi
   in-memory. Ini paling kecil, tetapi tidak memenuhi kebutuhan koordinasi
   lintas worker/replica.

Pilihan 2 dipilih. Tanpa URL shared store, aplikasi berjalan dalam mode satu
worker seperti deployment saat ini. Untuk beberapa worker/replica, deployment
harus mengonfigurasi shared store; mode tersebut tidak boleh diam-diam kembali
ke limiter atau idempotensi lokal bila shared store gagal.

## Perilaku produk

### Welcome dan contoh pertanyaan

- Saat percakapan kosong, tampilkan greeting khusus Wayang Sasak dan contoh
  pertanyaan yang sesuai dengan katalog atau kemampuan yang ada.
- Klik contoh hanya mengisi aksi/input tertunda. Greeting dan contoh tidak
  dimasukkan ke history atau dikirim ke `/chat` sampai pengguna mengirimnya.
- Rerun Streamlit tidak menambahkan greeting ke history.
- Sidebar menampilkan aksi **Chat Baru** sebagai tindakan utama dan tidak lagi
  memuat katalog permanen.

### Lampiran dan pemilihan katalog

- Popover lampiran berisi upload gambar dan pemilihan tokoh dari katalog.
- Pilihan tokoh dapat dilakukan tanpa unggah gambar. Nama dan thumbnail tampil
  bila tersedia; pilihan tampil di composer sebelum dikirim.
- Memilih gambar mempertahankan validasi ukuran, tipe, preview, dan endpoint
  `/predict-image` yang ada.
- Pilihan katalog diproses melalui jalur chat yang ada dengan identitas tokoh
  terstruktur, sehingga backend tetap menyusun jawaban dari evidence.

### Lock pilihan katalog dan konteks percakapan

- Katalog dapat dipilih sampai pilihan tokoh pertama berhasil dijawab backend.
- Setelah itu UI menampilkan pilihan sebagai container noninteraktif dengan
  label tokoh dan status terkunci berdasarkan state yang dikeluarkan backend.
- Backend memvalidasi bukti lock sebelum menerima perubahan pilihan melalui
  field pemilihan katalog. Untuk menjaga kontrak yang sudah diuji, lock ini
  tidak melarang pengguna menyebut nama tokoh lain dalam teks; backend tetap
  mengikuti nama eksplisit tersebut dan memperbarui konteks percakapan menurut
  alur yang ada.
- Konteks subject/evidence tetap dikirim dan divalidasi backend. Membuat chat
  baru menghapus state lock serta konteks lokal.

### Pengiriman dan idempotensi

- Semua pesan teks memakai satu jalur submit; unggah gambar tetap menggunakan
  endpoint pengenalan yang sama. Submit dari tombol atau Enter tidak boleh
  membuat dua entri history atau dua operasi backend.
- Setiap aksi kirim memperoleh idempotency key unik. Pengiriman dua teks
  identik yang disengaja tetap merupakan dua aksi dan mendapat key berbeda.
- Pengulangan request untuk aksi yang sama memakai key yang sama dan menerima
  hasil tersimpan, bukan menjalankan retrieval/provider/pengenalan dua kali.
- Request ID server tetap menjadi ID korelasi dan dikembalikan di respons serta
  header. Idempotency key terpisah dari request ID.
- Tidak menambahkan pengiriman ulang otomatis. Bila aplikasi menyediakan
  kontrol coba lagi, kontrol itu menggunakan key aksi yang gagal; kirim baru
  yang disengaja membuat key baru.

### Antrean, batas, dan respons sibuk

- Ganti penolakan kapasitas langsung dengan antrean tunggu FIFO berbatas
  sebelum concurrency slot. Antrean punya batas waktu tunggu dan batas jumlah
  waiter; penuh menghasilkan backpressure tanpa menyimpan kerja tak terbatas.
- Batas concurrency tetap mencakup endpoint sensitif yang ada. Request body
  tetap dibatasi sebelum parsing multipart/JSON.
- Rate limit meliputi peer/IP serta ID sesi anonim yang dibuat UI. ID sesi
  membantu membatasi spam per tab tetapi bukan autentikasi; batas peer tetap
  berlaku. Jangan mempercayai `X-Forwarded-For` tanpa konfigurasi proxy tepercaya.
- Gunakan `429` untuk pelampauan rate dan `503` untuk antrean/kapasitas yang
  penuh atau layanan shared store tidak tersedia. Keduanya mengirim
  `Retry-After` bila waktu tunggu bisa dihitung. Timeout antrean menggunakan
  respons sibuk yang sama dan tidak menjalankan handler.
- Dalam mode shared store, limiter, deduplikasi idempotensi, dan batas
  concurrency global dibagi antar worker. Kunci hasil idempotensi punya TTL dan
  batas ukuran. Store yang gagal tidak boleh memproses ulang key yang mungkin
  sudah berjalan.
- Tidak menjanjikan kuota identitas pengguna terverifikasi: aplikasi belum
  memiliki login. Penambahan autentikasi berada di luar desain ini.

### Status pemrosesan

- Selama request sinkron, tampilkan satu indikator yang sesuai: pemahaman dan
  pencarian informasi untuk teks, analisis gambar untuk upload.
- Tidak menampilkan persen atau urutan tahap retrieval/provider yang tidak
  dilaporkan backend. Jalur web diketahui dari respons akhir, bukan sebelum
  request selesai.
- UI menampilkan pesan sibuk dan waktu coba lagi berdasarkan status serta
  `Retry-After` yang benar-benar diterima.
- Karena endpoint tetap sinkron, posisi antrean tidak tersedia untuk
  ditampilkan secara langsung. Menampilkan “sedang dalam antrean” sebagai
  status terpisah membutuhkan endpoint job/streaming dan berada di luar pilihan
  bertahap ini.
- Indikator proses selalu berakhir pada sukses atau error.

## Batas keamanan dan kompatibilitas

- Pertahankan validasi schema Pydantic, batas upload, pemeriksaan konteks,
  pemfilteran evidence, guard prompt injection, provider OpenAI, cache, dan
  circuit breaker.
- Token lock katalog harus ditandatangani backend dan diverifikasi sebelum
  menerima perubahan pilihan katalog. Rahasia penandatanganan stabil harus
  dikonfigurasi pada deployment multi-worker; token tidak menjadi otorisasi
  provider atau sumber fakta.
- Pesan teks eksplisit tetap melewati klasifikasi domain dan grounding yang
  sama. Token lock tidak boleh mengubah retrieval atau memberi hak akses data.
- Respons dan error tidak mengekspos body provider, token, stack trace, atau
  data internal.
- History tetap sementara pada tab dan tidak dikirim sebagai transkrip lengkap
  ke provider. Tidak ada penyimpanan pertanyaan baru di disk.

## Area perubahan yang diharapkan

- `streamlit_app.py`: welcome, suggestion buttons, attachment/catalog popover,
  indikator pilihan dan proses, serta jalur submit tunggal.
- `ui_client.py` dan `ui_history.py`: ID aksi/sesi, header idempotensi,
  `Retry-After`, dan pemulihan state yang tetap ephemeral.
- `app/security.py`, `app/security_middleware.py`, `app/config.py`: antrean
  berbatas, timeout, rate limit, request IDs, idempotensi, shared-store adapter,
  dan konfigurasi mode deployment.
- `app/chat_schemas.py`, `app/chat_routes.py`, serta bagian kecil service chat:
  kontrak lock katalog yang ditandatangani backend dan tetap kompatibel dengan
  override nama eksplisit.
- Tes UI, API, security, conversation context, dan dokumentasi deployment.
- Adapter Redis memakai dependensi yang dipaketkan tetapi hanya perlu
  dikonfigurasi dan dijalankan untuk deployment multi-worker; mode satu worker
  tidak memerlukan service Redis berjalan.

## Kriteria verifikasi

- Greeting dan saran hanya muncul pada percakapan kosong; klik saran belum
  mengirim request, dan rerun tidak menduplikasi greeting.
- Upload dan pilihan katalog dapat dipakai sendiri-sendiri di popover; gambar,
  preview, dan konteks tetap mencapai backend yang benar.
- Enter dan tombol menghasilkan satu aksi. Retry dengan key sama hanya
  memproses sekali; pesan identik yang dikirim sengaja dua kali diproses dua
  kali.
- Permintaan kasual, typo ringan, follow-up filosofi/ciri, nama tokoh eksplisit,
  dan prompt injection mempertahankan hasil grounded serta batas existing.
- Kunci backend aktif setelah pilihan pertama sukses, menolak perubahan pilihan
  katalog yang tak sah, tetapi mempertahankan override nama melalui teks.
- Antrean dan limiter memiliki batas terukur; timeout melepas slot; respons
  429/503 membawa `Retry-After`; concurrency dan idempotensi shared store
  konsisten pada dua instance.
- Status loading hilang saat sukses/gagal dan hanya menyebut jalur yang benar-
  benar diketahui UI.
- Suite regresi existing dan tes baru untuk alur terkait lulus.

## Di luar cakupan

- Migrasi framework UI atau penggantian arsitektur/provider.
- Menambahkan login/OAuth untuk identitas pengguna terverifikasi.
- Mengubah kebijakan grounding, isi knowledge base, atau kapan nama eksplisit
  boleh mengganti konteks aktif.
- Streaming tahap demi tahap atau sistem job asinkron hanya untuk menampilkan
  posisi antrean.
- Menjamin koordinasi global bila deployment multi-worker tidak mengonfigurasi
  shared store.
