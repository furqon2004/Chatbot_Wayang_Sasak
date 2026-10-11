> Catatan migrasi OpenAI: uraian provider/konfigurasi lama di bawah merupakan
> catatan historis. Implementasi aktif memakai OpenAI untuk narasi dan web;
> lihat [konfigurasi chat](chat.md). Hasil audit lama bukan validasi migrasi terbaru.

# Audit percakapan lanjutan

Working tree setelah audit 603 tes. Implementasi tidak diubah.

## Metode dan batas audit

Enam alur dijalankan melalui FastAPI TestClient dalam proses lokal, endpoint
`POST /chat`, ChatService, retrieval hybrid, cache, validasi evidence, serta
adapter Gemini yang sebenarnya. SDK provider, skor semantic, dan profil
merupakan mock sintetis. Fixture global memblokir jaringan dan mengisolasi
environment; tidak ada API nyata atau pembukaan Chroma/model produksi.

Jayangrana (0) dan Maktal (2) masing-masing diberi tiga evidence sintetis:
deskripsi, filosofi, ciri-ciri. Threshold tidak diubah. Audit ini menguji
alur kontrol, bukan akurasi budaya, mutu embedding asli, atau kemampuan
bahasa model nyata. Respons provider memilih ID, bukan menghasilkan fakta.

`tests/test_conversation_audit.py` adalah pengujian observasional: PASS berarti
alur audit berhasil dijalankan dan pemeriksaan kontrak terpenuhi, **bukan**
fitur percakapan interaktif sudah didukung. Trace dapat dicetak dengan
`python -m pytest tests/test_conversation_audit.py -q -s`.

Jumlah panggilan di bawah dihitung pada batas SDK `generate_content` mock.
Semua menggunakan `tools=[]`; tidak ada Search. Setiap alur memakai backend
dan cache baru. Kontrol tambahan disebutkan terpisah.

## Hasil

| Alur | Observasi utama | Panggilan SDK per giliran | Penilaian |
| --- | --- | --- | --- |
| Tokoh → sederhana | `Jelaskan Jayangrana` menjawab; `Jelaskan lebih sederhana` fallback sebelum retrieval/provider | 1 → 0 | Referensi lanjutan dan penyederhanaan belum didukung |
| Tokoh → filosofi | `Apa filosofinya?` fallback sebelum retrieval; `Apa filosofi tokoh ini?` menjalankan retrieval tanpa identitas dan tetap fallback | 1 → 0 → 0 | Konteks tokoh tidak diteruskan |
| Ganti tokoh → tanpa nama | Jayangrana lalu Maktal benar; `Apa filosofinya?` fallback meskipun `seen_ids=[0,2]` | 1 → 1 → 0 | Pergantian eksplisit didukung; referensi implisit belum |
| Perbandingan | Dua tokoh ditemukan, tetapi tiga evidence pertama milik satu tokoh menyebabkan fallback sebelum provider | 0 | Kegagalan penyusunan context/perbandingan |
| Evidence kurang | Pertanyaan hubungan Menak fallback lokal; pertanyaan tanggal lahir tokoh dengan pilihan ID kosong juga fallback | 1 pembuka → 0 → 1 | Fallback didukung; kecukupan pertanyaan tokoh bergantung pada pilihan provider |
| Provider gagal | Cache lama tetap melayani; cache miss mengembalikan kutipan lokal berstatus gagal; hasil gagal tidak dicache | Detail di bawah | Degradasi lokal didukung, relevansi jawaban belum terjamin |

### Kontrol konteks dan identitas

- `Jelaskan Jayangrana lebih sederhana` dengan `character_id=0` dan
  `explain_more=true` memanggil provider sekali lagi. Query provider memuat
  permintaan sederhana; query retrieval menjadi `Jelaskan Jayangrana`.
  Jawaban tetap kutipan asli. Schema provider tidak menerima teks parafrasa.
- `Apa filosofi Jayangrana?` berhasil dengan satu panggilan tambahan;
  evidence filosofi berada di urutan pertama dan sumbernya
  `profile:0:filosofi:0`. Ini dukungan pertanyaan eksplisit, bukan ingatan chat.
- Kontrol API setelah beralih ke Maktal: kirim `Apa filosofi tokoh ini?`
  dengan ID lama 0 dan `explain_more=true`. Backend memakai Jayangrana,
  tanpa mendeteksi pertentangan dengan giliran sebelumnya. Ini uji request
  ber-ID lama, bukan klaim bahwa UI saat ini otomatis mengirim ID yang salah.
- Query eksplisit tidak dapat sepenuhnya diganti oleh `character_id`:
  intent guard berjalan terlebih dahulu. Bentuk seperti `Apa filosofinya?`
  tidak dikenali sebagai domain yang didukung.

### Perbandingan

Pada context enam evidence (tiga per tokoh), `context[:3]` hanya mencakup
Jayangrana. Pemeriksaan cakupan dua ID dilakukan sebelum pemanggilan LLM,
sehingga provider tidak mendapat kesempatan memilih evidence kedua tokoh.

Kontrol dengan satu evidence per tokoh berhasil: 1 panggilan SDK, dua sumber
profil yang benar, lalu pengulangan query menghasilkan 0 panggilan tambahan
dan satu entri cache. Jawaban hanya menggabungkan dua kutipan, belum menyusun
perbandingan. Field `character` tetap 0 (tokoh pertama), walaupun jawaban
mencakup dua tokoh. UI dapat menampilkan satu gambar dan mencatat hanya
tokoh pertama melalui `remember_response`; identitas komparatif belum tepat.

### Kegagalan provider dan cache

Urutan kontrol kegagalan menggunakan error SDK 400 yang tidak di-retry:

1. Jawaban Jayangrana berhasil: total 1 panggilan, cache 1.
2. Provider dibuat gagal; query Jayangrana yang sama memakai cache:
   tambahan 0 panggilan, cache tetap 1.
3. `Apa filosofi Maktal?` dua kali: masing-masing 1 panggilan gagal,
   `mode=answer`, `llm_status=invalid_request`, `llm_used=false`.
   Identitas 2 dan sumber lokal benar; tiga kutipan dikembalikan tanpa seleksi
   provider. Cache tetap 1.
4. `Kapan Jayangrana lahir?`: 1 panggilan gagal; backend tetap mengembalikan
   profil sebagai `answer`, padahal evidence tidak memuat tanggal lahir.
   Tidak ada tanggal yang dikarang, tetapi jawaban tidak menjawab pertanyaan.
5. Setelah tiga kegagalan, circuit breaker terbuka. Provider mock dipulihkan
   tetapi request sebelum cooldown tetap tidak memanggil SDK; status unavailable.
6. Clock provider maju ke detik 61 tanpa menunggu nyata: request Maktal
   berhasil, total SDK menjadi 5 dan cache menjadi 2. Pengulangan memakai
   cache, tambahan 0 panggilan.

Respons cache menyalin status `used`/`llm_used=true` dari jawaban sebelumnya,
meskipun giliran cache hit tidak memanggil provider. Flag ini bukan penghitung
panggilan per request. UI saat ini juga belum memasukkan `invalid_request`
ke daftar status yang menampilkan keterangan fallback profil lokal.

## Context, evidence, dan sumber

- Payload provider hanya `untrusted_query` dan `untrusted_evidence`; tidak
  ada riwayat pesan, jawaban sebelumnya, atau subjek aktif percakapan.
- UI menyimpan history untuk render; pesan teks berikutnya hanya membawa
  message, awaiting_more, dan seen_ids. seen_ids dipakai untuk pilihan tokoh
  berikutnya, bukan penyelesaian kata ganti. Ya/tidak adalah follow-up terbatas.
- Evidence diverifikasi terhadap profil, ID, label, dan nama. Sumber jawaban
  memakai `wayang_db.json` dengan evidence ID yang sesuai fixture. Ini bukan
  klaim bahwa fixture berasal dari artefak produksi.
- Fallback umum yang kurang evidence tidak menyertakan sumber. Fallback
  katalog tokoh menyertakan sumber katalog tanpa evidence IDs, bukan bukti
  untuk pertanyaan yang tidak terjawab.
- Cache membedakan query/context/identitas eksplisit, tetapi tidak mempunyai
  konteks percakapan implisit untuk dibedakan. Cache bukan memori percakapan.

## Perubahan minimum yang disarankan, belum diterapkan

1. `streamlit_app.py`, `ui_client.py`, `app/chat_schemas.py`: kirim konteks
   terstruktur terbatas (subjek aktif atau pasangan tokoh, permintaan gaya,
   referensi jawaban/evidence sebelumnya). Jangan menggunakan seen_ids sebagai
   subjek aktif. Pergantian tokoh eksplisit mengganti konteks lama.
2. `app/chat.py`, `app/query_normalizer.py`: selesaikan referensi lanjutan
   terhadap konteks tervalidasi tanpa melewati guard keamanan. Jika ambigu,
   minta klarifikasi. Pertahankan maksud filosofi dalam query retrieval,
   bukan selalu menggantinya menjadi `Jelaskan <nama>`.
3. `app/chat.py`: bangun context perbandingan yang mencakup kedua tokoh sebelum
   memotong jumlah evidence. Jangan menetapkan satu `character` untuk jawaban
   dua tokoh; sediakan identitas jamak atau biarkan character kosong.
4. `app/chat_schemas.py`, `app/gemini.py`, `app/chat.py`: agar LLM benar-benar
   menyederhanakan, diperlukan kontrak jawaban baru berisi paragraf dan ID
   evidence pendukung. Pemilihan ID saja hanya bisa memilih/mengurutkan kutipan.
   Validasi ID tidak membuktikan parafrasa setia; perlu evaluasi entailment
   atau review kasus uji yang memeriksa klaim tambahan. Tetap gunakan kutipan
   sebagai fallback jika hasil tidak dapat divalidasi.
5. `app/chat.py`: saat provider gagal dan relevansi evidence belum dipastikan,
   jawab ketidakcukupan/kegagalan yang jujur. Jangan menyajikan profil generik
   sebagai jawaban pertanyaan spesifik. Bedakan status cache hit dari panggilan
   provider aktual bila dipakai untuk observabilitas.
6. `app/rag_cache.py`: jika konteks percakapan ditambahkan, masukkan subjek
   hasil resolusi, pasangan tokoh, gaya jawaban, evidence/versi sumber, dan
   versi kontrak jawaban ke key. Pertahankan TTL, kapasitas, serta penolakan
   cache untuk kegagalan dan query sensitif.

Routing web, threshold, model, artefak, dan guard uncertain tidak perlu
diubah untuk perbaikan ini. Tidak ada implementasi perbaikan dalam audit ini.

## Validasi audit

`python -m pytest tests/test_conversation_audit.py tests/test_chat_api.py
tests/test_rag_cache.py tests/test_general_rag.py tests/test_gemini.py
tests/test_ui.py tests/test_provider_call_audit.py -q`:
**175 passed**, 2 peringatan deprecation dependensi. Seluruh suite tidak
dijalankan ulang karena tidak ada perubahan implementasi. Hasil baseline
603 tes tidak diklaim sebagai pengujian ulang pada tahap ini.

Perubahan tahap ini hanya berkas tes observasional dan laporan ini; perubahan
working tree dari audit sebelumnya dipertahankan. Tidak ada perubahan artefak,
threshold, API nyata, commit, atau push.
