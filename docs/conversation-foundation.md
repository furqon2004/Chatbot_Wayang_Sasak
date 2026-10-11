> Catatan migrasi OpenAI: uraian provider/konfigurasi lama di bawah merupakan
> catatan historis. Implementasi aktif memakai OpenAI untuk narasi dan web;
> lihat [konfigurasi chat](chat.md). Hasil audit lama bukan validasi migrasi terbaru.

# Fondasi percakapan berbasis evidence

Pembaruan: jalur chat OpenAI kini memakai [narasi interaktif](interactive-narrative.md).
Kontrak seleksi ID dan penolakan penyederhanaan yang dijelaskan di bawah berlaku
pada jalur lama/Gemini; OpenAI mendukung parafrasa terbatas dengan sitasi per bagian
dan batas validasi faktual yang dijelaskan dalam dokumen baru.

Implementasi lanjutan dari temuan historis `conversation-audit.md`.
Kontrak provider tetap memilih evidence IDs. Tidak ada parafrasa bebas,
perubahan model, threshold, artefak, atau pengaktifan KB umum.

## Kontrak konteks

Request `/chat` menerima `conversation_context` opsional:

```json
{
  "subject_ids": [0],
  "evidence_ids": ["profile:0:deskripsi:0"],
  "intent": "profile"
}
```

Maksimal dua subjek dan 12 evidence IDs; tidak boleh duplikat. Field tambahan,
teks evidence dari frontend, ID di luar katalog, dan format ID salah ditolak
oleh schema. Backend mengecek kepemilikan ID, keberadaan field/index, cakupan
subjek, isi yang tidak kosong, serta keamanan isi terhadap katalog lokal.
Isi evidence sebelumnya dibaca ulang dari backend, bukan dari browser.
Konteks semantik yang tidak cocok menghasilkan klarifikasi tanpa provider;
payload yang tidak memenuhi schema menghasilkan HTTP 422.

Ini referensi stateless yang tervalidasi, bukan bukti kriptografis bahwa suatu
jawaban pernah ditampilkan. Subjek tanpa evidence diperbolehkan untuk hasil
identifikasi gambar yang accepted. `seen_ids` tetap hanya untuk saran tokoh.

## Resolusi dan tampilan

- `Apa filosofinya?` dan `Apa cirinya?` menggunakan subjek aktif. Maksud
  diturunkan dari pesan terbaru, tidak dipercaya dari field intent frontend.
- Nama eksplisit terbaru mengalahkan subjek konteks lama dan `character_id`
  yang berbeda. Konteks malformed tetap harus diklarifikasi/ditolak.
- Dua subjek dengan kata ganti tunggal membutuhkan klarifikasi. Bentuk
  `Apa filosofi keduanya?` menggunakan kedua subjek. Guard keamanan tetap
  memeriksa pesan asli sebelum resolusi.
- Retrieval mempertahankan maksud filosofi/ciri; context dibatasi ke field
  yang sesuai. Field yang tidak tersedia ditolak sebelum provider dipanggil.
- Context perbandingan disusun bergantian per tokoh sebelum batas global.
  Kedua identitas harus tercakup pada context dan hasil seleksi. Respons
  memakai `subjects` dan `character=null`, bukan satu profil tokoh.
- UI meneruskan konteks respons, mencatat kedua subjek, dan menampilkan nama
  pasangan. Gambar accepted menetapkan subjek tanpa memanggil chat otomatis;
  gambar uncertain menghapus konteks aktif dan tidak menetapkan kandidat.
- `Jelaskan lebih sederhana` memberi keterangan bahwa penyederhanaan belum
  tersedia; kutipan yang sama tidak diklaim sebagai parafrasa.

## Kegagalan dan cache

Kegagalan/timeout/output tidak valid dari penyusun menghasilkan fallback
yang menyatakan penjelasan belum dapat diverifikasi. Tidak menampilkan profil
generik sebagai jawaban spesifik, tidak menyertakan sumber semu, dan tidak
menyimpan hasil gagal ke cache. Tombol profil lokal tetap tersedia.

Cache key menyertakan subjek dan intent hasil resolusi serta evidence
sebelumnya yang dibaca ulang. Perubahan isinya membedakan key meskipun ID
tetap. Query eksplisit baru tidak bergantung pada evidence giliran lama.
Privasi diperiksa pada pesan asli, query hasil resolusi, context baru, dan
isi evidence sebelumnya. TTL 900 detik, kapasitas 32, dan copy isolation tetap.

`llm_status`/`llm_used` mempertahankan arti lama, termasuk pada jawaban cache.
Field tambahan:

- `cache_hit`: hit cache RAG pada request ini (bukan indikator cache web).
- `provider_requests`: jumlah pemanggilan SDK generate_content pada request
  ini, termasuk retry yang dilakukan adapter; 0 untuk cache, klarifikasi,
  circuit breaker terbuka, dan profil lokal. Dihitung dengan meter per-request
  yang terisolasi melalui ContextVar, bukan penghitung global.

SDK selalu dimock dalam pengujian. Mock yang mengganti `select` langsung
tidak melewati SDK dan karena itu tidak menambah provider_requests.

## Batas yang tersisa

Referensi bahasa dibatasi pada pola yang diuji, bukan pemahaman dialog bebas.
Pasangan yang ambigu harus disebutkan kembali. Backend tidak menyimpan history
panjang. Penilaian kecukupan untuk pertanyaan spesifik di luar field filosofi/
ciri masih bergantung pada pemilihan ID kosong oleh provider; ketika provider
gagal sistem memilih fallback. Tidak tersedia validasi entailment atau
parafrasa bebas. Kekurangan KB umum dalam audit terdahulu masih berlaku.

Tes fondasi berada di `tests/test_conversation_context.py`; tes UI→HTTP→backend
→SDK mock pada `tests/test_provider_call_audit.py` membuktikan konteks pasangan,
field evidence, klarifikasi tanpa request, cache hit, dan penghitung aktual.

## Hasil verifikasi

- Tes terkait chat/cache/UI/provider: 162 passed. Setelah perbaikan prioritas
  evidence ciri, tes konteks/retrieval/katalog: 39 passed.
- Seluruh suite final: **627 passed**, 3 warnings (dua deprecation dependensi
  dan peringatan PIL pada tes batas ukuran gambar).
- `compileall` untuk app, tests, scripts, dan modul root: lulus.
- `pip check`: No broken requirements found.
- Secret scan allowlist proyek dan pemeriksaan tambahan modul UI: tidak ada
  temuan. `.env` dan artefak tetap dikecualikan dari scan.
- `git diff --check`: lulus; Git memberi pemberitahuan normalisasi LF/CRLF
  untuk rag_cache.py, bukan kesalahan whitespace.

Tidak ada API nyata, commit, push, atau perubahan JSON/ChromaDB/CLIP/dataset.


## Pencarian identitas berdasarkan peran dan sifat

Audit read-only `artifacts/wayang_db.json` menemukan pada deskripsi dan
deskripsi_lengkap Jayangrana pernyataan eksplisit: ?Jayangrana merupakan tokoh
utama dalam pewayangan Sasak yang digambarkan sebagai seorang kesatria dengan
kecerdasan, kebijaksanaan, dan akhlak mulia.? Umar Maya juga disebut sebagai
penasihat utama. JSON tidak diubah. Klaim tersebut berasal dari profil lokal;
audit ini tidak memverifikasi sumber budaya eksternal.

`role_search.py` memakai ranking retrieval yang ada dan threshold yang sama,
lalu memeriksa kalimat lengkap pada deskripsi/deskripsi_lengkap milik setiap
kandidat. Identitas tidak dipetakan dari pertanyaan ke ID tetap. Hanya predikat
langsung tentang tokoh yang diterima; kemiripan, negasi, dugaan, atau frasa
seperti ?pendamping tokoh utama? tidak membuktikan peran itu sendiri.

Pertanyaan peran didahulukan sebelum resolusi profil aktif. Konteks dengan
domain Wayang Sasak, termasuk fallback pencarian web yang eksplisit, cukup
untuk memahami ?Siapa tokoh utamanya??. Tanpa konteks, backend meminta
klarifikasi. Pencarian ini tidak mengganti pertanyaan pengantar umum.

Satu kandidat menghasilkan pembuka langsung, gambar dari pemetaan katalog,
kutipan deskripsi dan evidence ID; subjek aktif ditetapkan untuk pertanyaan
filosofi/ciri berikutnya. Beberapa kandidat menghasilkan maksimal tiga pilihan
dengan kutipan alasan dan tidak menetapkan satu subjek. Evidence tidak cukup
menghasilkan fallback jujur. Identifikasi peran deterministik tidak memanggil
Gemini. Penjelasan lanjutan tetap memakai kontrak RAG dan cache yang ada.

Pengenalan peran/sifat sengaja dibatasi pada pola pertanyaan identitas dan
predikat eksplisit yang diuji; ini bukan entailment bahasa bebas atau pencarian
sinonim umum. Bentuk yang tidak didukung tidak boleh dipaksakan menjadi identitas.
Tes tambahan: `tests/test_role_search.py` dan alur AppTest/HTTP mock dalam
`tests/test_provider_call_audit.py`.


Verifikasi tambahan pencarian peran (10 Oktober 2026): seluruh suite **647
passed**, 3 warnings. Tes terarah deduplikasi dan peran **34 passed**; alur
AppTest membuktikan discovery 0 panggilan provider dan follow-up filosofi 1
panggilan SDK mock. `compileall`, `pip check`, secret scan, dan `git diff --check`
lulus. Tidak ada panggilan provider nyata, commit, push, atau perubahan artefak.
