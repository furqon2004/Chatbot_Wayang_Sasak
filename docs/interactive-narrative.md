# Narasi interaktif RAG OpenAI

OpenAI memakai kontrak `local-narrative-v2`, terpisah dari seleksi ID lama.
Metode `select` lama tetap tersedia untuk kompatibilitas; jalur chat OpenAI
sekarang memanggil `narrate` sekali per operasi (dengan retry aplikasi yang sudah
dibatasi). Search sekarang memakai OpenAI web_search. Tidak ada API nyata dalam implementasi ini.

Output terstruktur memuat `status` dan maksimal enam `sections`. Setiap bagian
berisi `text`, `kind` (explanation/comparison), `subject_ids`, dan `evidence_ids`.
Model tidak mengirim URL atau metadata sumber. Backend memetakan ID ke context
evidence lokal yang sudah diperiksa, membangun `ChatSource`, dan menambahkan
nomor sitasi per bagian. UI menampilkan nomor yang sama pada daftar sumber.

Gaya explain/summary/simplify/detail mengikuti pesan pengguna. Referensi tunggal
tanpa subjek meminta klarifikasi; perbandingan dua tokoh mempertahankan cakupan
keduanya. Penyederhanaan/ringkasan lanjutan memakai ulang evidence sebelumnya
yang dibaca dari katalog backend, bukan teks narasi model pada giliran lama.
Pertanyaan filosofi/ciri tetap dibatasi ke field yang relevan. Penjelasan lanjut
boleh memakai hasil retrieval baru yang memenuhi batas dan threshold lama.

Cache tetap process-local, TTL 900 detik, 32 entri. Key mencakup provider/model,
query, gaya, subjek, konteks, evidence, instruksi/schema, dan versi kontrak.
Output gagal, insufficient, atau ditolak tidak disimpan. Query/context sensitif
dan narasi yang terdeteksi sensitif tidak disimpan. Tidak ada cache ke disk.

## Validasi dan batas faktual

### Batas domain dan instruksi tidak tepercaya

Pesan asli diperiksa sebelum retrieval, pengayaan konteks, profil dari tombol,
atau provider. Nama domain bukan izin untuk tugas pemrograman, politik, dan
topik lain yang terdeteksi. Klausa campuran hanya diteruskan jika bagian dalam
domain dapat dipisahkan secara jelas; sisanya meminta fokus pada Wayang Sasak
dan tetap menawarkan katalog. Injection menolak seluruh permintaan.

Kata umum seperti “filosofi”, “ciri”, atau “sejarah” sendiri tidak menetapkan
domain. Follow-up terbatas memakai konteks/subjek yang divalidasi; konteks lama
tidak mengizinkan perpindahan topik. Discovery peran tetap lokal. Routing web
memakai pemeriksaan domain yang sama; Search dalam domain tetap memerlukan
sumber, sedangkan evidence lokal kosong menghasilkan fallback keterbatasan.

Prompt narasi menegaskan bahwa pesan, konteks, dan evidence adalah data, bukan
instruksi pengganti. Guard evidence menyaring pola instruksi berbahaya sebelum
provider. Instruksi parafrasa juga melarang perluasan “kebijaksanaan” menjadi
seluruh “pemikiran” dan meminta perbandingan ringkas tanpa repetisi.

**Batas keamanan:** pemeriksaan domain dan injection bersifat deterministik dan
heuristik, bukan klasifikasi semantik lengkap. Bahasa lain, ejaan tersamarkan,
encoding baru, atau instruksi implisit dapat lolos; pertanyaan sah juga dapat
ditolak. Konteks divalidasi terhadap katalog, bukan ditandatangani sebagai bukti
riwayat percakapan. Tidak ada jaminan bebas prompt injection atau fakta tambahan.
Perubahan instruksi otomatis membedakan cache karena instruksi masuk cache key.

Validasi meliputi schema, keanggotaan ID pada context saat ini, duplikasi,
kepemilikan ID, cakupan subjek/bagian, nama tokoh yang tidak didukung sumber,
status kecukupan, pola URL/atribusi, angka baru, perubahan keberadaan negasi,
serta pola klaim keunggulan yang tidak ada dalam evidence. Error menghasilkan
fallback jujur dan pilihan profil lokal; tidak mengklaim narasi berhasil.

Fallback membedakan layanan provider tidak tersedia, bukti basis pengetahuan
lokal tidak cukup, dan penjelasan yang belum dapat diverifikasi karena output
ditolak. Ketiganya tidak disimpan dalam cache. Pesan belum dapat diverifikasi
tidak berarti output yang lolos guard telah terbukti benar secara faktual.

Diagnostik OpenAI hanya mencatat kategori, jenis exception dari daftar tetap,
HTTP status numerik yang valid, dan kode provider dari allowlist. Nilai yang
tidak dikenal disembunyikan; payload, pesan exception, respons mentah, credential,
prompt, project ID, request ID, dan header tidak dicatat oleh logger adapter.
Metadata ini hanya tersedia untuk error berikutnya, bukan rekonstruksi error lama.

Smoke narasi terakhir berhenti pada pertanyaan pertama setelah satu panggilan
SDK dengan kategori `unavailable`. Jenis exception, HTTP status, dan kode provider
tidak tersimpan; logger dinonaktifkan oleh harness smoke tersebut. Tidak ada
catatan error OpenAI terkait dalam empat file log backend/UI lokal yang diperiksa.
Penyebab tidak dapat disimpulkan dari `unavailable`. Harness berikutnya perlu
mengaktifkan logger `app.openai_rag` agar metadata aman ini dapat tersimpan.

**ID valid bukan bukti kesetiaan narasi.** Guard angka/negasi bersifat heuristik,
bukan entailment. Bisa terjadi false rejection atas parafrasa yang valid, atau
false acceptance atas fakta nonnumerik baru, perubahan relasi, subjek dalam
perbandingan, sebab-akibat, maupun nuansa ketidakpastian. Tidak ada verifikator
semantik independen pada implementasi ini. Karena itu status `used` menunjukkan
narasi lolos pemeriksaan yang tersedia, bukan sertifikasi kebenaran faktual.

Evaluasi mock `tests/test_narrative.py` mencakup:

- Fakta tanggal tambahan: ditolak oleh guard angka.
- Negasi dihilangkan/ditambahkan: ditolak oleh guard polaritas sederhana.
- Tokoh yang tidak tercakup evidence: ditolak.
- Salah satu sumber perbandingan hilang: ditolak.
- Klaim lebih unggul dengan kedua ID valid tetapi tanpa relasi pendukung: ditolak.
- Fakta baru nonnumerik dengan ID valid: contoh yang **masih lolos**, dicatat
  eksplisit oleh tes batas validasi, bukan dianggap jawaban faktual yang benar.
- Pertukaran makna dua tokoh dengan kedua ID valid: **masih lolos**; cakupan
  tokoh tidak membuktikan bahwa setiap predikat melekat pada tokoh yang benar.
- Teks kosong dan perbandingan yang salah berlabel explanation: ditolak.

Kontrak API mengikuti [OpenAI Docs: Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Schema ketat mengatur bentuk output, bukan membuktikan kesetiaan faktual.

## Contoh mock (fixture sintetis, bukan audit fakta budaya)

Evidence `profile:0:filosofi:0`: “Dalam fixture, Jayangrana menjadi lambang raga
manusia.” Narasi: “Jayangrana menggambarkan raga manusia. [1]”

Evidence tambahan `profile:2:filosofi:0`: “Dalam fixture, Maktal melambangkan
keberanian.” Narasi perbandingan: “Jayangrana melambangkan raga manusia,
sedangkan Maktal melambangkan keberanian. [1] [2]”

Backend menghasilkan kedua sumber sebagai referensi `wayang_db.json` dengan
ID evidence masing-masing. Model hanya menghasilkan isi bagian dan ID.

Discovery peran tetap deterministik tanpa provider; upload tetap on-demand;
uncertain tidak menetapkan identitas final. Tidak mengubah threshold, CLIP,
ChromaDB, JSON, dataset, atau artefak. Smoke OpenAI seleksi-ID sebelumnya sukses;
kontrak narasi baru ini belum diuji live. Search live tetap berstatus 429 lama.

## Verifikasi working tree (10 Oktober 2026)

- Seluruh suite: **695 passed**, 3 warnings (dua deprecation dependency dan
  satu warning PIL dari tes batas resolusi), 24,32 detik.
- `compileall` untuk app, tests, scripts, evaluation, dan entrypoint/UI: lulus.
- `pip check` pada `.venv`: tidak ada dependency rusak.
- `scripts/security_scan.py`: tidak ada temuan dalam allowlist source;
  `.env` dan artefak dikecualikan oleh scanner.
- `git diff --check`: lulus.
- Provider diuji melalui mock, termasuk SDK HTTP transport, schema nested,
  cache hit/miss, follow-up, sumber lokal, kegagalan, dan penolakan output.
  Fixture tes memblokir jaringan. Tidak ada API provider nyata, commit, atau push.
