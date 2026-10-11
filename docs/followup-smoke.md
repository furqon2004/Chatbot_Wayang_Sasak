# Smoke lanjutan lokal

Input `smoke-jayangrana-first.json` adalah hasil sukses yang diberikan pengguna,
bukan respons yang dibuat ulang oleh skrip. Konteks direkonstruksi dari sumber
respons pertama, lalu ID dan teks evidence diperiksa kembali terhadap JSON lokal.
Jika evidence berubah, smoke berhenti sebelum mengirim permintaan.

Jalankan dari PowerShell di `C:\Chatbot_Wayang_Sasak`:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\local_narrative_followup_smoke.py --previous .\docs\smoke-jayangrana-first.json
```

Skrip menanyakan “Jelaskan lebih sederhana” dengan konteks pertama, kemudian
“Bandingkan filosofi Jayangrana dan Maktal”. Jalur ChatService memeriksa kecukupan
evidence kedua tokoh sebelum memanggil provider. Fallback diterima dan menghentikan
urutan; tidak ada pergantian pertanyaan atau usaha memaksakan jawaban.

Maksimal dua panggilan SDK total, masing-masing dijaga batas satu panggilan.
Model gpt-4.1-mini, kontrak local-narrative-v2, aplikasi satu percobaan, SDK tanpa
retry, tanpa Search. HTTPX trust_env=False hanya untuk skrip, TLS tetap aktif.
Tidak mengubah proxy, .env, default aplikasi, atau artefak sumber.

Setelah dua respons berhasil, Streamlit AppTest memutar ulang ketiga respons nyata
melalui UI dan ui_client, memeriksa penerusan konteks serta teks narasi dan nomor
sumber di tampilan. Endpoint backend digantikan replay dalam pemeriksaan UI ini:
tidak ada API tambahan. Ini membuktikan perilaku UI pada AppTest, bukan konektivitas
deployment backend atau pemeriksaan visual browser manual.

Output memuat hasil kedua alur dan status ui_smoke. Kesetiaan narasi tetap perlu
dinilai manusia terhadap evidence yang dicetak. Nomor sitasi yang cocok atau
status used tidak membuktikan semua klaim benar. Tidak ada verifikasi budaya
eksternal dalam smoke ini.
