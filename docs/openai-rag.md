# OpenAI RAG dan web

Konfigurasi aktif: [chat](chat.md). Narasi: [local-narrative-v2](interactive-narrative.md).

`app/llm.py` mengarahkan narasi ke `OpenAIRagClient` dan pencarian ke
`app/provider.py`, keduanya SDK OpenAI dengan `LLM_API_KEY` dan `LLM_MODEL`.
Adapter seleksi ID dipertahankan untuk kompatibilitas internal; jalur narasi
produksi menggunakan structured sections. Tidak ada fallback ke Gemini.

`provider_configured` dan `provider_client_ready` pada health menunjukkan
konfigurasi/client, bukan keberhasilan koneksi atau validitas credential.
Model default gpt-4.1-mini; kegagalan menghasilkan fallback lokal yang jujur.
Migrasi diuji dengan mock; belum diverifikasi melalui API nyata.
