"""HTTP-only UI client. No inference, disk persistence, or payload logging."""
import os

import httpx


class BackendError(Exception):
    """A safe, user-facing failure; never contains a server response body."""


def request_api(path, *, payload=None, image=None):
    base = os.getenv("WAYANG_API_URL", "http://127.0.0.1:8000").rstrip("/")
    token = os.getenv("WAYANG_SERVICE_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=105, follow_redirects=False, trust_env=False) as client:
            if path == "/health":
                response = client.get(base + path, timeout=3)
            elif image is not None:
                content, mime = image
                response = client.post(base + path, headers=headers,
                                       files={"file": ("upload", content, mime)})
            else:
                response = client.post(base + path, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Invalid response")
            return data
    except httpx.HTTPStatusError as exc:
        messages = {
            401: "Akses backend belum dikonfigurasi. Hubungi pengelola.",
            403: "Akses backend ditolak. Hubungi pengelola.",
            404: "Fitur ini belum aktif di backend. Hubungi pengelola.",
            413: "Gambar terlalu besar. Gunakan gambar dengan ukuran atau resolusi lebih kecil.",
            415: "Gunakan gambar JPEG, PNG, atau WebP yang valid.",
            422: "Input belum valid. Periksa pertanyaan atau gambar Anda.",
            429: "Layanan sedang sibuk. Tunggu sebentar lalu coba lagi.",
            500: "Gangguan internal backend (HTTP 500). Silakan coba lagi.",
            502: "Gateway backend bermasalah (HTTP 502). Silakan coba lagi.",
            503: "Layanan backend belum siap (HTTP 503). Silakan coba lagi.",
            504: "Gateway backend kehabisan waktu (HTTP 504). Silakan coba lagi.",
        }
        raise BackendError(messages.get(exc.response.status_code,
                           f"Permintaan ditolak backend (HTTP {exc.response.status_code}). Silakan coba lagi.")) from None
    except httpx.TimeoutException:
        raise BackendError("Waktu tunggu backend habis. Silakan coba lagi.") from None
    except httpx.HTTPError:
        raise BackendError("Koneksi ke backend gagal. Silakan coba lagi.") from None
    except ValueError:
        raise BackendError("Format respons backend tidak valid. Silakan coba lagi.") from None


def send_message(text, uploaded=None, *, awaiting_more=False, seen_ids=(), character_id=None, explain_more=False):
    if uploaded is not None:
        if not uploaded.size or uploaded.size > 10 * 1024 * 1024:
            raise BackendError("Gunakan gambar tidak kosong dengan ukuran maksimal 10 MB.")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp"}.get(uploaded.name.rsplit(".", 1)[-1].lower())
        if mime is None:
            raise BackendError("Gunakan gambar JPEG, PNG, atau WebP.")
        return "image", request_api("/predict-image", image=(uploaded.getvalue(), mime))
    if not text.strip() or not any(c.isalnum() for c in text):
        raise BackendError("Tulis pertanyaan atau lampirkan gambar terlebih dahulu.")
    payload = {"message": text.strip()}
    if character_id is not None:
        payload["character_id"] = character_id
    if explain_more:
        payload["explain_more"] = True
    if awaiting_more:
        payload.update(awaiting_more=True, seen_ids=list(seen_ids))
    return "chat", request_api("/chat", payload=payload)
