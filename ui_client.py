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
        with httpx.Client(timeout=75, follow_redirects=False, trust_env=False) as client:
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
        }
        raise BackendError(messages.get(exc.response.status_code,
                           "Layanan belum dapat memproses permintaan. Silakan coba lagi.")) from None
    except (httpx.HTTPError, ValueError):
        raise BackendError("Backend tidak tersedia atau respons tidak valid. Silakan coba lagi.") from None


def send_message(text, uploaded=None):
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
    return "chat", request_api("/chat", payload={"message": text.strip()})
