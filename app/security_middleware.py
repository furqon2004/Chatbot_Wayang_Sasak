"""ASGI limits apply before FastAPI multipart/JSON parsing."""
import asyncio
import hashlib
import hmac
import time
from uuid import uuid4

from starlette.responses import JSONResponse

from .security import Capacity, MemoryRateLimiter, REQUEST_ID, event


class SecurityMiddleware:
    def __init__(self, app, settings, max_upload_bytes, limiter=None, capacity=None):
        self.app, self.settings, self.max_upload_bytes = app, settings, max_upload_bytes
        self.limiter = limiter or MemoryRateLimiter(settings.requests_per_minute)
        self.capacity = capacity or Capacity(settings.concurrency)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id, started = uuid4().hex, time.monotonic()
        reset_id = REQUEST_ID.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        path, method = scope.get("path", ""), scope.get("method", "")
        endpoint = path if path in {"/search", "/chat", "/predict-image", "/health"} else "other"
        status, category, sent, admitted = 500, "ok", False, False

        async def safe_send(message):
            nonlocal status, sent
            if message["type"] == "http.response.start":
                sent, status = True, message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode()), (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"), (b"referrer-policy", b"no-referrer"),
                    (b"cache-control", b"no-store"),
                    (b"content-security-policy", b"frame-ancestors 'none'"),
                ]
            await send(message)

        async def reject(code, detail, reason):
            nonlocal category
            category = reason
            await JSONResponse({"detail": detail, "request_id": request_id}, status_code=code)(scope, receive, safe_send)

        try:
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            peer = (scope.get("client") or ("unknown", 0))[0]
            if endpoint in {"/search", "/chat", "/predict-image"} and method != "OPTIONS":
                # Transport peer only. Never trust X-Forwarded-For from clients.
                if not await self.limiter.allow(f"{peer}:{endpoint}"):
                    return await reject(429, "Batas request tercapai", "rate_limit")
            if path != "/health" and method != "OPTIONS" and self.settings.service_token:
                actual = headers.get(b"authorization", b"")
                expected = ("Bearer " + self.settings.service_token).encode()
                if not hmac.compare_digest(hashlib.sha256(actual).digest(), hashlib.sha256(expected).digest()):
                    return await reject(401, "Autentikasi diperlukan", "authentication")
            if endpoint in {"/search", "/chat", "/predict-image"} and method != "OPTIONS":
                if not self.capacity.acquire():
                    return await reject(503, "Layanan sibuk", "concurrency")
                admitted = True
                limit = self.max_upload_bytes + 65536 if endpoint == "/predict-image" else 16384
                try:
                    declared = int(headers.get(b"content-length", b"0"))
                except ValueError:
                    return await reject(400, "Request tidak valid", "body")
                if declared < 0 or declared > limit:
                    return await reject(413, "Request terlalu besar", "body")
                body, deadline = bytearray(), time.monotonic() + self.settings.body_timeout
                while True:
                    part = await asyncio.wait_for(receive(), max(0, deadline - time.monotonic()))
                    if part["type"] == "http.disconnect":
                        return
                    if len(body) + len(part.get("body", b"")) > limit:
                        return await reject(413, "Request terlalu besar", "body")
                    body.extend(part.get("body", b""))
                    if not part.get("more_body", False):
                        break
                delivered = False
                async def buffered_receive():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()
                await self.app(scope, buffered_receive, safe_send)
            else:
                await self.app(scope, receive, safe_send)
        except asyncio.TimeoutError:
            if not sent:
                await reject(408, "Request timeout", "timeout")
        except Exception:
            if not sent:
                await reject(500, "Layanan gagal", "internal")
        finally:
            if admitted:
                self.capacity.release()
            event(endpoint=endpoint, method=method if method in {"GET", "POST", "OPTIONS"} else "other",
                  status=status, category=category, latency_ms=round((time.monotonic() - started) * 1000, 2))
            REQUEST_ID.reset(reset_id)
