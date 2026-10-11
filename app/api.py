from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings
from .inference import predict_image
from .resources import ModelResources, ResourceValidationError, load_resources
from .catalog import Catalog
from .retrieval import RetrievalService
from .retrieval_resources import load_retrieval
from .search_routes import router as search_router
from .security_middleware import SecurityMiddleware
from .security import event
from .schemas import PredictionResponse
from .memory_upload import MemoryUploadRoute

LOGGER = logging.getLogger(__name__)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
READ_CHUNK_SIZE = 1024 * 1024


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(READ_CHUNK_SIZE):
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Ukuran file melebihi batas {limit} byte",
            )
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="File gambar kosong")
    return b"".join(chunks)


def _decode_image(data: bytes, max_pixels: int = 16_000_000) -> Image.Image:
    try:
        with io.BytesIO(data) as buffer, Image.open(buffer, formats=tuple(ALLOWED_IMAGE_FORMATS)) as source:
            if source.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail="Format gambar tidak didukung")
            if (
                source.width * source.height > max_pixels
                or (Image.MAX_IMAGE_PIXELS is not None and source.width * source.height > Image.MAX_IMAGE_PIXELS)
            ):
                raise HTTPException(status_code=413, detail="Resolusi gambar terlalu besar")
            source.load()
            return source.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=413, detail="Resolusi gambar terlalu besar") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="File rusak atau bukan gambar valid") from exc


def create_app(
    settings: Settings | None = None,
    resource_loader: Callable[[Settings], ModelResources] = load_resources,
    retrieval_loader=load_retrieval,
    provider_loader=None,
    rate_limiter=None,
    capacity=None,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    active_provider_loader = provider_loader
    if active_provider_loader is None and active_settings.chat_enabled:
        from .llm import load_llm
        active_provider_loader = load_llm

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        LOGGER.info("Startup backend Wayang Sasak")
        app.state.ready = False
        app.state.resources = None
        app.state.resources = await run_in_threadpool(resource_loader, active_settings)
        catalog = Catalog(app.state.resources.labels, app.state.resources.characters)
        retrieval_resources = None
        try:
            retrieval_resources = await run_in_threadpool(retrieval_loader, active_settings, catalog)
        except Exception:
            if active_settings.security.production:
                app.state.resources = None
                LOGGER.warning("Production retrieval startup validation failed")
                raise ResourceValidationError("Resource retrieval production tidak tersedia") from None
            # Search gagal terkontrol tanpa menonaktifkan prediksi gambar/katalog.
            LOGGER.warning("Retrieval unavailable saat startup")
        semantic = retrieval_resources.semantic if retrieval_resources else None
        app.state.retrieval = RetrievalService(catalog, semantic, active_settings.search_threshold)
        provider = None
        try:
            if active_provider_loader is not None:
                provider = active_provider_loader(active_settings)
        except Exception:
            LOGGER.warning("Provider unavailable saat startup")
        if active_provider_loader is not None:
            from .chat import ChatService
            app.state.chat = ChatService(app.state.retrieval, provider, active_settings.llm_timeout_seconds,
                                         active_settings.web_timeout_seconds)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.resources = None
            app.state.retrieval = None
            app.state.chat = None
            try:
                if provider is not None:
                    await provider.close()
            finally:
                if retrieval_resources is not None:
                    await run_in_threadpool(retrieval_resources.close)
            LOGGER.info("Shutdown backend Wayang Sasak")

    app = FastAPI(
        title="API Prediksi Wayang Sasak", version="1.0.0", lifespan=lifespan, debug=False,
        docs_url="/docs" if not active_settings.security.production or active_settings.security.public_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not active_settings.security.production or active_settings.security.public_docs else None,
    )
    if active_settings.security.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(active_settings.security.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"],
                           allow_credentials=False)
    if active_settings.security.production:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(active_settings.security.allowed_hosts))
    app.add_middleware(SecurityMiddleware, settings=active_settings.security,
                       max_upload_bytes=active_settings.max_upload_bytes, limiter=rate_limiter, capacity=capacity)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": "Input tidak valid", "request_id": request.state.request_id}, status_code=422)
    app.state.ready = False
    app.state.resources = None
    app.state.settings = active_settings
    app.state.retrieval = None
    app.state.chat = None
    app.include_router(search_router)
    # Modul Tahap 3 lama dipertahankan, tetapi aplikasi default hanya Tahap 1–2.
    if active_provider_loader is not None:
        from .chat_routes import router as chat_router
        app.include_router(chat_router)

    @app.get("/health")
    async def health(request: Request):
        resources = request.app.state.resources
        if not request.app.state.ready or resources is None:
            raise HTTPException(status_code=503, detail="Model belum siap")
        semantic = request.app.state.retrieval.semantic if request.app.state.retrieval else None
        if active_settings.security.production:
            return {"status": "ok"}
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": "CLIP ViT-B/32 + linear classifier",
            "device": resources.device.type,
            "num_classes": len(resources.labels),
            "knowledge_base_loaded": len(resources.characters) == 20,
            "retrieval_loaded": semantic is not None,
            "collection_name": active_settings.collection_name if semantic is not None else None,
            "document_count": semantic.document_count if semantic is not None else 0,
            "provider_configured": bool(active_settings.llm_api_key),
            "provider_client_ready": getattr(request.app.state.chat.provider, "web_client", request.app.state.chat.provider) is not None if request.app.state.chat else False,
        }

    prediction_router = APIRouter(route_class=MemoryUploadRoute)

    @prediction_router.post("/predict-image", response_model=PredictionResponse, response_model_exclude_unset=True)
    async def prediction(request: Request, file: UploadFile = File(...)):
        resources = request.app.state.resources
        if not request.app.state.ready or resources is None:
            raise HTTPException(status_code=503, detail="Model belum siap")
        started = time.perf_counter()
        try:
            if file.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(status_code=415, detail="Tipe file harus JPEG, PNG, atau WebP")
            data = await _read_limited(file, active_settings.max_upload_bytes)
            image = await run_in_threadpool(_decode_image, data, active_settings.security.max_pixels)
            try:
                result = await run_in_threadpool(
                    predict_image, image, resources,
                    min_confidence=active_settings.min_confidence,
                    min_margin=active_settings.min_margin,
                )
            finally:
                image.close()
        except HTTPException:
            raise
        except Exception:
            LOGGER.error("Inferensi gambar gagal: internal")
            raise HTTPException(status_code=500, detail="Inferensi gambar gagal") from None
        finally:
            await file.close()
        event(mode=result["status"], category="_and_".join(result["decision"]["reasons"]) or "thresholds_met",
              latency_ms=round((time.perf_counter() - started) * 1000, 2))
        return result

    app.include_router(prediction_router)
    return app
