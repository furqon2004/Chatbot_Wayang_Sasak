from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .inference import predict_image
from .resources import ModelResources, load_resources
from .catalog import Catalog
from .retrieval import RetrievalService
from .retrieval_resources import load_retrieval
from .search_routes import router as search_router
from .chat import ChatService
from .chat_routes import router as chat_router
from .gemini import load_gemini

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


def _decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail="Format gambar tidak didukung")
            if (
                Image.MAX_IMAGE_PIXELS is not None
                and source.width * source.height > Image.MAX_IMAGE_PIXELS
            ):
                raise HTTPException(status_code=413, detail="Resolusi gambar terlalu besar")
            source.load()
            return source.convert("RGB")
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Resolusi gambar terlalu besar") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="File rusak atau bukan gambar valid") from exc


def create_app(
    settings: Settings | None = None,
    resource_loader: Callable[[Settings], ModelResources] = load_resources,
    retrieval_loader=load_retrieval,
    gemini_loader=load_gemini,
) -> FastAPI:
    active_settings = settings or Settings.from_env()

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
            # Search gagal terkontrol tanpa menonaktifkan prediksi gambar/katalog.
            LOGGER.warning("Retrieval unavailable saat startup")
        semantic = retrieval_resources.semantic if retrieval_resources else None
        app.state.retrieval = RetrievalService(catalog, semantic, active_settings.search_threshold)
        gemini = None
        try:
            gemini = gemini_loader(active_settings)
        except Exception:
            LOGGER.warning("Gemini unavailable saat startup")
        app.state.chat = ChatService(app.state.retrieval, gemini, active_settings.gemini_timeout_seconds)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.resources = None
            app.state.retrieval = None
            app.state.chat = None
            try:
                if gemini is not None:
                    await gemini.close()
            finally:
                if retrieval_resources is not None:
                    await run_in_threadpool(retrieval_resources.close)
            LOGGER.info("Shutdown backend Wayang Sasak")

    app = FastAPI(
        title="API Prediksi Wayang Sasak", version="1.0.0", lifespan=lifespan
    )
    app.state.ready = False
    app.state.resources = None
    app.state.settings = active_settings
    app.state.retrieval = None
    app.state.chat = None
    app.include_router(search_router)
    app.include_router(chat_router)

    @app.get("/health")
    async def health(request: Request):
        resources = request.app.state.resources
        if not request.app.state.ready or resources is None:
            raise HTTPException(status_code=503, detail="Model belum siap")
        semantic = request.app.state.retrieval.semantic if request.app.state.retrieval else None
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
            "gemini_configured": bool(active_settings.gemini_api_key),
            "gemini_client_ready": request.app.state.chat.gemini is not None if request.app.state.chat else False,
        }

    @app.post("/predict-image")
    async def prediction(request: Request, file: UploadFile = File(...)):
        resources = request.app.state.resources
        if not request.app.state.ready or resources is None:
            raise HTTPException(status_code=503, detail="Model belum siap")
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Tipe file harus JPEG, PNG, atau WebP")

        started = time.perf_counter()
        try:
            data = await _read_limited(file, active_settings.max_upload_bytes)
            image = await run_in_threadpool(_decode_image, data)
            try:
                result = await run_in_threadpool(predict_image, image, resources)
            finally:
                image.close()
        except HTTPException:
            raise
        except Exception:
            LOGGER.error("Inferensi gambar gagal: internal")
            raise HTTPException(status_code=500, detail="Inferensi gambar gagal") from None
        finally:
            await file.close()
        LOGGER.info("Inferensi selesai dalam %.3f detik", time.perf_counter() - started)
        return result

    return app
