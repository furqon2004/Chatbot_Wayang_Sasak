import logging
import time

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .schemas import SearchRequest, SearchResponse
from .semantic import RetrievalUnavailable
from .security import event

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search(payload: SearchRequest, request: Request):
    service = request.app.state.retrieval
    if service is None:
        raise HTTPException(status_code=503, detail="Retrieval belum siap")
    started = time.perf_counter()
    try:
        response = await run_in_threadpool(service.search, payload.query, payload.top_k)
        event(endpoint="/search", mode=response.mode, category="retrieval")
        return response
    except RetrievalUnavailable:
        raise HTTPException(status_code=503, detail="Retrieval belum siap") from None
    except Exception:
        LOGGER.error("Search gagal: internal")
        raise HTTPException(status_code=500, detail="Pencarian gagal") from None
    finally:
        LOGGER.info("Search selesai dalam %.3f detik", time.perf_counter() - started)
