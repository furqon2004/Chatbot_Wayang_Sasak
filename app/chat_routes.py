import logging

from fastapi import APIRouter, HTTPException, Request

from .chat_schemas import ChatRequest, ChatResponse
from .semantic import RetrievalUnavailable

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(payload: ChatRequest, request: Request):
    service = request.app.state.chat
    if service is None:
        raise HTTPException(status_code=503, detail="Layanan chat belum siap")
    try:
        return await service.chat(payload.message)
    except RetrievalUnavailable:
        raise HTTPException(status_code=503, detail="Retrieval belum siap") from None
    except Exception:
        LOGGER.error("Chat gagal: internal")
        raise HTTPException(status_code=500, detail="Chat gagal") from None
