import logging

from fastapi import APIRouter, HTTPException, Request

from .chat_schemas import ChatRequest, ChatResponse
from .semantic import RetrievalUnavailable
from .security import event

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(payload: ChatRequest, request: Request):
    service = request.app.state.chat
    if service is None:
        raise HTTPException(status_code=503, detail="Layanan chat belum siap")
    try:
        response = await service.chat(payload.message, awaiting_more=payload.awaiting_more,
                                      seen_ids=payload.seen_ids, character_id=payload.character_id, explain_more=payload.explain_more,
                                      conversation_context=payload.conversation_context)
        event(endpoint="/chat", mode=response.mode, category=response.llm_status, llm_used=response.llm_used)
        return response
    except RetrievalUnavailable:
        raise HTTPException(status_code=503, detail="Retrieval belum siap") from None
    except Exception:
        LOGGER.error("Chat gagal: internal")
        raise HTTPException(status_code=500, detail="Chat gagal") from None
