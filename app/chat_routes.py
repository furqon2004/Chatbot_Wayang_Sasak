import logging

from fastapi import APIRouter, HTTPException, Request

from .chat_schemas import ChatRequest, ChatResponse
from .semantic import RetrievalUnavailable
from .security import IdempotencyConflict, IdempotencyUnavailable, event

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(payload: ChatRequest, request: Request):
    service = request.app.state.chat
    if service is None:
        raise HTTPException(status_code=503, detail="Layanan chat belum siap")
    try:
        async def process():
            response = await service.chat(payload.message, awaiting_more=payload.awaiting_more,
                seen_ids=payload.seen_ids, character_id=payload.character_id, explain_more=payload.explain_more,
                conversation_context=payload.conversation_context)
            event(endpoint="/chat", mode=response.mode, category=response.llm_status, llm_used=response.llm_used)
            return response
        if payload.message_id:
            response = await request.app.state.idempotency.run(payload.message_id,
                {"endpoint": "/chat", **payload.model_dump(mode="json", exclude={"message_id"})}, process)
        else:
            response = await process()
        return response
    except IdempotencyConflict:
        raise HTTPException(status_code=409, detail="ID pesan sudah digunakan untuk permintaan lain") from None
    except IdempotencyUnavailable:
        raise HTTPException(status_code=503, detail="Layanan sedang sibuk") from None
    except RetrievalUnavailable:
        raise HTTPException(status_code=503, detail="Retrieval belum siap") from None
    except Exception:
        LOGGER.error("Chat gagal: internal")
        raise HTTPException(status_code=500, detail="Chat gagal") from None
