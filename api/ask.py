import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models import AskRequest, AskResponse
from app.pipelines.retrieval import answer_question, stream_answer_question
from auth.dependencies import get_current_user
from repositories.user_repository import UserRepository
from services.conversation_service import ConversationService

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _get_vertex_answer(question: str) -> str:
    """Get answer using Vertex AI RAG engine."""
    from vertex_rag.config import VertexRAGConfig
    from vertex_rag.retrieval import RetrievalEngine
    logger.info("Getting answer using Vertex AI RAG engine")
    settings = get_settings()
    config = VertexRAGConfig.from_env(
        corpus_id=settings.vertex_rag_corpus_id,
        bucket_name=settings.google_cloud_bucket,
    )
    engine = RetrievalEngine.from_config(config)
    result = engine.ask(question)
    return result.answer


def _get_vertex_answer_stream(question: str):
    """Get streaming answer using Vertex AI RAG engine (yields full response)."""
    from vertex_rag.config import VertexRAGConfig
    from vertex_rag.retrieval import RetrievalEngine

    settings = get_settings()
    config = VertexRAGConfig.from_env(
        corpus_id=settings.vertex_rag_corpus_id,
        bucket_name=settings.google_cloud_bucket,
    )
    engine = RetrievalEngine.from_config(config)
    result = engine.ask(question)
    # Vertex API doesn't support token streaming, yield full response
    yield result.answer


@router.post("")
def ask(
    body: AskRequest,
    stream: bool = Query(default=True, description="Stream tokens via SSE"),
    current_user: dict = Depends(get_current_user),
):
    settings = get_settings()
    user_repo = UserRepository()
    conversation_service = ConversationService()
    
    # Look up user_id from username
    user_record = user_repo.get_by_username(current_user["sub"])
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user_record["id"]
    
    # Handle conversation
    conversation_id = body.conversation_id
    if conversation_id:
        # Verify conversation exists and belongs to user
        conversation = conversation_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        # Load history
        max_messages = settings.chat_history_max_turns * 2
        history = conversation_service.get_history(conversation_id, max_messages)
    else:
        # Create new conversation with question as title
        title = body.question[:50] + "..." if len(body.question) > 50 else body.question
        conversation = conversation_service.create_conversation(user_id, title)
        conversation_id = conversation.id
        history = []
    
    # Select answer function based on RAG engine type
    logger.info(f"RAG engine type: {settings.rag_engine_type}")
    if settings.is_vertex_engine:
        answer_fn = _get_vertex_answer
        stream_fn = _get_vertex_answer_stream
    else:
        answer_fn = lambda q: answer_question(q, history)
        stream_fn = lambda q: stream_answer_question(q, history)
    
    if stream:
        def event_generator():
            try:
                yield _sse_event({
                    "type": "start",
                    "question": body.question,
                    "conversation_id": conversation_id,
                })
                
                # Collect full response for saving
                full_response = []
                for token in stream_fn(body.question):
                    full_response.append(token)
                    yield _sse_event({"type": "token", "content": token})
                
                # Save messages after streaming completes
                full_response_text = "".join(full_response)
                conversation_service.save_messages(
                    conversation_id,
                    body.question,
                    full_response_text
                )
                
                yield _sse_event({
                    "type": "done",
                    "question": body.question,
                    "conversation_id": conversation_id,
                })
            except Exception as exc:
                logger.exception("Ask stream failed")
                yield _sse_event({"type": "error", "detail": str(exc)})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        answer = answer_fn(body.question)
        # Save messages for non-streaming response
        conversation_service.save_messages(
            conversation_id,
            body.question,
            answer
        )
    except Exception as exc:
        logger.exception("Ask pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return AskResponse(question=body.question, answer=answer)
