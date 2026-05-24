import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.models import AskRequest, AskResponse
from app.pipelines.retrieval import answer_question, stream_answer_question
from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("")
def ask(
    body: AskRequest,
    stream: bool = Query(default=True, description="Stream tokens via SSE"),
    current_user: dict = Depends(get_current_user),
):
    if stream:
        def event_generator():
            try:
                yield _sse_event({"type": "start", "question": body.question})
                for token in stream_answer_question(body.question):
                    yield _sse_event({"type": "token", "content": token})
                yield _sse_event({"type": "done", "question": body.question})
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
        answer = answer_question(body.question)
    except Exception as exc:
        logger.exception("Ask pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return AskResponse(question=body.question, answer=answer)
