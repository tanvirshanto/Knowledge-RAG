import logging
import re
from collections.abc import Iterator

from app.config import get_settings
from app.services.embeddings import get_embedding_service
from app.services.llm import GeminiLLM
from app.services.vector_store import get_vector_store
from services.system_log_service import SystemLogService

logger = logging.getLogger(__name__)


def _log_pipeline_error(level: str, message: str, exception: Exception) -> None:
    try:
        SystemLogService().log_error(level=level, message=message, exception=exception)
    except Exception:
        logger.warning("Failed to persist pipeline error to system_logs", exc_info=True)


def _extract_figure_mentions(text: str) -> list[str]:
    """Extract figure numbers (e.g., '9.1' from 'Figure 9.1' or 'Fig. 9.1')."""
    return re.findall(r"\b(?:Fig\.?|Figure)\s*(\d+\.\d+)\b", text, re.IGNORECASE)


def retrieve_contexts(question: str) -> list[dict]:
    settings = get_settings()
    try:
        embedder = get_embedding_service(settings)
        logger.info("Embedder initialized successfully.")
        store = get_vector_store(settings)
        logger.info("Vector store initialized successfully.")

        # 1. Primary search
        query_vector = embedder.embed_query(question)
        logger.info("Query vector embedded successfully.")
        contexts = store.search(query_vector, top_k=settings.retrieval_top_k)
        logger.info("Retrieved %d primary contexts", len(contexts))

        # 2. Extract figure mentions and perform secondary search if found
        figure_mentions = _extract_figure_mentions(question)
        secondary_contexts = []
        if figure_mentions:
            logger.info("Detected figure mentions: %s. Performing secondary legend search...", figure_mentions)
            for fig in figure_mentions:
                target_query = f"Legend for Figure {fig}"
                target_vector = embedder.embed_query(target_query)
                fig_contexts = store.search(target_vector, top_k=5)
                logger.info("Retrieved %d contexts for Figure %s", len(fig_contexts), fig)
                secondary_contexts.extend(fig_contexts)

        # 3. Combine and deduplicate contexts based on text content
        seen_texts = set()
        combined_contexts = []

        # Add primary contexts first to preserve their rank/score order
        for ctx in contexts:
            text = ctx.get("text", "")
            if text not in seen_texts:
                seen_texts.add(text)
                combined_contexts.append(ctx)

        for ctx in secondary_contexts:
            text = ctx.get("text", "")
            if text not in seen_texts:
                seen_texts.add(text)
                combined_contexts.append(ctx)

        # 4. Programmatic Legend-Aware Weighting
        # Only apply the boost when query contains figure reference or diagnosis request
        has_fig_ref = len(figure_mentions) > 0
        has_diagnosis_req = any(
            word in question.lower()
            for word in ["diagnosis", "diagnose", "differential diagnosis", "definitive diagnosis"]
        )

        if has_fig_ref or has_diagnosis_req:
            logger.info("Applying legend-aware context boosting (figure_ref=%s, diagnosis_req=%s)", has_fig_ref, has_diagnosis_req)
            boosted = False
            for ctx in combined_contexts:
                text = ctx.get("text", "").lower()
                if "answer guide" in text or "legends for introductory figures" in text:
                    ctx["score"] = ctx.get("score", 0.0) + 0.5
                    boosted = True

            if boosted:
                # Re-sort contexts by the new boosted score
                combined_contexts.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return combined_contexts
    except Exception as exc:
        logger.exception("Retrieval pipeline failed for question: %s", question)
        _log_pipeline_error(
            "ERROR",
            f"Retrieval pipeline failed: {exc}",
            exc,
        )
        raise


def stream_answer_question(question: str) -> Iterator[str]:
    try:
        settings = get_settings()
        contexts = retrieve_contexts(question)
        llm = GeminiLLM(settings)
        yield from llm.stream_answer(question, contexts)
    except Exception as exc:
        logger.exception("Stream-answer failed for question: %s", question)
        _log_pipeline_error(
            "ERROR",
            f"Stream-answer pipeline failed: {exc}",
            exc,
        )
        raise


def answer_question(question: str) -> str:
    try:
        settings = get_settings()
        contexts = retrieve_contexts(question)
        return GeminiLLM(settings).answer(question, contexts)
    except Exception as exc:
        logger.exception("Answer pipeline failed for question: %s", question)
        _log_pipeline_error(
            "ERROR",
            f"Answer pipeline failed: {exc}",
            exc,
        )
        raise
