from app.pipelines.ingestion import logger
from collections.abc import Iterator

from app.config import get_settings
from app.services.embeddings import get_embedding_service
from app.services.llm import GeminiLLM
from app.services.vector_store import get_vector_store


def retrieve_contexts(question: str) -> list[dict]:
    settings = get_settings()
    embedder = get_embedding_service(settings)
    logger.info("Embedder initialized successfully.")
    store = get_vector_store(settings)
    logger.info("Vector store initialized successfully.")
    query_vector = embedder.embed_query(question)
    logger.info("Query vector embedded successfully.")
    contexts = store.search(query_vector, top_k=settings.retrieval_top_k)
    logger.info("Retrieved %d contexts", len(contexts))
    return contexts
    


def stream_answer_question(question: str) -> Iterator[str]:
    settings = get_settings()
    contexts = retrieve_contexts(question)
    llm = GeminiLLM(settings)
    yield from llm.stream_answer(question, contexts)


def answer_question(question: str) -> str:
    settings = get_settings()
    contexts = retrieve_contexts(question)
    return GeminiLLM(settings).answer(question, contexts)
