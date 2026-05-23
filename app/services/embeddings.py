import logging
from typing import List, Literal

from sentence_transformers import SentenceTransformer
from app.config import Settings

logger = logging.getLogger(__name__)

InputType = Literal["query", "document"]

class LocalEmbeddingService:
    """Local BGE-M3 embeddings (Singleton pattern). Adaptive to host hardware."""

    # Class-level variable keeps the model in RAM across all requests
    _model_instance: SentenceTransformer | None = None

    def __init__(self, settings: Settings):
        self._settings = settings
        # Warm up the model during service init
        self._get_model()

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    def _get_model(self) -> SentenceTransformer:
        if LocalEmbeddingService._model_instance is None:
            logger.info("Cold start: Loading %s into memory...", self.model_name)
            # We don't specify thread counts; PyTorch will auto-detect the host CPU
            LocalEmbeddingService._model_instance = SentenceTransformer(
                self.model_name, 
                device="cpu"
            )
            logger.info("Model loaded and ready.")
        return LocalEmbeddingService._model_instance

    def embed(
        self,
        texts: List[str],
        *,
        input_type: InputType,
    ) -> List[List[float]]:
        _ = input_type 
        if not texts:
            return []

        model = self._get_model()
        
        # We pass the full list and batch_size to the library.
        # It handles the batching logic internally much more efficiently than a for-loop.
        vectors = model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return vectors.tolist()

    def embed_query(self, question: str) -> List[float]:
        # Single query embedding - will be fast after the first load
        vectors = self.embed([question], input_type="query")
        return vectors[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts, input_type="document")


# --- GLOBAL SINGLETON ACCESSOR ---

_embedding_service_instance: LocalEmbeddingService | None = None

def get_embedding_service(settings: Settings) -> LocalEmbeddingService:
    """Returns the persistent embedding service. Call this in your routes/pipelines."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = LocalEmbeddingService(settings)
    return _embedding_service_instance