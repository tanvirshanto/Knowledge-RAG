import hashlib
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings
from app.services.chunking import TextChunk
from utils.retry import retry_sync


class QdrantVectorStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._collection = settings.qdrant_collection_name
        self._ensured = False

    def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        if self._ensured:
            return

        if vector_size is None:
            vector_size = self._settings.embedding_dimension

        collections_response = retry_sync(
            lambda: self._client.get_collections(),
            "qdrant.get_collections",
        )
        names = {c.name for c in collections_response.collections}
        if self._collection not in names:
            retry_sync(
                lambda: self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                ),
                "qdrant.create_collection",
            )
        self._ensured = True

    def upsert_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        *,
        job_id: str,
        filename: str = "document.pdf",
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count must match embedding count.")

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_hash = hashlib.md5(f"{filename}_{i}".encode("utf-8")).hexdigest()
            deterministic_id = str(uuid.UUID(chunk_hash))
            points.append(
                qmodels.PointStruct(
                    id=deterministic_id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "page": chunk.page,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        "subsection": chunk.subsection,
                        "job_id": job_id,
                        "filename": filename,
                    },
                )
            )

        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            retry_sync(
                lambda b=batch: self._client.upsert(
                    collection_name=self._collection,
                    points=b,
                ),
                f"qdrant.upsert(batch_{i // batch_size})",
            )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        *,
        top_k: int,
    ) -> List[dict]:
        response = retry_sync(
            lambda: self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            ),
            "qdrant.query_points",
        )
        hits = response.points
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "page": payload.get("page"),
                    "chapter": payload.get("chapter"),
                    "score": hit.score,
                }
            )
        return results


# --- GLOBAL SINGLETON ACCESSOR ---

_vector_store_instance: QdrantVectorStore | None = None


def get_vector_store(settings: Settings) -> QdrantVectorStore:
    """Returns the persistent vector store. Call this in your routes/pipelines."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = QdrantVectorStore(settings)
    return _vector_store_instance
