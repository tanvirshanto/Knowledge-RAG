"""
Module 3: Grounded Retrieval Engine
Generates grounded answers using Vertex AI RAG retrieval.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from google.cloud.aiplatform_v1beta1 import VertexRagServiceClient
from google.cloud.aiplatform_v1beta1.types.vertex_rag_service import (
    RetrieveContextsRequest,
    RagQuery,
)

logger = logging.getLogger(__name__)


@dataclass
class GroundedAnswer:
    """Structured response from grounded retrieval."""
    
    answer: str
    citations: List[dict]
    grounding_chunks: List[dict]
    grounding_supports: List[dict]
    raw_response: object  # Full API response for advanced usage


class RetrievalError(Exception):
    """Raised when retrieval fails."""
    pass


class RetrievalEngine:
    """
    Generates grounded answers using Vertex AI RAG retrieval.
    
    Responsibilities:
    1. Build RAG corpus resource path
    2. Execute grounded content generation
    3. Return synthesized answer with citations
    """
    
    def __init__(
        self,
        project_id: str,
        location: str,
        corpus_id: str,
        similarity_top_k: int = 5,
    ):
        self.project_id = project_id
        self.location = location
        self.corpus_id = corpus_id
        self.similarity_top_k = similarity_top_k
        
        # Initialize client
        self._service_client = VertexRagServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
    
    @property
    def rag_corpus_path(self) -> str:
        """Return fully qualified RAG corpus resource path."""
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/ragCorpora/{self.corpus_id}"
        )
    
    def ask(self, question: str) -> GroundedAnswer:
        """
        Ask a question and get a grounded answer.
        
        Args:
            question: User's question to answer
            
        Returns:
            GroundedAnswer with synthesized text and citations
            
        Raises:
            RetrievalError: If retrieval or generation fails
            ValueError: If question is empty
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            # 1. Retrieve contexts from Vertex RAG
            contexts = self.retrieve_contexts(question)
            
            # 2. Generate answer using GeminiLLM
            from app.config import get_settings
            from app.services.llm import GeminiLLM
            
            settings = get_settings()
            llm = GeminiLLM(settings)
            
            answer_text = llm.answer(question, contexts)
            
            return GroundedAnswer(
                answer=answer_text,
                citations=[],
                grounding_chunks=contexts,
                grounding_supports=[],
                raw_response=None,
            )
            
        except Exception as e:
            if isinstance(e, RetrievalError):
                raise
            raise RetrievalError(f"Failed to generate grounded answer: {e}") from e
    
    def retrieve_contexts(self, question: str, top_k: Optional[int] = None) -> List[dict]:
        """
        Retrieve relevant contexts without generation.
        
        Args:
            question: Query to search for
            top_k: Number of results (overrides default)
            
        Returns:
            List of context dictionaries with text and metadata
        """
        k = top_k or self.similarity_top_k
        
        try:
            request = RetrieveContextsRequest(
                parent=f"projects/{self.project_id}/locations/{self.location}",
                vertex_rag_store=RetrieveContextsRequest.VertexRagStore(
                    rag_resources=[
                        RetrieveContextsRequest.VertexRagStore.RagResource(
                            rag_corpus=self.rag_corpus_path,
                        )
                    ]
                ),
                query=RagQuery(
                    text=question,
                    similarity_top_k=k,
                ),
            )
            
            response = self._service_client.retrieve_contexts(request=request)
            
            contexts = []
            for ctx in response.contexts.contexts:
                context_dict = {
                    "text": ctx.text,
                    "source_uri": ctx.source_uri,
                    "relevance_score": getattr(ctx, "score", 0.0),
                }
                
                # Extract page info if present in chunk page_span
                chunk = getattr(ctx, "chunk", None)
                if chunk:
                    page_span = getattr(chunk, "page_span", None)
                    if page_span:
                        first_page = getattr(page_span, "first_page", None)
                        if first_page is not None:
                            context_dict["page"] = first_page
                
                # Extract chapter from source display name or URI
                source_name = getattr(ctx, "source_display_name", "") or ctx.source_uri or ""
                import re
                chapter_match = re.search(r"chapter\s*(\d+)", source_name, re.IGNORECASE)
                if chapter_match:
                    context_dict["chapter"] = chapter_match.group(1)
                
                contexts.append(context_dict)
            
            return contexts
            
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve contexts: {e}") from e
    
    @classmethod
    def from_config(cls, config) -> "RetrievalEngine":
        """Create engine from VertexRAGConfig."""
        return cls(
            project_id=config.project_id,
            location=config.location,
            corpus_id=config.corpus_id,
            similarity_top_k=config.similarity_top_k,
        )
