"""
Configuration management for Vertex AI RAG pipeline.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VertexRAGConfig:
    """Configuration for Vertex AI RAG operations."""
    
    project_id: str
    location: str
    corpus_id: str
    bucket_name: str
    
    # Optional overrides
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_top_k: int = 5
    
    @classmethod
    def from_env(cls, corpus_id: str, bucket_name: str) -> "VertexRAGConfig":
        """
        Create config from environment variables.
        
        Required env vars:
            GOOGLE_CLOUD_PROJECT: GCP project ID
            GOOGLE_CLOUD_LOCATION: GCP region (e.g., us-central1)
        """
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-east1")
        chunk_size = os.environ.get("CHUNK_SIZE")
        chunk_overlap = os.environ.get("CHUNK_OVERLAP")
        retrieval_top_k = os.environ.get("RETRIEVAL_TOP_K")
        
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
        
        return cls(
            project_id=project_id,
            location=location,
            corpus_id=corpus_id,
            bucket_name=bucket_name,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            similarity_top_k=int(retrieval_top_k),
        )
    
    @property
    def rag_corpus_path(self) -> str:
        """Return fully qualified RAG corpus resource path."""
        return f"projects/{self.project_id}/locations/{self.location}/ragCorpora/{self.corpus_id}"
