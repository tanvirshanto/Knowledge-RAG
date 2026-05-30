"""
Module 1: Ingestion Engine
Handles file upload to GCS and triggers Vertex AI RAG corpus import.
"""

import logging
import os
from pathlib import Path

from google.cloud import storage
from google.cloud.aiplatform_v1beta1 import VertexRagDataServiceClient
from google.cloud.aiplatform_v1beta1.types import (
    ImportRagFilesConfig,
    ImportRagFilesRequest,
    RagFileTransformationConfig,
    RagFileChunkingConfig,
    GcsSource,
)

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when ingestion fails."""
    pass


class IngestionEngine:
    """
    Handles PDF ingestion into Vertex AI RAG corpus.
    
    Responsibilities:
    1. Upload local file to GCS bucket
    2. Trigger non-blocking import with semantic chunking
    3. Return LRO ID for status tracking
    """
    
    def __init__(
        self,
        project_id: str,
        location: str,
        corpus_id: str,
        bucket_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.project_id = project_id
        self.location = location
        self.corpus_id = corpus_id
        self.bucket_name = bucket_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize clients
        self._storage_client = storage.Client(project=project_id)
        self._data_client = VertexRagDataServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
    
    def ingest(self, local_file_path: str) -> str:
        """
        Upload file to GCS and trigger RAG corpus import.
        
        Args:
            local_file_path: Path to local PDF file
            
        Returns:
            LRO operation name for tracking (e.g., 'projects/.../operations/abc123')
            
        Raises:
            IngestionError: If file upload or import fails
            FileNotFoundError: If local file doesn't exist
        """
        file_path = Path(local_file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {local_file_path}")
        
        # Step 1: Upload to GCS
        gcs_uri = self._upload_to_gcs(file_path)
        
        # Step 2: Trigger import
        operation_name = self._trigger_import(gcs_uri)
        
        logger.info(f"Ingestion triggered. LRO: {operation_name}")
        return operation_name
    
    def _upload_to_gcs(self, file_path: Path) -> str:
        """Upload file to GCS and return the gs:// URI."""
        blob_name = f"uploads/{file_path.name}"
        
        try:
            bucket = self._storage_client.bucket(self.bucket_name)
            
            # Verify bucket exists
            if not bucket.exists():
                raise IngestionError(f"GCS bucket does not exist: {self.bucket_name}")
            
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"Uploaded {file_path.name} to {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            if isinstance(e, IngestionError):
                raise
            raise IngestionError(f"Failed to upload file to GCS: {e}") from e
    
    def _trigger_import(self, gcs_uri: str) -> str:
        """Trigger Vertex AI RAG import and return operation name."""
        try:
            # Build corpus resource path
            rag_corpus_path = (
                f"projects/{self.project_id}/locations/{self.location}"
                f"/ragCorpora/{self.corpus_id}"
            )
            
            # Configure chunking
            chunking_config = RagFileChunkingConfig(
                fixed_length_chunking=RagFileChunkingConfig.FixedLengthChunking(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            )
            
            # Build transformation config
            transformation_config = RagFileTransformationConfig(
                rag_file_chunking_config=chunking_config,
            )
            
            # Build import request
            request = ImportRagFilesRequest(
                parent=rag_corpus_path,
                import_rag_files_config=ImportRagFilesConfig(
                    gcs_source=GcsSource(uris=[gcs_uri]),
                    rag_file_transformation_config=transformation_config,
                ),
            )
            
            # Trigger non-blocking import
            operation = self._data_client.import_rag_files(request=request)
            
            return operation.operation.name
            
        except Exception as e:
            raise IngestionError(f"Failed to trigger RAG import: {e}") from e
    
    @classmethod
    def from_config(cls, config) -> "IngestionEngine":
        """Create engine from VertexRAGConfig."""
        return cls(
            project_id=config.project_id,
            location=config.location,
            corpus_id=config.corpus_id,
            bucket_name=config.bucket_name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
