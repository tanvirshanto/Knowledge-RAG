"""
Example: Complete Vertex AI RAG Pipeline
Demonstrates sequential usage of all three modules.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vertex_rag.config import VertexRAGConfig
from vertex_rag.ingestion import IngestionEngine, IngestionError
from vertex_rag.poller import StatusPoller, PollerError, OperationFailedError
from vertex_rag.retrieval import RetrievalEngine, RetrievalError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_path: str,
    corpus_id: str,
    bucket_name: str,
    question: str,
) -> None:
    """
    Execute complete RAG pipeline: Ingest → Poll → Ask.
    
    Args:
        pdf_path: Path to local PDF file
        corpus_id: Vertex AI RAG corpus ID
        bucket_name: GCS bucket name for uploads
        question: Question to ask after ingestion
    """
    # Load config from environment
    config = VertexRAGConfig.from_env(
        corpus_id=corpus_id,
        bucket_name=bucket_name,
    )
    
    logger.info("=" * 60)
    logger.info("STEP 1: INGESTION")
    logger.info("=" * 60)
    
    # Module 1: Ingest
    ingestion_engine = IngestionEngine.from_config(config)
    
    try:
        operation_name = ingestion_engine.ingest(pdf_path)
        logger.info(f"Ingestion triggered. LRO: {operation_name}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return
    except IngestionError as e:
        logger.error(f"Ingestion failed: {e}")
        return
    
    logger.info("=" * 60)
    logger.info("STEP 2: POLLING")
    logger.info("=" * 60)
    
    # Module 2: Poll for completion
    poller = StatusPoller.from_config(config, poll_interval_seconds=10)
    
    try:
        success = poller.poll(operation_name)
        logger.info(f"Operation completed: {success}")
    except OperationFailedError as e:
        logger.error(f"Operation failed - Code: {e.error_code}, Message: {e.error_message}")
        return
    except PollerError as e:
        logger.error(f"Polling error: {e}")
        return
    
    logger.info("=" * 60)
    logger.info("STEP 3: ASK QUESTION")
    logger.info("=" * 60)
    
    # Module 3: Ask grounded question
    retrieval_engine = RetrievalEngine.from_config(config)
    
    try:
        answer = retrieval_engine.ask(question)
        
        logger.info(f"\nQuestion: {question}")
        logger.info(f"\nAnswer: {answer.answer}")
        
        if answer.citations:
            logger.info("\nCitations:")
            for i, citation in enumerate(answer.citations, 1):
                logger.info(f"  {i}. {citation}")
        
        if answer.grounding_chunks:
            logger.info("\nGrounding Sources:")
            for i, chunk in enumerate(answer.grounding_chunks, 1):
                logger.info(f"  {i}. [{chunk['relevance_score']:.2f}] {chunk['source_uri']}")
                
    except RetrievalError as e:
        logger.error(f"Retrieval failed: {e}")
        return
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


def main():
    """Entry point with example configuration."""
    
    # Example configuration - replace with your values
    pdf_path = "path/to/your/textbook.pdf"
    corpus_id = "your-corpus-id"
    bucket_name = "your-gcs-bucket"
    question = "What are the key findings in Chapter 3?"
    
    # Validate environment variables
    required_vars = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.info("Set them with: export GOOGLE_CLOUD_PROJECT=your-project-id")
        return
    
    run_pipeline(pdf_path, corpus_id, bucket_name, question)


if __name__ == "__main__":
    main()
