"""
Module 2: Status Verification Poller
Polls Long-Running Operation status until completion.
"""

import logging
import time
from typing import Optional

from google.api_core import operations_v1
from google.cloud.aiplatform_v1beta1 import VertexRagDataServiceClient

logger = logging.getLogger(__name__)


class PollerError(Exception):
    """Raised when polling encounters an error."""
    pass


class OperationFailedError(PollerError):
    """Raised when the operation fails."""
    
    def __init__(self, message: str, error_code: Optional[int] = None, error_message: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.error_message = error_message


class StatusPoller:
    """
    Polls Vertex AI Long-Running Operation until completion.
    
    Responsibilities:
    1. Periodically check operation status
    2. Log progress updates
    3. Handle failures with descriptive exceptions
    4. Return True when operation completes successfully
    """
    
    def __init__(self, location: str, poll_interval_seconds: int = 10):
        self.location = location
        self.poll_interval_seconds = poll_interval_seconds
        
        # Initialize client
        self._data_client = VertexRagDataServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
    
    def poll(self, operation_name: str) -> bool:
        """
        Poll operation until completion.
        
        Args:
            operation_name: Full LRO name (e.g., 'projects/.../operations/abc123')
            
        Returns:
            True when operation completes successfully
            
        Raises:
            OperationFailedError: If operation fails
            PollerError: If polling encounters unexpected errors
        """
        if not operation_name:
            raise ValueError("operation_name is required")
        
        logger.info(f"Starting poll for operation: {operation_name}")
        
        attempt = 0
        while True:
            attempt += 1
            
            try:
                status = self._check_status(operation_name)
                
                if status.done:
                    return self._handle_completion(status, operation_name)
                
                self._log_progress(attempt, operation_name)
                time.sleep(self.poll_interval_seconds)
                
            except Exception as e:
                if isinstance(e, (OperationFailedError, PollerError)):
                    raise
                raise PollerError(f"Error polling operation: {e}") from e
    
    def _check_status(self, operation_name: str):
        """Check current operation status."""
        try:
            # Use the operations client to get status
            # The operation name format is: projects/{project}/locations/{location}/operations/{id}
            return self._data_client.transport.operations_client.get_operation(
                name=operation_name
            )
        except Exception as e:
            raise PollerError(f"Failed to get operation status: {e}") from e
    
    def _handle_completion(self, status, operation_name: str) -> bool:
        """Handle operation completion - check for errors."""
        if status.error and status.error.code != 0:
            error_code = status.error.code
            error_message = status.error.message
            
            logger.error(
                f"Operation failed: code={error_code}, message={error_message}"
            )
            
            raise OperationFailedError(
                f"Operation {operation_name} failed with code {error_code}: {error_message}",
                error_code=error_code,
                error_message=error_message,
            )
        
        logger.info(f"Operation completed successfully: {operation_name}")
        return True
    
    def _log_progress(self, attempt: int, operation_name: str):
        """Log polling progress."""
        logger.info(
            f"Polling attempt {attempt} - Operation still in progress. "
            f"Next check in {self.poll_interval_seconds}s"
        )
    
    @classmethod
    def from_config(cls, config, poll_interval_seconds: int = 10) -> "StatusPoller":
        """Create poller from VertexRAGConfig."""
        return cls(
            location=config.location,
            poll_interval_seconds=poll_interval_seconds,
        )
