from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "Queued"
    PARSING = "Parsing..."
    CHUNKING = "Chunking..."
    EMBEDDING = "Embedding..."
    INDEXING = "Indexing..."
    COMPLETED = "Completed"
    FAILED = "Failed"


class UploadResponse(BaseModel):
    job_id: str
    status: str = "Queued"


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    detail: Optional[str] = None
    chunks_indexed: Optional[int] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
