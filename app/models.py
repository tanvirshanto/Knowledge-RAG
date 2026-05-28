from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class UploadResponse(BaseModel):
    job_id: str
    status: str = "QUEUED"


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    detail: Optional[str] = None
    chunks_indexed: Optional[int] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
