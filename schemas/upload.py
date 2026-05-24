from typing import Optional

from pydantic import BaseModel


class UploadJobResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    uploaded_by: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = None


class UploadBulkResponse(BaseModel):
    jobs: list[UploadJobResponse]


class UploadListResponse(BaseModel):
    jobs: list[UploadJobResponse]
    total: int
