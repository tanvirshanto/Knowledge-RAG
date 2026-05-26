from typing import Optional

from pydantic import BaseModel


class SystemLogResponse(BaseModel):
    id: str
    level: str
    message: str
    traceback: Optional[str] = None
    created_at: str
