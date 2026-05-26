import logging

from fastapi import APIRouter, Depends, Query

from auth.dependencies import require_maintainer
from schemas.system_log import SystemLogResponse
from services.system_log_service import SystemLogService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_maintainer)])


@router.get("", response_model=list[SystemLogResponse])
def list_system_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    service = SystemLogService()
    return service.list_logs(limit=limit, offset=offset)
