from fastapi import APIRouter

from api.auth import router as auth_router
from api.ask import router as ask_router
from api.users import router as users_router
from api.uploads import router as uploads_router
from api.system_logs import router as system_logs_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(ask_router, prefix="/ask", tags=["ask"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(uploads_router, prefix="/uploads", tags=["uploads"])
api_router.include_router(system_logs_router, prefix="/system-logs", tags=["system-logs"])

