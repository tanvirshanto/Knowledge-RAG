import logging

from fastapi import APIRouter, Depends, HTTPException, status

from auth.jwt import create_access_token
from schemas.auth import LoginRequest
from services.user_service import UserService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login")
def login(body: LoginRequest):
    service = UserService()
    user = service.authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )

    return {
        "access_token": token,
        "token_type": "bearer",
    }
