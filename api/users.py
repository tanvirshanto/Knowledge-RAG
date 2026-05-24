import logging

from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import get_current_user, require_maintainer
from schemas.user import UserCreate, UserResponse, UserUpdate
from services.user_service import UserService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_maintainer)])


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreate):
    service = UserService()
    try:
        return service.create_user(body.username, body.password, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get("", response_model=list[UserResponse])
def list_users():
    service = UserService()
    return service.list_users()


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: str):
    service = UserService()
    user = service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.patch("/{user_id}", response_model=UserResponse)
def update_user(user_id: str, body: UserUpdate):
    service = UserService()
    try:
        return service.update_user(user_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: str,
    current_user: dict = Depends(get_current_user),
):
    service = UserService()
    target = service.get_user(str(user_id))
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if current_user["sub"] == target.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    deleted = service.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
