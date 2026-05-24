import logging
from typing import Optional

from auth.security import hash_password, verify_password
from repositories.user_repository import UserRepository
from schemas.user import UserCreate, UserResponse, UserUpdate

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self, user_repo: UserRepository | None = None):
        self.user_repo = user_repo or UserRepository()

    def create_user(self, username: str, password: str, role: str) -> UserResponse:
        if self.user_repo.exists_by_username(username):
            raise ValueError(f"Username '{username}' already exists")

        pw_hash = hash_password(password)
        record = self.user_repo.create(username, pw_hash, role)
        return UserResponse(
            id=record["id"],
            username=record["username"],
            role=record["role"],
            is_active=record.get("is_active", True),
            created_at=record.get("created_at", ""),
        )

    def get_user(self, user_id: str) -> Optional[UserResponse]:
        record = self.user_repo.get_by_id(user_id)
        if record is None:
            return None
        return UserResponse(
            id=record["id"],
            username=record["username"],
            role=record["role"],
            is_active=record.get("is_active", True),
            created_at=record.get("created_at", ""),
        )

    def list_users(self) -> list[UserResponse]:
        records = self.user_repo.list_all()
        return [
            UserResponse(
                id=r["id"],
                username=r["username"],
                role=r["role"],
                is_active=r.get("is_active", True),
                created_at=r.get("created_at", ""),
            )
            for r in records
        ]

    def update_user(self, user_id: str, update_data: UserUpdate) -> UserResponse:
        fields = {}
        if update_data.role is not None:
            fields["role"] = update_data.role
        if update_data.is_active is not None:
            fields["is_active"] = update_data.is_active
        if update_data.password is not None:
            fields["password_hash"] = hash_password(update_data.password)

        record = self.user_repo.update(user_id, **fields)
        if record is None:
            raise ValueError(f"User '{user_id}' not found")

        return UserResponse(
            id=record["id"],
            username=record["username"],
            role=record["role"],
            is_active=record.get("is_active", True),
            created_at=record.get("created_at", ""),
        )

    def delete_user(self, user_id: str) -> bool:
        return self.user_repo.delete(user_id)

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        record = self.user_repo.get_by_username(username)
        if record is None:
            return None
        if not verify_password(password, record["password_hash"]):
            return None
        return record
