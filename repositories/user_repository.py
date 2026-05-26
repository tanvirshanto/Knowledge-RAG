import logging
from typing import Optional

from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository):
    def __init__(self):
        super().__init__("users")

    def create(self, username: str, password_hash: str, role: str) -> dict:
        return self._execute_with_retry(
            lambda: self.table.insert({
                "username": username,
                "password_hash": password_hash,
                "role": role,
            }).execute(),
            single=True,
        )

    def get_by_id(self, user_id: str) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("*").eq("id", user_id).execute(),
            single=True,
        )

    def get_by_username(self, username: str) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("*").eq("username", username).eq("is_active", True).execute(),
            single=True,
        )

    def list_all(self) -> list[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("id,username,role,is_active,created_at").order("created_at", desc=True).execute(),
        )

    def update(self, user_id: str, **fields) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.update(fields).eq("id", user_id).execute(),
            single=True,
        )

    def delete(self, user_id: str) -> bool:
        data = self._execute_with_retry(
            lambda: self.table.delete().eq("id", user_id).execute(),
        )
        return len(data) > 0

    def exists_by_username(self, username: str) -> bool:
        data = self._execute_with_retry(
            lambda: self.table.select("id").eq("username", username).execute(),
        )
        return len(data) > 0

    def table_exists(self) -> bool:
        try:
            self._execute_with_retry(
                lambda: self.table.select("id").limit(1).execute(),
            )
            return True
        except Exception:
            return False
