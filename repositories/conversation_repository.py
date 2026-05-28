import logging
from typing import Optional

from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ConversationRepository(BaseRepository):
    def __init__(self):
        super().__init__("conversations")

    def create(self, user_id: str, title: str) -> dict:
        return self._execute_with_retry(
            lambda: self.table.insert({
                "user_id": user_id,
                "title": title,
            }).execute(),
            single=True,
        )

    def get_by_id(self, conversation_id: str) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("*").eq("id", conversation_id).execute(),
            single=True,
        )

    def list_by_user(self, user_id: str) -> list[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute(),
        )

    def update_title(self, conversation_id: str, title: str) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.update({"title": title})
            .eq("id", conversation_id)
            .execute(),
            single=True,
        )

    def touch(self, conversation_id: str) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.update({"updated_at": "NOW()"})
            .eq("id", conversation_id)
            .execute(),
            single=True,
        )

    def delete(self, conversation_id: str) -> bool:
        data = self._execute_with_retry(
            lambda: self.table.delete().eq("id", conversation_id).execute(),
        )
        return len(data) > 0
