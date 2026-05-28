import logging
from typing import Optional

from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class MessageRepository(BaseRepository):
    def __init__(self):
        super().__init__("messages")

    def create(self, conversation_id: str, role: str, content: str) -> dict:
        return self._execute_with_retry(
            lambda: self.table.insert({
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
            }).execute(),
            single=True,
        )

    def list_by_conversation(self, conversation_id: str, limit: int) -> list[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute(),
        )

    def bulk_create(self, messages: list[dict]) -> list[dict]:
        return self._execute_with_retry(
            lambda: self.table.insert(messages).execute(),
        )
