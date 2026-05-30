import logging
from typing import Optional

from repositories.conversation_repository import ConversationRepository
from repositories.message_repository import MessageRepository
from schemas.conversation import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageResponse,
)

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(
        self,
        conversation_repo: Optional[ConversationRepository] = None,
        message_repo: Optional[MessageRepository] = None,
    ):
        self.conversation_repo = conversation_repo or ConversationRepository()
        self.message_repo = message_repo or MessageRepository()

    def create_conversation(self, user_id: str, title: str) -> ConversationResponse:
        record = self.conversation_repo.create(user_id, title)
        return ConversationResponse(
            id=record["id"],
            user_id=record["user_id"],
            title=record["title"],
            created_at=record.get("created_at", ""),
            updated_at=record.get("updated_at", ""),
        )

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[ConversationDetailResponse]:
        record = self.conversation_repo.get_by_id(conversation_id)
        if record is None:
            return None
        if record["user_id"] != user_id:
            logger.warning("User %s attempted to access conversation %s owned by %s", user_id, conversation_id, record["user_id"])
            return None

        message_records = self.message_repo.list_by_conversation(conversation_id, limit=1000)
        messages = [
            MessageResponse(
                id=m["id"],
                conversation_id=m["conversation_id"],
                role=m["role"],
                content=m["content"],
                created_at=m.get("created_at", ""),
            )
            for m in message_records
        ]

        return ConversationDetailResponse(
            id=record["id"],
            user_id=record["user_id"],
            title=record["title"],
            created_at=record.get("created_at", ""),
            updated_at=record.get("updated_at", ""),
            messages=messages,
        )

    def list_conversations(self, user_id: str) -> ConversationListResponse:
        records = self.conversation_repo.list_by_user(user_id)
        conversations = [
            ConversationResponse(
                id=r["id"],
                user_id=r["user_id"],
                title=r["title"],
                created_at=r.get("created_at", ""),
                updated_at=r.get("updated_at", ""),
            )
            for r in records
        ]
        return ConversationListResponse(conversations=conversations)

    def update_title(self, conversation_id: str, user_id: str, title: str) -> Optional[ConversationResponse]:
        record = self.conversation_repo.get_by_id(conversation_id)
        if record is None or record["user_id"] != user_id:
            return None

        updated = self.conversation_repo.update_title(conversation_id, title)
        if updated is None:
            return None

        return ConversationResponse(
            id=updated["id"],
            user_id=updated["user_id"],
            title=updated["title"],
            created_at=updated.get("created_at", ""),
            updated_at=updated.get("updated_at", ""),
        )

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        record = self.conversation_repo.get_by_id(conversation_id)
        if record is None or record["user_id"] != user_id:
            return False
        return self.conversation_repo.delete(conversation_id)

    def save_messages(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        self.message_repo.bulk_create([
            {"conversation_id": conversation_id, "role": "user", "content": user_message},
            {"conversation_id": conversation_id, "role": "assistant", "content": assistant_message},
        ])
        self.conversation_repo.touch(conversation_id)

    def get_history(self, conversation_id: str, max_messages: int) -> list[dict]:
        records = self.message_repo.list_by_conversation(conversation_id, limit=max_messages)
        return [{"role": r["role"], "content": r["content"]} for r in records]
