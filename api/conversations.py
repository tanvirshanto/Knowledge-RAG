from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import get_current_user
from schemas.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
)
from repositories.user_repository import UserRepository
from services.conversation_service import ConversationService

router = APIRouter()


def get_conversation_service() -> ConversationService:
    return ConversationService()


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """List all conversations for the current user."""
    user_repo = UserRepository()
    user_record = user_repo.get_by_username(current_user["sub"])
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user_record["id"]
    return service.list_conversations(user_id)


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    request: ConversationCreate,
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Create a new conversation."""
    user_repo = UserRepository()
    user_record = user_repo.get_by_username(current_user["sub"])
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user_record["id"]
    title = request.title or "New Conversation"
    return service.create_conversation(user_id, title)


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Get a conversation with all its messages."""
    user_repo = UserRepository()
    user_record = user_repo.get_by_username(current_user["sub"])
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user_record["id"]
    conversation = service.get_conversation(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Update a conversation's title."""
    user_repo = UserRepository()
    user_record = user_repo.get_by_username(current_user["sub"])
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user_record["id"]
    conversation = service.update_title(conversation_id, user_id, request.title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Delete a conversation and all its messages."""
    user_repo = UserRepository()
    user_record = user_repo.get_by_username(current_user["sub"])
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user_record["id"]
    success = service.delete_conversation(conversation_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return None
