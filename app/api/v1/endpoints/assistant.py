"""
AI Medical Assistant endpoints.

Thin async controllers: validate, delegate to one use case, map the result.
Domain errors propagate to the handlers already registered in
`app/middleware/exceptions.py`, so no new error handling is introduced.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import RoleChecker, get_current_active_user
from app.assistant.application.ports import AssistantLLMPort, KnowledgeRetrievalPort
from app.assistant.application.use_cases import (
    DeleteConversationUseCase,
    GetConversationUseCase,
    ListConversationsUseCase,
    SendMessageUseCase,
)
from app.assistant.config import get_assistant_settings
from app.assistant.dependencies import (
    get_assistant_llm,
    get_conversation_use_case,
    get_delete_conversation_use_case,
    get_list_conversations_use_case,
    get_patient_user_id,
    get_retriever,
    get_send_message_use_case,
)
from app.schemas.assistant_api import (
    AssistantHealthResponse,
    ConversationDetailResponse,
    ConversationSummaryResponse,
    SendMessageRequest,
    SendMessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(RoleChecker(["patient"]))])
"""Patient-facing assistant routes."""

monitor_router = APIRouter()
"""Health route, available to any authenticated user."""


@router.post(
    "/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send a message to the AI medical assistant",
)
async def send_message(
    request: SendMessageRequest,
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: SendMessageUseCase = Depends(get_send_message_use_case),
) -> Any:
    """
    Send one patient message and receive a structured reply.

    Detects the patient's language and answers in it, carries conversation
    memory forward, and returns the `structured_data` payload the existing
    response cards render. Omit `conversation_id` to begin a new thread.
    """
    conversation, message = await use_case.execute(
        patient_user_id=patient_user_id,
        text=request.message,
        conversation_id=request.conversation_id,
    )
    return SendMessageResponse.build(conversation, message)


@router.get(
    "/conversations",
    response_model=list[ConversationSummaryResponse],
    summary="List the patient's assistant conversations",
)
async def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: ListConversationsUseCase = Depends(get_list_conversations_use_case),
) -> Any:
    """Backs the existing chat-history drawer."""
    conversations = await use_case.execute(patient_user_id, limit=limit)
    return [ConversationSummaryResponse.build(c) for c in conversations]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Load one conversation with its full transcript",
)
async def get_conversation(
    conversation_id: str,
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: GetConversationUseCase = Depends(get_conversation_use_case),
) -> Any:
    """Replays a stored conversation, including each turn's rendered cards."""
    conversation = await use_case.execute(conversation_id, patient_user_id)
    return ConversationDetailResponse.build(conversation)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: str,
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: DeleteConversationUseCase = Depends(get_delete_conversation_use_case),
) -> Any:
    """Soft-deletes the thread; the record is retained for audit."""
    await use_case.execute(conversation_id, patient_user_id)
    return {"message": "Conversation deleted.", "conversation_id": conversation_id}


@monitor_router.get(
    "/health",
    response_model=AssistantHealthResponse,
    summary="AI assistant dependency health",
)
async def assistant_health(
    llm: AssistantLLMPort = Depends(get_assistant_llm),
    retriever: KnowledgeRetrievalPort = Depends(get_retriever),
    _=Depends(get_current_active_user),
) -> Any:
    """Reports model reachability and the isolated environment's state."""
    settings = get_assistant_settings()
    llm_health = await llm.health()
    healthy = llm_health.get("status") == "healthy"

    return AssistantHealthResponse(
        status="healthy" if healthy else "degraded",
        llm=llm_health,
        environment=settings.describe(),
        retrieval_available=retriever.is_available,
        guardrails_enabled=settings.enable_guardrails,
    )
