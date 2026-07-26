"""
Composition root for the AI Medical Assistant.

Binds ports to adapters in one place. The LLM and the retriever are
process-wide singletons (both are expensive to construct); repositories and use
cases are request-scoped because they carry the database session.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user, get_db
from app.assistant.application.ports import (
    AssistantLLMPort,
    AssistantPipelinePort,
    ConversationRepositoryPort,
    GuardrailsPort,
    KnowledgeRetrievalPort,
)
from app.assistant.application.use_cases import (
    DeleteConversationUseCase,
    GetConversationUseCase,
    ListConversationsUseCase,
    SendMessageUseCase,
)
from app.assistant.config import get_assistant_settings
from app.assistant.infrastructure.guardrails import LLMGuardrails
from app.assistant.infrastructure.llm import AssistantGroqAdapter
from app.assistant.infrastructure.repositories import SqlConversationRepository
from app.assistant.infrastructure.retrieval import (
    NullKnowledgeRetriever,
    QdrantKnowledgeRetriever,
)
from app.assistant.pipeline.graph import LangGraphAssistantPipeline
from app.models.user import User

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_assistant_llm() -> AssistantLLMPort:
    """Process-wide LLM, credentialed from `.env.ai-assistant`."""
    logger.info("[ASSISTANT_DI] initialising Groq adapter (isolated env)")
    return AssistantGroqAdapter()


@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeRetrievalPort:
    """
    Process-wide knowledge retriever.

    Returns the null retriever when retrieval is disabled, so the pipeline
    always has a valid collaborator and never branches on None.
    """
    settings = get_assistant_settings()
    if not settings.enable_retrieval:
        return NullKnowledgeRetriever()
    return QdrantKnowledgeRetriever(settings)


def get_guardrails(
    llm: AssistantLLMPort = Depends(get_assistant_llm),
) -> GuardrailsPort:
    return LLMGuardrails(llm=llm)


def get_pipeline(
    llm: AssistantLLMPort = Depends(get_assistant_llm),
    retriever: KnowledgeRetrievalPort = Depends(get_retriever),
) -> AssistantPipelinePort:
    """Cheap to build: the compiled graph itself is cached by `llm` identity."""
    return LangGraphAssistantPipeline(llm=llm, retriever=retriever)


def get_conversation_repository(
    db: AsyncSession = Depends(get_db),
) -> ConversationRepositoryPort:
    return SqlConversationRepository(db)


def get_send_message_use_case(
    conversations: ConversationRepositoryPort = Depends(get_conversation_repository),
    pipeline: AssistantPipelinePort = Depends(get_pipeline),
    guardrails: GuardrailsPort = Depends(get_guardrails),
) -> SendMessageUseCase:
    return SendMessageUseCase(
        conversations=conversations, pipeline=pipeline, guardrails=guardrails
    )


def get_conversation_use_case(
    conversations: ConversationRepositoryPort = Depends(get_conversation_repository),
) -> GetConversationUseCase:
    return GetConversationUseCase(conversations=conversations)


def get_list_conversations_use_case(
    conversations: ConversationRepositoryPort = Depends(get_conversation_repository),
) -> ListConversationsUseCase:
    return ListConversationsUseCase(conversations=conversations)


def get_delete_conversation_use_case(
    conversations: ConversationRepositoryPort = Depends(get_conversation_repository),
) -> DeleteConversationUseCase:
    return DeleteConversationUseCase(conversations=conversations)


def get_patient_user_id(
    current_user: User = Depends(get_current_active_user),
) -> str:
    """The authenticated caller's id, used as the conversation owner."""
    return str(current_user.id)
