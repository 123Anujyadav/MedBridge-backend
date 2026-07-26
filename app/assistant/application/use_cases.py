"""
Use cases for the AI Medical Assistant.

Each depends only on ports. They own conversation lifecycle, authorisation and
persistence sequencing; all AI reasoning is delegated to `AssistantPipelinePort`.
"""

from __future__ import annotations

import logging

from app.assistant.application.ports import (
    AssistantPipelinePort,
    ConversationRepositoryPort,
    GuardrailsPort,
)
from app.assistant.domain.entities import AssistantAnswer, ChatMessage, Conversation
from app.assistant.domain.enums import UrgencyLevel
from app.core.exceptions import AuthorizationException, BusinessRuleValidationException
from app.intake.domain.errors import SessionNotFoundError

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000
MAX_MESSAGES_PER_CONVERSATION = 200
"""
Ceiling on a single thread.

Prevents an unbounded transcript from growing the prompt (and cost) without
limit; the UI starts a new conversation past this point.
"""


def _require_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise BusinessRuleValidationException("Message text cannot be empty.")
    if len(cleaned) > MAX_MESSAGE_CHARS:
        raise BusinessRuleValidationException(
            f"Message exceeds the maximum of {MAX_MESSAGE_CHARS} characters."
        )
    return cleaned


class SendMessageUseCase:
    """
    Handle one patient message end to end.

    Order matters: input guardrails run before the model sees the text, the
    pipeline produces the answer, output guardrails revise it, and only then is
    anything persisted. A blocked message is still recorded so the conversation
    reads coherently.
    """

    def __init__(
        self,
        *,
        conversations: ConversationRepositoryPort,
        pipeline: AssistantPipelinePort,
        guardrails: GuardrailsPort,
    ) -> None:
        self._conversations = conversations
        self._pipeline = pipeline
        self._guardrails = guardrails

    async def execute(
        self,
        *,
        patient_user_id: str,
        text: str,
        conversation_id: str | None = None,
    ) -> tuple[Conversation, ChatMessage]:
        message_text = _require_text(text)

        conversation = await self._load_or_create(conversation_id, patient_user_id)

        if len(conversation.messages) >= MAX_MESSAGES_PER_CONVERSATION:
            raise BusinessRuleValidationException(
                "This conversation has reached its maximum length. "
                "Please start a new one."
            )

        conversation.add_user_message(message_text)

        allowed, guard_message = await self._guardrails.check_input(message_text)
        if not allowed:
            logger.warning(
                "[ASSISTANT_INPUT_BLOCKED] conversation=%s", conversation.conversation_id
            )
            answer = AssistantAnswer(
                reply_text=guard_message,
                summary=guard_message,
                conversation_title=conversation.title,
                urgency=None,
            )
            ai_message = conversation.add_ai_message(answer)
            await self._conversations.save(conversation)
            return conversation, ai_message

        answer = await self._pipeline.run(conversation, message_text)

        revised = await self._guardrails.check_output(
            answer.reply_text, user_input=message_text
        )
        if revised and revised != answer.reply_text:
            logger.info(
                "[ASSISTANT_OUTPUT_REVISED] conversation=%s",
                conversation.conversation_id,
            )
            answer.reply_text = revised

        ai_message = conversation.add_ai_message(answer)
        await self._conversations.save(conversation)

        logger.info(
            "[ASSISTANT_REPLY] conversation=%s lang=%s intent=%s urgency=%s "
            "source=%s degraded=%s",
            conversation.conversation_id,
            answer.language.value,
            answer.intent.value,
            answer.urgency.level.value if answer.urgency else UrgencyLevel.LOW.value,
            answer.knowledge_source.value,
            answer.degraded,
        )
        return conversation, ai_message

    async def _load_or_create(
        self, conversation_id: str | None, patient_user_id: str
    ) -> Conversation:
        if not conversation_id:
            return Conversation(patient_user_id=patient_user_id)

        conversation = await self._conversations.get(conversation_id, patient_user_id)
        if conversation is None:
            raise SessionNotFoundError(conversation_id)
        if conversation.patient_user_id != patient_user_id:
            raise AuthorizationException(
                "You are not authorised to access this conversation."
            )
        return conversation


class GetConversationUseCase:
    def __init__(self, *, conversations: ConversationRepositoryPort) -> None:
        self._conversations = conversations

    async def execute(self, conversation_id: str, patient_user_id: str) -> Conversation:
        conversation = await self._conversations.get(conversation_id, patient_user_id)
        if conversation is None:
            raise SessionNotFoundError(conversation_id)
        if conversation.patient_user_id != patient_user_id:
            raise AuthorizationException(
                "You are not authorised to access this conversation."
            )
        return conversation


class ListConversationsUseCase:
    """Backs the existing chat-history drawer."""

    def __init__(self, *, conversations: ConversationRepositoryPort) -> None:
        self._conversations = conversations

    async def execute(
        self, patient_user_id: str, *, limit: int = 50
    ) -> list[Conversation]:
        return await self._conversations.list_for_patient(patient_user_id, limit=limit)


class DeleteConversationUseCase:
    def __init__(self, *, conversations: ConversationRepositoryPort) -> None:
        self._conversations = conversations

    async def execute(self, conversation_id: str, patient_user_id: str) -> bool:
        deleted = await self._conversations.delete(conversation_id, patient_user_id)
        if not deleted:
            raise SessionNotFoundError(conversation_id)
        logger.info("[ASSISTANT_CONVERSATION_DELETED] id=%s", conversation_id)
        return True
