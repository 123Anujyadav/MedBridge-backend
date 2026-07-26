"""
SQLAlchemy repository for assistant conversations.

Follows the project's transaction convention: flush here, let the `get_db`
request wrapper own the commit.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.assistant.domain.entities import (
    AssistantAnswer,
    ChatMessage,
    Conversation,
    MedicalEntity,
)
from app.assistant.domain.enums import (
    ConversationStatus,
    EmergencyRisk,
    IntentType,
    KnowledgeSource,
    MessageRole,
    UrgencyLevel,
)
from app.intake.domain.enums import Language
from app.models.assistant import AssistantConversation, AssistantMessage

logger = logging.getLogger(__name__)


class SqlConversationRepository:
    """Maps the `Conversation` aggregate to the assistant tables."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # -- reads -------------------------------------------------------------

    async def get(
        self, conversation_id: str, patient_user_id: str
    ) -> Conversation | None:
        row = await self._load_row(conversation_id)
        if row is None:
            return None
        if str(row.patient_user_id) != str(patient_user_id):
            # Return None rather than the row: the use case turns this into a
            # 404, so a probing client cannot confirm the id exists.
            logger.warning(
                "[ASSISTANT_CONV_ACCESS_DENIED] conversation=%s", conversation_id
            )
            return None
        return self._to_domain(row)

    async def list_for_patient(
        self, patient_user_id: str, *, limit: int = 50
    ) -> list[Conversation]:
        """
        Conversation list for the history drawer.

        Transcripts are intentionally not eager-loaded — the drawer renders
        titles and previews. The message count the UI needs comes from a
        correlated subquery instead, so listing stays a single cheap query
        regardless of how long the conversations are.
        """
        try:
            parsed = uuid.UUID(str(patient_user_id))
        except (ValueError, TypeError):
            return []

        message_count = (
            select(func.count(AssistantMessage.id))
            .where(AssistantMessage.conversation_id == AssistantConversation.id)
            .correlate(AssistantConversation)
            .scalar_subquery()
        )

        result = await self._db.execute(
            select(AssistantConversation, message_count.label("message_count"))
            .where(AssistantConversation.patient_user_id == parsed)
            .where(AssistantConversation.deleted_at.is_(None))
            .order_by(AssistantConversation.updated_at.desc())
            .limit(limit)
        )

        conversations: list[Conversation] = []
        for row, count in result.all():
            conversation = self._to_domain(row, include_messages=False)
            conversation.message_count = int(count or 0)
            conversations.append(conversation)
        return conversations

    # -- writes ------------------------------------------------------------

    async def save(self, conversation: Conversation) -> None:
        """
        Upsert the conversation and append any messages not yet persisted.

        Messages are matched on `message_key` so repeated saves of the same
        aggregate do not duplicate rows.
        """
        row = await self._load_row(conversation.conversation_id)

        if row is None:
            row = AssistantConversation(
                conversation_key=conversation.conversation_id,
                patient_user_id=uuid.UUID(conversation.patient_user_id),
            )
            self._db.add(row)
            await self._db.flush()

        row.title = conversation.title[:200]
        row.status = conversation.status.value
        row.summary = conversation.summary or ""
        row.language = conversation.language.value
        row.emergency_risk = conversation.emergency_risk.value
        row.last_specialist = (
            conversation.last_specialist[:150] if conversation.last_specialist else None
        )
        row.known_symptoms = list(conversation.known_symptoms)
        row.asked_questions = list(conversation.asked_questions)
        row.references = list(conversation.references)

        # Selecting a single column yields the raw values, not ORM rows.
        existing_keys = set(
            (
                await self._db.execute(
                    select(AssistantMessage.message_key).where(
                        AssistantMessage.conversation_id == row.id
                    )
                )
            )
            .scalars()
            .all()
        )

        for message in conversation.messages:
            if message.message_id in existing_keys:
                continue
            self._db.add(self._to_row(row.id, message))

        await self._db.flush()
        logger.info(
            "[ASSISTANT_CONV_SAVED] conversation=%s messages=%d",
            conversation.conversation_id,
            len(conversation.messages),
        )

    async def delete(self, conversation_id: str, patient_user_id: str) -> bool:
        """Soft delete, preserving the clinical record for audit."""
        row = await self._load_row(conversation_id)
        if row is None or str(row.patient_user_id) != str(patient_user_id):
            return False
        row.soft_delete()
        await self._db.flush()
        return True

    # -- mapping -----------------------------------------------------------

    async def _load_row(self, conversation_id: str) -> AssistantConversation | None:
        result = await self._db.execute(
            select(AssistantConversation)
            .options(selectinload(AssistantConversation.messages))
            .where(AssistantConversation.conversation_key == conversation_id)
            .where(AssistantConversation.deleted_at.is_(None))
        )
        return result.scalars().first()

    @staticmethod
    def _to_row(conversation_row_id: uuid.UUID, message: ChatMessage) -> AssistantMessage:
        answer = message.answer
        return AssistantMessage(
            conversation_id=conversation_row_id,
            message_key=message.message_id,
            role=message.role.value,
            text=message.text or "",
            structured=answer.to_ui_payload() if answer else None,
            entities=[e.to_dict() for e in answer.entities] if answer else [],
            intent=answer.intent.value if answer else IntentType.UNCLEAR.value,
            urgency=(
                answer.urgency.level.value
                if answer and answer.urgency
                else UrgencyLevel.LOW.value
            ),
            knowledge_source=(
                answer.knowledge_source.value
                if answer
                else KnowledgeSource.NONE.value
            ),
            confidence=float(answer.confidence) if answer else 0.0,
            degraded=bool(answer.degraded) if answer else False,
        )

    @staticmethod
    def _to_domain(
        row: AssistantConversation, *, include_messages: bool = True
    ) -> Conversation:
        conversation = Conversation(
            conversation_id=row.conversation_key,
            patient_user_id=str(row.patient_user_id),
            title=row.title or "New consultation",
            status=ConversationStatus(row.status or "active"),
            known_symptoms=list(row.known_symptoms or []),
            asked_questions=list(row.asked_questions or []),
            summary=row.summary or "",
            last_specialist=row.last_specialist,
            references=list(row.references or []),
            language=Language(row.language or Language.ENGLISH.value),
            emergency_risk=EmergencyRisk(row.emergency_risk or "normal"),
            created_at=row.created_at.isoformat() if row.created_at else "",
            updated_at=row.updated_at.isoformat() if row.updated_at else "",
        )

        if include_messages:
            for message_row in row.messages:
                answer = None
                if message_row.role == MessageRole.AI.value:
                    answer = AssistantAnswer(
                        reply_text=message_row.text or "",
                        summary=(message_row.structured or {}).get("summary", ""),
                        language=conversation.language,
                        confidence=float(message_row.confidence or 0.0),
                        degraded=bool(message_row.degraded),
                        entities=[
                            MedicalEntity.from_dict(e)
                            for e in (message_row.entities or [])
                        ],
                    )
                conversation.messages.append(
                    ChatMessage(
                        role=MessageRole(message_row.role),
                        text=message_row.text or "",
                        message_id=message_row.message_key,
                        answer=answer,
                        created_at=(
                            message_row.created_at.isoformat()
                            if message_row.created_at
                            else ""
                        ),
                    )
                )

        return conversation

    @staticmethod
    def stored_payload(row: AssistantMessage) -> dict:
        """The structured payload exactly as it was sent to the UI."""
        return row.structured or {}
