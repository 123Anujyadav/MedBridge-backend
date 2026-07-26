"""
Domain entities for the AI Medical Assistant.

`AssistantAnswer.to_ui_payload()` is the contract with the existing frontend: it
emits exactly the `AIResponseData` shape that `StructuredAIResponse.tsx` renders,
including its camelCase keys. Nothing in the React layer changes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.assistant.domain.enums import (
    ConfidenceLabel,
    ConversationStatus,
    EmergencyRisk,
    IntentType,
    KnowledgeSource,
    MedicineType,
    MessageRole,
    SpecialistPriority,
    UrgencyLevel,
)
from app.intake.domain.enums import Language


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Structured answer parts — each maps to one existing React card
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CauseItem:
    """-> CausesCard `CauseItem`."""

    cause: str
    confidence: ConfidenceLabel = ConfidenceLabel.MEDIUM

    def to_ui(self) -> dict[str, Any]:
        return {"cause": self.cause, "confidence": self.confidence.value}


@dataclass(slots=True)
class LifestyleAdviceItem:
    """-> LifestyleCard `LifestyleAdviceItem`."""

    category: str
    description: str

    def to_ui(self) -> dict[str, Any]:
        return {"category": self.category, "description": self.description}


@dataclass(slots=True)
class MedicineGuidance:
    """
    -> MedicineCard `MedicineGuidanceItem`.

    Deliberately guidance, never a prescription: the assistant may describe a
    medicine class the patient already mentioned or a common OTC option, and
    the card renders precautions alongside it.
    """

    name: str
    purpose: str
    side_effects: list[str] = field(default_factory=list)
    precautions: str = ""
    type: MedicineType = MedicineType.OTC

    def to_ui(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "sideEffects": list(self.side_effects),
            "precautions": self.precautions,
            "type": self.type.value,
        }


@dataclass(slots=True)
class SpecialistSuggestion:
    """-> SpecialistCard."""

    name: str
    reason: str
    priority: SpecialistPriority = SpecialistPriority.ROUTINE

    def to_ui(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reason": self.reason,
            "priority": self.priority.value,
        }


@dataclass(slots=True)
class UrgencyAssessment:
    """-> UrgencyCard."""

    level: UrgencyLevel = UrgencyLevel.LOW
    explanation: str = ""

    def to_ui(self) -> dict[str, Any]:
        return {"level": self.level.value, "explanation": self.explanation}


@dataclass(slots=True)
class EmergencyNotice:
    """-> EmergencyCard. Present only when red flags fire."""

    heading: str
    description: str
    nearest_hospital: str | None = None

    def to_ui(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "heading": self.heading,
            "description": self.description,
        }
        if self.nearest_hospital:
            payload["nearestHospital"] = self.nearest_hospital
        return payload


@dataclass(slots=True)
class MedicalEntity:
    """
    A clinical fact extracted from the patient's message.

    Carries confidence and the verbatim span it came from, so the same
    evidence-grounding check the intake agent uses can reject fabrications here
    too.
    """

    kind: str
    value: str
    confidence: float = 0.0
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MedicalEntity:
        return cls(
            kind=str(raw.get("kind", "symptom")),
            value=str(raw.get("value", "")),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            evidence=str(raw.get("evidence", "")),
        )


# --------------------------------------------------------------------------
# The assembled answer
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AssistantAnswer:
    """One structured reply, ready for the existing cards."""

    reply_text: str = ""
    """Prose shown in the chat bubble above the cards."""

    summary: str = ""
    symptoms: list[str] = field(default_factory=list)
    causes: list[CauseItem] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    lifestyle_advice: list[LifestyleAdviceItem] = field(default_factory=list)
    medicines: list[MedicineGuidance] = field(default_factory=list)
    specialist: SpecialistSuggestion | None = None
    urgency: UrgencyAssessment | None = None
    emergency: EmergencyNotice | None = None
    references: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)

    # Metadata (not rendered as cards, used by the page and persistence)
    conversation_title: str = "New consultation"
    language: Language = Language.ENGLISH
    intent: IntentType = IntentType.UNCLEAR
    confidence: float = 0.0
    knowledge_source: KnowledgeSource = KnowledgeSource.MODEL_ONLY
    red_flags: list[str] = field(default_factory=list)
    entities: list[MedicalEntity] = field(default_factory=list)
    generated_at: str = field(default_factory=_utc_now_iso)
    degraded: bool = False

    @property
    def emergency_risk(self) -> EmergencyRisk:
        if self.emergency is not None:
            return EmergencyRisk.CRITICAL
        return EmergencyRisk.from_urgency(
            self.urgency.level if self.urgency else UrgencyLevel.LOW
        )

    def to_ui_payload(self) -> dict[str, Any]:
        """
        Emit the exact `AIResponseData` object the React cards consume.

        Empty collections are omitted rather than sent as `[]` because each card
        renders on a truthy-and-non-empty check; omitting keeps the payload
        honest about what the assistant actually produced.
        """
        payload: dict[str, Any] = {}
        if self.summary:
            payload["summary"] = self.summary
        if self.symptoms:
            payload["symptoms"] = list(self.symptoms)
        if self.causes:
            payload["causes"] = [c.to_ui() for c in self.causes]
        if self.actions:
            payload["actions"] = list(self.actions)
        if self.lifestyle_advice:
            payload["lifestyleAdvice"] = [a.to_ui() for a in self.lifestyle_advice]
        if self.medicines:
            payload["medicines"] = [m.to_ui() for m in self.medicines]
        if self.specialist:
            payload["specialist"] = self.specialist.to_ui()
        if self.urgency:
            payload["urgency"] = self.urgency.to_ui()
        if self.emergency:
            payload["emergency"] = self.emergency.to_ui()
        if self.references:
            payload["references"] = list(self.references)
        if self.follow_up_questions:
            payload["followUpQuestions"] = list(self.follow_up_questions)
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation, for persistence and the session store."""
        return {
            "reply_text": self.reply_text,
            "structured": self.to_ui_payload(),
            "conversation_title": self.conversation_title,
            "language": self.language.value,
            "intent": self.intent.value,
            "confidence": round(float(self.confidence), 4),
            "knowledge_source": self.knowledge_source.value,
            "emergency_risk": self.emergency_risk.value,
            "red_flags": list(self.red_flags),
            "entities": [e.to_dict() for e in self.entities],
            "generated_at": self.generated_at,
            "degraded": self.degraded,
        }


# --------------------------------------------------------------------------
# Conversation aggregate
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ChatMessage:
    """One turn in a conversation."""

    role: MessageRole
    text: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    answer: AssistantAnswer | None = None
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "text": self.text,
            "answer": self.answer.to_dict() if self.answer else None,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class Conversation:
    """
    Aggregate root: one patient's assistant thread.

    Accumulated symptoms and asked questions live here so the pipeline can
    honour "remember previous symptoms" and "avoid asking duplicate questions"
    without re-deriving them from the transcript each turn.
    """

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patient_user_id: str = ""
    title: str = "New consultation"
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: list[ChatMessage] = field(default_factory=list)
    known_symptoms: list[str] = field(default_factory=list)
    asked_questions: list[str] = field(default_factory=list)
    summary: str = ""
    last_specialist: str | None = None
    references: list[str] = field(default_factory=list)
    language: Language = Language.ENGLISH
    emergency_risk: EmergencyRisk = EmergencyRisk.NORMAL
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    message_count: int = 0
    """
    Turns in this thread.

    Populated by the repository's list query, where transcripts are not loaded.
    When messages *are* loaded, `len(self.messages)` is authoritative.
    """

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()

    def add_user_message(self, text: str) -> ChatMessage:
        message = ChatMessage(role=MessageRole.USER, text=text)
        self.messages.append(message)
        self.touch()
        return message

    def add_ai_message(self, answer: AssistantAnswer) -> ChatMessage:
        """Append the assistant's turn and fold its findings into memory."""
        message = ChatMessage(
            role=MessageRole.AI, text=answer.reply_text, answer=answer
        )
        self.messages.append(message)

        self.remember_symptoms(answer.symptoms)
        self.remember_questions(answer.follow_up_questions)

        if answer.summary:
            self.summary = answer.summary
        if answer.specialist:
            self.last_specialist = answer.specialist.name
        for reference in answer.references:
            if reference not in self.references:
                self.references.append(reference)
        if self.title == "New consultation" and answer.conversation_title:
            self.title = answer.conversation_title[:120]
        self.language = answer.language
        self.emergency_risk = answer.emergency_risk

        self.touch()
        return message

    def remember_symptoms(self, symptoms: list[str]) -> None:
        """Union of symptoms, case-insensitive, order preserved."""
        seen = {s.casefold() for s in self.known_symptoms}
        for symptom in symptoms:
            if symptom and symptom.casefold() not in seen:
                self.known_symptoms.append(symptom)
                seen.add(symptom.casefold())

    def remember_questions(self, questions: list[str]) -> None:
        seen = {q.casefold() for q in self.asked_questions}
        for question in questions:
            if question and question.casefold() not in seen:
                self.asked_questions.append(question)
                seen.add(question.casefold())

    def was_already_asked(self, question: str) -> bool:
        """Used to suppress duplicate follow-up questions across turns."""
        return question.casefold() in {q.casefold() for q in self.asked_questions}

    def recent_messages(self, limit: int) -> list[ChatMessage]:
        return self.messages[-limit:] if limit > 0 else []

    def transcript_for_grounding(self) -> str:
        """
        Patient turns only.

        Assistant text is excluded so the model cannot cite its own earlier
        wording as evidence of something the patient said.
        """
        return "\n".join(
            m.text for m in self.messages if m.role == MessageRole.USER and m.text
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "patient_user_id": self.patient_user_id,
            "title": self.title,
            "status": self.status.value,
            "messages": [m.to_dict() for m in self.messages],
            "known_symptoms": list(self.known_symptoms),
            "asked_questions": list(self.asked_questions),
            "summary": self.summary,
            "last_specialist": self.last_specialist,
            "references": list(self.references),
            "language": self.language.value,
            "emergency_risk": self.emergency_risk.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
