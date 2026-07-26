"""
Enumerations for the AI Medical Assistant.

Values are chosen to serialise directly into the shapes the existing React
cards already expect (`UrgencyLevel`, `CauseItem.confidence`,
`MedicineGuidanceItem.type`, `SpecialistCard.priority`). Changing a value here
changes the frontend contract.
"""

from enum import StrEnum


class MessageRole(StrEnum):
    """Author of a chat message. Mirrors the frontend `sender` field."""

    USER = "user"
    AI = "ai"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class IntentType(StrEnum):
    """What the patient wants from a given message."""

    SYMPTOM_REPORT = "symptom_report"
    MEDICAL_QUESTION = "medical_question"
    MEDICATION_QUESTION = "medication_question"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    SMALL_TALK = "small_talk"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"


class UrgencyLevel(StrEnum):
    """
    Clinical urgency.

    Serialised values are capitalised to match the frontend `UrgencyLevel`
    union exactly: "Low" | "Medium" | "High" | "Emergency".
    """

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    EMERGENCY = "Emergency"

    @property
    def rank(self) -> int:
        return _URGENCY_RANK[self]


_URGENCY_RANK: dict[UrgencyLevel, int] = {
    UrgencyLevel.LOW: 0,
    UrgencyLevel.MEDIUM: 1,
    UrgencyLevel.HIGH: 2,
    UrgencyLevel.EMERGENCY: 3,
}


class ConfidenceLabel(StrEnum):
    """Matches `CauseItem.confidence`: "Low" | "Medium" | "High"."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class MedicineType(StrEnum):
    """Matches `MedicineGuidanceItem.type`: "OTC" | "Prescription"."""

    OTC = "OTC"
    PRESCRIPTION = "Prescription"


class SpecialistPriority(StrEnum):
    """Matches `SpecialistCard.priority`: "Routine" | "Recommended" | "Urgent"."""

    ROUTINE = "Routine"
    RECOMMENDED = "Recommended"
    URGENT = "Urgent"


class EmergencyRisk(StrEnum):
    """
    Drives the right-hand panel's risk indicator.

    Values match the frontend union: "normal" | "moderate" | "critical".
    """

    NORMAL = "normal"
    MODERATE = "moderate"
    CRITICAL = "critical"

    @classmethod
    def from_urgency(cls, urgency: UrgencyLevel) -> "EmergencyRisk":
        if urgency is UrgencyLevel.EMERGENCY:
            return cls.CRITICAL
        if urgency is UrgencyLevel.HIGH:
            return cls.MODERATE
        return cls.NORMAL


class KnowledgeSource(StrEnum):
    """Where the grounding context for an answer came from."""

    RAG = "rag"
    WEB_SEARCH = "web_search"
    MODEL_ONLY = "model_only"
    NONE = "none"
