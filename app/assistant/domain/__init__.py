"""
Domain layer for the AI Medical Assistant.

Pure Python, no framework imports. Multilingual emergency detection and
evidence grounding are imported from `app.intake.domain.policies` rather than
reimplemented — those rules are platform-wide clinical safety policy, and two
copies would inevitably drift apart.
"""

from app.assistant.domain.entities import (
    AssistantAnswer,
    CauseItem,
    ChatMessage,
    Conversation,
    EmergencyNotice,
    LifestyleAdviceItem,
    MedicalEntity,
    MedicineGuidance,
    SpecialistSuggestion,
    UrgencyAssessment,
)
from app.assistant.domain.enums import (
    ConfidenceLabel,
    ConversationStatus,
    EmergencyRisk,
    IntentType,
    MedicineType,
    MessageRole,
    SpecialistPriority,
    UrgencyLevel,
)

__all__ = [
    "AssistantAnswer",
    "CauseItem",
    "ChatMessage",
    "ConfidenceLabel",
    "Conversation",
    "ConversationStatus",
    "EmergencyNotice",
    "EmergencyRisk",
    "IntentType",
    "LifestyleAdviceItem",
    "MedicalEntity",
    "MedicineGuidance",
    "MedicineType",
    "MessageRole",
    "SpecialistPriority",
    "SpecialistSuggestion",
    "UrgencyAssessment",
    "UrgencyLevel",
]
