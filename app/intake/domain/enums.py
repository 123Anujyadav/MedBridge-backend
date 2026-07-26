"""
Domain enumerations for the Medical Case Intake Agent.

All string-valued so they serialize cleanly to JSON/Redis/Postgres without
custom encoders.
"""

from enum import StrEnum


class Language(StrEnum):
    """Detected language of a patient utterance."""

    ENGLISH = "english"
    HINDI = "hindi"
    HINGLISH = "hinglish"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class IntentType(StrEnum):
    """What the patient is trying to do in a given turn."""

    SYMPTOM_REPORT = "symptom_report"
    FOLLOWUP_ANSWER = "followup_answer"
    EMERGENCY = "emergency"
    QUESTION = "question"
    SMALL_TALK = "small_talk"
    UNCLEAR = "unclear"


class UrgencyLevel(StrEnum):
    """
    Clinical urgency. Ordered least -> most severe.

    Deliberately mirrors the `urgency_level` CHECK constraint on the existing
    `cases` table so triage output maps onto persistence without translation.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric severity, for comparisons and max()."""
        return _URGENCY_RANK[self]


_URGENCY_RANK: dict[UrgencyLevel, int] = {
    UrgencyLevel.LOW: 0,
    UrgencyLevel.MEDIUM: 1,
    UrgencyLevel.HIGH: 2,
    UrgencyLevel.CRITICAL: 3,
}


class EntityKind(StrEnum):
    """
    Categories of clinical information the agent extracts.

    Kept deliberately small and concrete. Anything the model wants to report
    that does not fit one of these is discarded rather than invented into a new
    category.
    """

    SYMPTOM = "symptom"
    DURATION = "duration"
    SEVERITY = "severity"
    BODY_SITE = "body_site"
    ONSET = "onset"
    ALLERGY = "allergy"
    MEDICATION = "medication"
    MEDICAL_HISTORY = "medical_history"
    AGGRAVATING_FACTOR = "aggravating_factor"
    RELIEVING_FACTOR = "relieving_factor"


class ConfidenceBand(StrEnum):
    """Human-readable bucket for a confidence score."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class SessionStatus(StrEnum):
    """Lifecycle of an intake session."""

    COLLECTING = "collecting"
    """Agent has asked a follow-up and is waiting for the patient to answer."""

    AWAITING_DOCTOR_SELECTION = "awaiting_doctor_selection"
    """Structured case is ready; waiting for the human to pick a specialist."""

    ROUTED = "routed"
    """Case persisted and routed to the selected doctor. Terminal."""

    EMERGENCY_ESCALATED = "emergency_escalated"
    """Red flags detected. Intake short-circuited to emergency guidance. Terminal."""

    ABANDONED = "abandoned"
    """Session expired or was explicitly cancelled. Terminal."""

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {
        SessionStatus.ROUTED,
        SessionStatus.EMERGENCY_ESCALATED,
        SessionStatus.ABANDONED,
    }
)


class TurnRole(StrEnum):
    """Author of a conversation turn."""

    PATIENT = "patient"
    AGENT = "agent"
