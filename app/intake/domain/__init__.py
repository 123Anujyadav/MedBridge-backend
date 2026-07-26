"""
Domain layer for the Medical Case Intake Agent.

Pure Python. No FastAPI, no SQLAlchemy, no LangChain, no network. Everything in
here is directly unit-testable without any I/O.
"""

from app.intake.domain.entities import (
    ConversationTurn,
    ExtractedEntity,
    IntakeSession,
    MedicalCase,
    SpecialistRecommendation,
)
from app.intake.domain.enums import (
    ConfidenceBand,
    EntityKind,
    IntentType,
    Language,
    SessionStatus,
    TurnRole,
    UrgencyLevel,
)
from app.intake.domain.errors import (
    DomainError,
    EvidenceNotGroundedError,
    InvalidSessionStateError,
    SessionNotFoundError,
)
from app.intake.domain.policies import (
    MANDATORY_ENTITY_KINDS,
    MAX_FOLLOWUP_ROUNDS,
    MIN_ENTITY_CONFIDENCE,
    MIN_OVERALL_CONFIDENCE,
    ReadinessVerdict,
    detect_red_flags,
    evaluate_readiness,
    is_evidence_grounded,
)
from app.intake.domain.value_objects import Confidence, Evidence

__all__ = [
    "ConfidenceBand",
    "Confidence",
    "ConversationTurn",
    "DomainError",
    "EntityKind",
    "Evidence",
    "EvidenceNotGroundedError",
    "ExtractedEntity",
    "IntakeSession",
    "IntentType",
    "InvalidSessionStateError",
    "Language",
    "MANDATORY_ENTITY_KINDS",
    "MAX_FOLLOWUP_ROUNDS",
    "MIN_ENTITY_CONFIDENCE",
    "MIN_OVERALL_CONFIDENCE",
    "MedicalCase",
    "ReadinessVerdict",
    "SessionNotFoundError",
    "SessionStatus",
    "SpecialistRecommendation",
    "TurnRole",
    "UrgencyLevel",
    "detect_red_flags",
    "evaluate_readiness",
    "is_evidence_grounded",
]
