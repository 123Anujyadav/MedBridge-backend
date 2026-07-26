"""
Domain entities for the Medical Case Intake Agent.

`IntakeSession` is the aggregate root: it owns the conversation, the extracted
clinical entities, and the generated case. Everything is plain-dataclass and
round-trips through `to_dict`/`from_dict` so the session store can persist it as
JSON without the domain knowing that Redis exists.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.intake.domain.enums import (
    EntityKind,
    IntentType,
    Language,
    SessionStatus,
    TurnRole,
    UrgencyLevel,
)
from app.intake.domain.value_objects import Confidence, Evidence

# Sentinel written into the structured case when a field was never stated.
# The agent reports absence explicitly rather than guessing a plausible value.
UNKNOWN_VALUE = "Unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ConversationTurn:
    """One utterance in the intake dialogue."""

    role: TurnRole
    text: str
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "text": self.text,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ConversationTurn:
        return cls(
            role=TurnRole(raw.get("role", TurnRole.PATIENT.value)),
            text=str(raw.get("text", "")),
            timestamp=str(raw.get("timestamp") or _utc_now_iso()),
        )


@dataclass(slots=True)
class ExtractedEntity:
    """
    A single piece of clinical information pulled from the patient's words.

    Carries the three things the accuracy requirements demand: the value, a
    confidence score, and the evidence span it was derived from.
    """

    kind: EntityKind
    value: str
    confidence: Confidence
    evidence: Evidence

    @property
    def is_unknown(self) -> bool:
        return self.value == UNKNOWN_VALUE or self.confidence.is_unknown

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "confidence": self.confidence.to_dict(),
            "evidence": self.evidence.to_dict(),
            "is_unknown": self.is_unknown,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExtractedEntity:
        return cls(
            kind=EntityKind(raw["kind"]),
            value=str(raw.get("value", UNKNOWN_VALUE)),
            confidence=Confidence.from_dict(raw.get("confidence")),
            evidence=Evidence.from_dict(raw.get("evidence")),
        )

    @classmethod
    def unknown(cls, kind: EntityKind) -> ExtractedEntity:
        """An explicit 'the patient did not tell us this' marker."""
        return cls(
            kind=kind,
            value=UNKNOWN_VALUE,
            confidence=Confidence.unknown(),
            evidence=Evidence(quote=""),
        )


@dataclass(slots=True)
class SpecialistRecommendation:
    """A suggested specialist, ranked against the generated case."""

    specialty: str
    rationale: str
    match_score: float = 0.0
    doctor_id: str | None = None
    doctor_name: str | None = None
    hospital_name: str | None = None
    is_available: bool = True
    avatar_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialty": self.specialty,
            "rationale": self.rationale,
            "match_score": round(float(self.match_score), 2),
            "doctor_id": self.doctor_id,
            "doctor_name": self.doctor_name,
            "hospital_name": self.hospital_name,
            "is_available": self.is_available,
            "avatar_url": self.avatar_url,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SpecialistRecommendation:
        return cls(
            specialty=str(raw.get("specialty", "General Medicine")),
            rationale=str(raw.get("rationale", "")),
            match_score=float(raw.get("match_score", 0.0) or 0.0),
            doctor_id=raw.get("doctor_id"),
            doctor_name=raw.get("doctor_name"),
            hospital_name=raw.get("hospital_name"),
            is_available=bool(raw.get("is_available", True)),
            avatar_url=raw.get("avatar_url"),
        )


@dataclass(slots=True)
class MedicalCase:
    """
    The structured clinical summary handed to a doctor.

    Explicitly *not* a diagnosis. `differential_considerations` are prompts for
    a clinician to evaluate, never conclusions, and every list here is derived
    from grounded entities rather than model free-association.
    """

    chief_complaint: str
    symptoms: list[str] = field(default_factory=list)
    duration: str = UNKNOWN_VALUE
    severity: str = UNKNOWN_VALUE
    onset: str = UNKNOWN_VALUE
    body_sites: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    medical_history: list[str] = field(default_factory=list)
    aggravating_factors: list[str] = field(default_factory=list)
    relieving_factors: list[str] = field(default_factory=list)
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    red_flags: list[str] = field(default_factory=list)
    differential_considerations: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    recommended_specialty: str = "General Medicine"
    overall_confidence: Confidence = field(default_factory=Confidence.unknown)
    patient_language: Language = Language.ENGLISH
    summary_for_doctor: str = ""
    generated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chief_complaint": self.chief_complaint,
            "symptoms": list(self.symptoms),
            "duration": self.duration,
            "severity": self.severity,
            "onset": self.onset,
            "body_sites": list(self.body_sites),
            "allergies": list(self.allergies),
            "current_medications": list(self.current_medications),
            "medical_history": list(self.medical_history),
            "aggravating_factors": list(self.aggravating_factors),
            "relieving_factors": list(self.relieving_factors),
            "urgency": self.urgency.value,
            "red_flags": list(self.red_flags),
            "differential_considerations": list(self.differential_considerations),
            "missing_information": list(self.missing_information),
            "recommended_specialty": self.recommended_specialty,
            "overall_confidence": self.overall_confidence.to_dict(),
            "patient_language": self.patient_language.value,
            "summary_for_doctor": self.summary_for_doctor,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MedicalCase:
        return cls(
            chief_complaint=str(raw.get("chief_complaint", "")),
            symptoms=list(raw.get("symptoms") or []),
            duration=str(raw.get("duration", UNKNOWN_VALUE)),
            severity=str(raw.get("severity", UNKNOWN_VALUE)),
            onset=str(raw.get("onset", UNKNOWN_VALUE)),
            body_sites=list(raw.get("body_sites") or []),
            allergies=list(raw.get("allergies") or []),
            current_medications=list(raw.get("current_medications") or []),
            medical_history=list(raw.get("medical_history") or []),
            aggravating_factors=list(raw.get("aggravating_factors") or []),
            relieving_factors=list(raw.get("relieving_factors") or []),
            urgency=UrgencyLevel(raw.get("urgency", UrgencyLevel.MEDIUM.value)),
            red_flags=list(raw.get("red_flags") or []),
            differential_considerations=list(
                raw.get("differential_considerations") or []
            ),
            missing_information=list(raw.get("missing_information") or []),
            recommended_specialty=str(
                raw.get("recommended_specialty", "General Medicine")
            ),
            overall_confidence=Confidence.from_dict(raw.get("overall_confidence")),
            patient_language=Language(
                raw.get("patient_language", Language.ENGLISH.value)
            ),
            summary_for_doctor=str(raw.get("summary_for_doctor", "")),
            generated_at=str(raw.get("generated_at") or _utc_now_iso()),
        )


@dataclass(slots=True)
class IntakeSession:
    """
    Aggregate root. Owns one patient's intake conversation from first utterance
    through to doctor routing.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patient_user_id: str = ""
    status: SessionStatus = SessionStatus.COLLECTING
    language: Language = Language.UNKNOWN
    intent: IntentType = IntentType.UNCLEAR
    turns: list[ConversationTurn] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    followup_rounds: int = 0
    pending_question: str | None = None
    medical_case: MedicalCase | None = None
    recommendations: list[SpecialistRecommendation] = field(default_factory=list)
    routed_case_id: str | None = None
    routed_doctor_id: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    # ---- transcript helpers -------------------------------------------------

    def add_turn(self, role: TurnRole, text: str) -> None:
        self.turns.append(ConversationTurn(role=role, text=text))
        self.touch()

    def patient_transcript(self) -> str:
        """
        All patient utterances concatenated.

        This is the *only* text an extraction may cite as evidence — agent
        questions are excluded so the model cannot launder its own prior wording
        into fake patient statements.
        """
        return "\n".join(
            t.text for t in self.turns if t.role == TurnRole.PATIENT and t.text
        )

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()

    # ---- entity helpers -----------------------------------------------------

    def entities_of(self, kind: EntityKind) -> list[ExtractedEntity]:
        return [e for e in self.entities if e.kind == kind and not e.is_unknown]

    def values_of(self, kind: EntityKind) -> list[str]:
        """Deduplicated, order-preserving values for one entity kind."""
        seen: dict[str, None] = {}
        for entity in self.entities_of(kind):
            seen.setdefault(entity.value, None)
        return list(seen.keys())

    def merge_entities(self, incoming: list[ExtractedEntity]) -> None:
        """
        Fold a new extraction pass into the accumulated set.

        On a (kind, normalised value) collision the higher-confidence version
        wins, so later turns can strengthen an earlier tentative reading without
        duplicating it.
        """
        index: dict[tuple[EntityKind, str], int] = {
            (e.kind, e.value.strip().lower()): i for i, e in enumerate(self.entities)
        }
        for entity in incoming:
            key = (entity.kind, entity.value.strip().lower())
            existing_pos = index.get(key)
            if existing_pos is None:
                self.entities.append(entity)
                index[key] = len(self.entities) - 1
            elif entity.confidence.score > self.entities[existing_pos].confidence.score:
                self.entities[existing_pos] = entity
        self.touch()

    # ---- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "patient_user_id": self.patient_user_id,
            "status": self.status.value,
            "language": self.language.value,
            "intent": self.intent.value,
            "turns": [t.to_dict() for t in self.turns],
            "entities": [e.to_dict() for e in self.entities],
            "red_flags": list(self.red_flags),
            "followup_rounds": self.followup_rounds,
            "pending_question": self.pending_question,
            "medical_case": self.medical_case.to_dict() if self.medical_case else None,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "routed_case_id": self.routed_case_id,
            "routed_doctor_id": self.routed_doctor_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> IntakeSession:
        case_raw = raw.get("medical_case")
        return cls(
            session_id=str(raw.get("session_id") or uuid.uuid4()),
            patient_user_id=str(raw.get("patient_user_id", "")),
            status=SessionStatus(raw.get("status", SessionStatus.COLLECTING.value)),
            language=Language(raw.get("language", Language.UNKNOWN.value)),
            intent=IntentType(raw.get("intent", IntentType.UNCLEAR.value)),
            turns=[ConversationTurn.from_dict(t) for t in raw.get("turns") or []],
            entities=[ExtractedEntity.from_dict(e) for e in raw.get("entities") or []],
            red_flags=list(raw.get("red_flags") or []),
            followup_rounds=int(raw.get("followup_rounds", 0) or 0),
            pending_question=raw.get("pending_question"),
            medical_case=MedicalCase.from_dict(case_raw) if case_raw else None,
            recommendations=[
                SpecialistRecommendation.from_dict(r)
                for r in raw.get("recommendations") or []
            ],
            routed_case_id=raw.get("routed_case_id"),
            routed_doctor_id=raw.get("routed_doctor_id"),
            created_at=str(raw.get("created_at") or _utc_now_iso()),
            updated_at=str(raw.get("updated_at") or _utc_now_iso()),
        )
