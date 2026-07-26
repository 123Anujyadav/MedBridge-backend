"""
Intake workflow nodes.

One method per stage of the agent, each independently testable. Nodes are
async, take and return `IntakeState`, and are required to degrade safely: if the
model returns nothing usable the node records a notice and falls back to
deterministic behaviour rather than emitting empty or invented clinical data.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.ai_core.utils.validators import sanitize_prompt_input
from app.intake.application.dto import DoctorRef
from app.intake.application.ports import DoctorDirectoryPort, LLMPort
from app.intake.domain.entities import (
    UNKNOWN_VALUE,
    ExtractedEntity,
    MedicalCase,
    SpecialistRecommendation,
)
from app.intake.domain.enums import (
    EntityKind,
    IntentType,
    Language,
    SessionStatus,
    TurnRole,
    UrgencyLevel,
)
from app.intake.domain.policies import (
    detect_red_flags,
    enforce_grounding,
    evaluate_readiness,
)
from app.intake.domain.specialties import (
    EMERGENCY_MEDICINE,
    GENERAL_MEDICINE,
    canonicalise_specialty,
)
from app.intake.domain.value_objects import Confidence, Evidence
from app.intake.workflow import prompts
from app.intake.workflow.state import IntakeState

logger = logging.getLogger(__name__)

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Romanised-Hindi markers. Common enough to be diagnostic, rare enough in
# English clinical text to avoid false positives.
_HINGLISH_MARKERS = frozenset(
    {
        "hai", "hain", "mujhe", "mera", "meri", "nahi", "nahin", "kya", "aur",
        "bahut", "thoda", "dard", "din", "raha", "rahi", "ho", "se", "ka", "ki",
        "ke", "me", "mein", "par", "kar", "kuch", "abhi", "saans", "pet", "sir",
        "bukhar", "khansi", "chakkar", "kamzori", "ulti",
    }
)

_EMERGENCY_GUIDANCE = (
    "Your description contains signs that may indicate a medical emergency. "
    "Do not wait for an online consultation. Call your local emergency number "
    "or go to the nearest emergency department immediately. "
    "If you are with someone, tell them now."
)

# Ceiling on entities accepted from one extraction pass, to bound both prompt
# growth and the blast radius of a runaway model response.
_MAX_ENTITIES_PER_PASS = 40


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class IntakeNodes:
    """
    Dependency-injected node collection.

    Holds only the LLM, which is process-wide. The doctor directory is
    request-scoped and arrives via `IntakeState`, which keeps the compiled graph
    reusable across requests.
    """

    def __init__(self, *, llm: LLMPort) -> None:
        self._llm = llm

    # -- stage 1: receive input -------------------------------------------

    async def receive_input(self, state: IntakeState) -> IntakeState:
        """
        Sanitise the newest turn and run deterministic red-flag detection.

        Red flags are computed here, before any model call, so an unavailable or
        misbehaving LLM can never suppress an emergency escalation.
        """
        session = state["session"]
        state.setdefault("notices", [])
        state.setdefault("rejected_entities", [])
        state["llm_degraded"] = False

        latest = ""
        for turn in reversed(session.turns):
            if turn.role == TurnRole.PATIENT:
                latest = turn.text
                break

        state["latest_text"] = sanitize_prompt_input(latest, max_length=4000)

        flags = detect_red_flags(session.patient_transcript())
        if flags:
            new_flags = [f for f in flags if f not in session.red_flags]
            session.red_flags.extend(new_flags)
            if new_flags:
                logger.warning(
                    "[INTAKE_RED_FLAG] session=%s flags=%s",
                    session.session_id,
                    new_flags,
                )

        return state

    # -- stage 2: language detection ---------------------------------------

    async def detect_language(self, state: IntakeState) -> IntakeState:
        """
        Identify the patient's language.

        Script inspection is decisive and free, so it runs first; the model is
        consulted only for the genuinely ambiguous Latin-script case.
        """
        session = state["session"]
        text = state.get("latest_text", "")
        if not text:
            session.language = Language.UNKNOWN
            return state

        has_devanagari = bool(_DEVANAGARI_RE.search(text))
        words = {w.strip(".,!?;:").casefold() for w in text.split()}
        hinglish_hits = len(words & _HINGLISH_MARKERS)
        has_latin = bool(re.search(r"[A-Za-z]", text))

        if has_devanagari and has_latin:
            session.language = Language.MIXED
            return state
        if has_devanagari:
            session.language = Language.HINDI
            return state
        if hinglish_hits >= 2:
            session.language = Language.HINGLISH
            return state

        # Ambiguous Latin script: one marker word, or very short input.
        if hinglish_hits == 1 or len(words) <= 2:
            payload = await self._llm.complete_json(
                system_prompt=prompts.LANGUAGE_SYSTEM_PROMPT,
                user_content=prompts.build_language_user_content(text),
                max_tokens=100,
            )
            raw = str(payload.get("language", "")).strip().casefold()
            try:
                session.language = Language(raw)
                return state
            except ValueError:
                pass

        session.language = Language.ENGLISH
        return state

    # -- stage 3: intent detection -----------------------------------------

    async def detect_intent(self, state: IntakeState) -> IntakeState:
        """Classify what the patient is doing in this turn."""
        session = state["session"]
        text = state.get("latest_text", "")

        # A red flag settles intent on its own; no model call needed.
        if session.red_flags:
            session.intent = IntentType.EMERGENCY
            return state

        if not text:
            session.intent = IntentType.UNCLEAR
            return state

        payload = await self._llm.complete_json(
            system_prompt=prompts.INTENT_SYSTEM_PROMPT,
            user_content=prompts.build_intent_user_content(
                text, had_pending_question=bool(session.pending_question)
            ),
            max_tokens=120,
        )

        raw = str(payload.get("intent", "")).strip().casefold()
        try:
            session.intent = IntentType(raw)
        except ValueError:
            state["llm_degraded"] = True
            # Positional fallback: an opening message is a symptom report, a
            # later one answers whatever we just asked.
            session.intent = (
                IntentType.FOLLOWUP_ANSWER
                if session.followup_rounds > 0
                else IntentType.SYMPTOM_REPORT
            )
        return state

    # -- stage 4: entity extraction ----------------------------------------

    async def extract_entities(self, state: IntakeState) -> IntakeState:
        """
        Pull clinical entities from the transcript, then verify every one
        against the patient's actual words before accepting it.
        """
        session = state["session"]
        transcript = session.patient_transcript()
        if not transcript.strip():
            return state

        payload = await self._llm.complete_json(
            system_prompt=prompts.EXTRACTION_SYSTEM_PROMPT,
            user_content=prompts.build_extraction_user_content(transcript),
            max_tokens=1600,
        )

        raw_entities = payload.get("entities")
        if not isinstance(raw_entities, list):
            state["llm_degraded"] = True
            state["notices"].append(
                "Extraction model returned no usable output; no entities were added."
            )
            logger.warning(
                "[INTAKE_EXTRACTION_EMPTY] session=%s", session.session_id
            )
            return state

        candidates: list[ExtractedEntity] = []
        for item in raw_entities[:_MAX_ENTITIES_PER_PASS]:
            if not isinstance(item, dict):
                continue
            try:
                kind = EntityKind(str(item.get("kind", "")).strip().casefold())
            except ValueError:
                # Unknown category: discard rather than inventing a bucket.
                continue

            value = str(item.get("value", "")).strip()
            if not value or value.casefold() == UNKNOWN_VALUE.casefold():
                continue

            candidates.append(
                ExtractedEntity(
                    kind=kind,
                    value=value,
                    confidence=Confidence(_as_float(item.get("confidence"))),
                    evidence=Evidence(
                        quote=str(item.get("evidence", "")).strip(),
                        turn_index=max(0, len(session.turns) - 1),
                    ),
                )
            )

        kept, rejected = enforce_grounding(candidates, transcript)

        if rejected:
            state["rejected_entities"].extend(rejected)
            state["notices"].append(
                f"{len(rejected)} extraction(s) were discarded for citing text the "
                f"patient never said."
            )
            logger.warning(
                "[INTAKE_UNGROUNDED_DROPPED] session=%s count=%d values=%s",
                session.session_id,
                len(rejected),
                [e.value for e in rejected][:5],
            )

        session.merge_entities(kept)
        logger.info(
            "[INTAKE_EXTRACTED] session=%s kept=%d rejected=%d total=%d",
            session.session_id,
            len(kept),
            len(rejected),
            len(session.entities),
        )
        return state

    # -- stage 5: confidence evaluation ------------------------------------

    async def evaluate_confidence(self, state: IntakeState) -> IntakeState:
        """Log the readiness verdict. The routing decision reads it separately."""
        session = state["session"]
        verdict = evaluate_readiness(session)
        logger.info(
            "[INTAKE_READINESS] session=%s ready=%s forced=%s overall=%.2f missing=%s",
            session.session_id,
            verdict.is_ready,
            verdict.forced,
            verdict.overall_confidence.score,
            verdict.missing_labels,
        )
        if verdict.forced:
            state["notices"].append(verdict.reason)
        return state

    # -- stage 6: follow-up question ---------------------------------------

    async def generate_followup(self, state: IntakeState) -> IntakeState:
        """Ask for the single most important missing field."""
        session = state["session"]
        verdict = evaluate_readiness(session)

        payload = await self._llm.complete_json(
            system_prompt=prompts.FOLLOWUP_SYSTEM_PROMPT,
            user_content=prompts.build_followup_user_content(
                missing=[k.value for k in verdict.missing_kinds],
                weak=[k.value for k in verdict.weak_kinds],
                transcript=session.patient_transcript(),
                language=session.language.value,
            ),
            max_tokens=200,
        )

        question = str(payload.get("question", "")).strip()
        if not question:
            state["llm_degraded"] = True
            question = self._fallback_question(verdict.missing_kinds)

        session.pending_question = question
        session.followup_rounds += 1
        session.status = SessionStatus.COLLECTING
        session.add_turn(TurnRole.AGENT, question)

        logger.info(
            "[INTAKE_FOLLOWUP] session=%s round=%d",
            session.session_id,
            session.followup_rounds,
        )
        return state

    @staticmethod
    def _fallback_question(missing: tuple[EntityKind, ...]) -> str:
        """Deterministic question used when the model cannot supply one."""
        first = missing[0] if missing else EntityKind.SYMPTOM
        return {
            EntityKind.SYMPTOM: "Could you describe what you are feeling right now?",
            EntityKind.DURATION: "How long have you had this problem?",
            EntityKind.SEVERITY: "How severe is it on a scale of 1 to 10?",
        }.get(first, "Could you tell me a little more about your symptoms?")

    # -- stage 7: case generation ------------------------------------------

    async def generate_case(self, state: IntakeState) -> IntakeState:
        """
        Assemble the structured case.

        Deterministic fields are built directly from grounded entities; the model
        contributes only narrative (chief complaint phrasing, considerations to
        rule out, handover summary). It is never the source of a clinical fact.
        """
        session = state["session"]
        verdict = evaluate_readiness(session)

        symptoms = session.values_of(EntityKind.SYMPTOM)
        entities_payload = json.dumps(
            [
                {
                    "kind": e.kind.value,
                    "value": e.value,
                    "confidence": e.confidence.score,
                }
                for e in session.entities
                if not e.is_unknown
            ],
            ensure_ascii=False,
        )

        payload = await self._llm.complete_json(
            system_prompt=prompts.CASE_SYSTEM_PROMPT,
            user_content=prompts.build_case_user_content(
                entities_json=entities_payload, red_flags=session.red_flags
            ),
            max_tokens=1200,
        )
        if not payload:
            state["llm_degraded"] = True
            state["notices"].append(
                "Case narrative model was unavailable; case built from verified "
                "entities only."
            )

        chief = str(payload.get("chief_complaint", "")).strip()
        if not chief:
            chief = symptoms[0] if symptoms else "Unspecified health concern"

        # Urgency: take the more severe of the model's read and our own
        # deterministic red-flag assessment. The model may not lower it.
        urgency = UrgencyLevel.CRITICAL if session.red_flags else UrgencyLevel.MEDIUM
        try:
            proposed = UrgencyLevel(str(payload.get("urgency", "")).strip().casefold())
            if proposed.rank > urgency.rank:
                urgency = proposed
        except ValueError:
            pass

        missing_info = [k.value for k in verdict.missing_kinds] + [
            k.value for k in verdict.weak_kinds
        ]

        case = MedicalCase(
            chief_complaint=chief,
            symptoms=symptoms,
            duration=self._first_or_unknown(session, EntityKind.DURATION),
            severity=self._first_or_unknown(session, EntityKind.SEVERITY),
            onset=self._first_or_unknown(session, EntityKind.ONSET),
            body_sites=session.values_of(EntityKind.BODY_SITE),
            allergies=session.values_of(EntityKind.ALLERGY),
            current_medications=session.values_of(EntityKind.MEDICATION),
            medical_history=session.values_of(EntityKind.MEDICAL_HISTORY),
            aggravating_factors=session.values_of(EntityKind.AGGRAVATING_FACTOR),
            relieving_factors=session.values_of(EntityKind.RELIEVING_FACTOR),
            urgency=urgency,
            red_flags=list(session.red_flags),
            differential_considerations=[
                str(d).strip()
                for d in (payload.get("differential_considerations") or [])
                if str(d).strip()
            ][:3],
            missing_information=missing_info,
            recommended_specialty=canonicalise_specialty(
                payload.get("recommended_specialty")
            ),
            overall_confidence=verdict.overall_confidence,
            patient_language=session.language,
            summary_for_doctor=str(payload.get("summary_for_doctor", "")).strip()
            or self._fallback_summary(chief, symptoms),
            generated_at=session.updated_at,
        )

        session.medical_case = case
        session.touch()
        logger.info(
            "[INTAKE_CASE_GENERATED] session=%s specialty=%s urgency=%s confidence=%.2f",
            session.session_id,
            case.recommended_specialty,
            case.urgency.value,
            case.overall_confidence.score,
        )
        return state

    @staticmethod
    def _first_or_unknown(session, kind: EntityKind) -> str:
        values = session.values_of(kind)
        return values[0] if values else UNKNOWN_VALUE

    @staticmethod
    def _fallback_summary(chief: str, symptoms: list[str]) -> str:
        listed = ", ".join(symptoms) if symptoms else "no specific symptoms recorded"
        return (
            f"Patient reports: {chief}. Recorded findings: {listed}. "
            f"This summary was assembled from verified patient statements only and "
            f"requires clinician review."
        )

    # -- stage 8: specialist recommendation --------------------------------

    async def recommend_specialist(self, state: IntakeState) -> IntakeState:
        """
        Match the case to real, available clinicians.

        The specialty is canonicalised before it reaches the directory, and the
        directory is queried for genuine candidates — no placeholder doctors.
        """
        session = state["session"]
        case = session.medical_case
        if case is None:
            return state

        doctors: DoctorDirectoryPort | None = state.get("doctors")
        if doctors is None:
            logger.error(
                "[INTAKE_NO_DIRECTORY] session=%s — cannot recommend specialists",
                session.session_id,
            )
            session.status = SessionStatus.AWAITING_DOCTOR_SELECTION
            return state

        specialty = canonicalise_specialty(case.recommended_specialty)
        candidates = await doctors.find_for_specialty(specialty, limit=3)

        # Nobody in that specialty: fall back to general medicine rather than
        # leaving the patient with an empty list.
        if not candidates and specialty != GENERAL_MEDICINE:
            state["notices"].append(
                f"No {specialty} clinician was available; showing General Medicine."
            )
            candidates = await doctors.find_for_specialty(GENERAL_MEDICINE, limit=3)

        session.recommendations = [
            self._to_recommendation(doctor, specialty, case.urgency, rank)
            for rank, doctor in enumerate(candidates)
        ]
        session.status = SessionStatus.AWAITING_DOCTOR_SELECTION
        session.pending_question = None
        session.touch()

        if not session.recommendations:
            state["notices"].append(
                "No clinicians are currently registered to receive this case."
            )
            logger.warning(
                "[INTAKE_NO_DOCTORS] session=%s specialty=%s",
                session.session_id,
                specialty,
            )

        logger.info(
            "[INTAKE_RECOMMENDED] session=%s specialty=%s candidates=%d",
            session.session_id,
            specialty,
            len(session.recommendations),
        )
        return state

    @staticmethod
    def _to_recommendation(
        doctor: DoctorRef, specialty: str, urgency: UrgencyLevel, rank: int
    ) -> SpecialistRecommendation:
        """
        Score a candidate.

        Deliberately transparent and deterministic: specialty match dominates,
        with verification, rating, experience and availability as tie-breakers,
        minus a small ordering penalty. A clinician can read this score and
        understand it.
        """
        score = 60.0
        if doctor.specialty.casefold() == specialty.casefold():
            score += 20.0
        if doctor.is_verified:
            score += 6.0
        score += min(doctor.rating, 5.0) * 2.0
        score += min(doctor.years_of_experience, 20) * 0.2
        if doctor.is_available:
            score += 4.0
        score -= rank * 1.5

        reasons = [f"Specialises in {doctor.specialty}"]
        if doctor.is_verified:
            reasons.append("credentials verified")
        if doctor.years_of_experience:
            reasons.append(f"{doctor.years_of_experience} years of experience")
        if doctor.is_available:
            reasons.append("currently available")
        if urgency.rank >= UrgencyLevel.HIGH.rank:
            reasons.append("able to take an urgent case")

        return SpecialistRecommendation(
            specialty=doctor.specialty,
            rationale=", ".join(reasons).capitalize() + ".",
            match_score=round(min(score, 99.0), 1),
            doctor_id=doctor.doctor_id,
            doctor_name=doctor.full_name,
            hospital_name=doctor.hospital_name,
            is_available=doctor.is_available,
            avatar_url=doctor.avatar_url,
        )

    # -- emergency short-circuit -------------------------------------------

    async def escalate_emergency(self, state: IntakeState) -> IntakeState:
        """
        Terminate intake immediately with emergency guidance.

        No follow-up questions: a patient describing crushing chest pain must not
        be asked to rate their symptoms on a scale before being told to seek
        help. A minimal case is still produced so an on-call clinician has
        context if the patient also reaches the platform.
        """
        session = state["session"]
        symptoms = session.values_of(EntityKind.SYMPTOM)
        transcript = session.patient_transcript()

        session.medical_case = MedicalCase(
            chief_complaint=symptoms[0]
            if symptoms
            else (transcript[:120] or "Possible medical emergency"),
            symptoms=symptoms,
            duration=self._first_or_unknown(session, EntityKind.DURATION),
            severity=self._first_or_unknown(session, EntityKind.SEVERITY),
            urgency=UrgencyLevel.CRITICAL,
            red_flags=list(session.red_flags),
            missing_information=["Intake halted for emergency escalation."],
            recommended_specialty=EMERGENCY_MEDICINE,
            overall_confidence=Confidence(1.0),
            patient_language=session.language,
            summary_for_doctor=(
                "EMERGENCY ESCALATION. Automated red-flag screening matched: "
                + "; ".join(session.red_flags)
                + ". Patient was advised to seek immediate emergency care. "
                "Intake was stopped and no follow-up questions were asked."
            ),
        )

        session.status = SessionStatus.EMERGENCY_ESCALATED
        session.pending_question = None
        session.recommendations = []
        session.add_turn(TurnRole.AGENT, _EMERGENCY_GUIDANCE)
        state["notices"].append(_EMERGENCY_GUIDANCE)
        session.touch()

        logger.warning(
            "[INTAKE_EMERGENCY_ESCALATED] session=%s flags=%s",
            session.session_id,
            session.red_flags,
        )
        return state
