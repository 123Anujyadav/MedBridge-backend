"""
Unit tests for the intake domain layer.

Covers the three safety mechanisms directly: evidence grounding, multilingual
red-flag detection, and the confidence readiness gate. Pure functions, no I/O.
"""

from __future__ import annotations

import pytest

from app.intake.domain.entities import (
    UNKNOWN_VALUE,
    ExtractedEntity,
    IntakeSession,
    MedicalCase,
)
from app.intake.domain.enums import (
    ConfidenceBand,
    EntityKind,
    SessionStatus,
    TurnRole,
    UrgencyLevel,
)
from app.intake.domain.policies import (
    MAX_FOLLOWUP_ROUNDS,
    detect_red_flags,
    enforce_grounding,
    evaluate_readiness,
    is_evidence_grounded,
    normalise,
)
from app.intake.domain.specialties import canonicalise_specialty
from app.intake.domain.value_objects import Confidence, Evidence


# ---------------------------------------------------------------- Confidence


class TestConfidence:
    @pytest.mark.parametrize(
        "raw,expected", [(-5.0, 0.0), (0.5, 0.5), (1.7, 1.0), (0.0, 0.0)]
    )
    def test_clamps_out_of_range_scores(self, raw, expected):
        assert Confidence(raw).score == expected

    @pytest.mark.parametrize(
        "score,band",
        [
            (0.0, ConfidenceBand.UNKNOWN),
            (0.30, ConfidenceBand.LOW),
            (0.60, ConfidenceBand.MEDIUM),
            (0.95, ConfidenceBand.HIGH),
        ],
    )
    def test_band_classification(self, score, band):
        assert Confidence(score).band is band

    def test_penalised_reduces_score(self):
        assert Confidence(0.8).penalised(0.5).score == 0.4


# ------------------------------------------------------------ Red flag rules


class TestRedFlagDetection:
    @pytest.mark.parametrize(
        "text,expected_fragment",
        [
            ("I have crushing chest pain", "coronary"),
            ("mujhe seene me dard ho raha hai", "coronary"),
            ("मुझे सीने में दर्द है", "coronary"),
            ("I cannot breathe properly", "Respiratory"),
            ("saans nahi aa raha hai", "Respiratory"),
            ("सांस नहीं आ रही", "Respiratory"),
            ("he is unconscious and unresponsive", "consciousness"),
            ("wo behosh ho gaya", "consciousness"),
            ("I want to kill myself", "Suicidal"),
            ("khudkushi karna chahta hoon", "Suicidal"),
            ("she is having a seizure", "Seizure"),
            ("my throat is closing up", "anaphylaxis"),
            ("my tongue is swelling", "anaphylaxis"),
            ("my chest hurts really badly", "coronary"),
            ("there is pressure in my chest", "coronary"),
            ("I am having trouble breathing", "Respiratory"),
            ("I am short of breath", "Respiratory"),
        ],
    )
    def test_detects_emergencies_across_languages(self, text, expected_fragment):
        flags = detect_red_flags(text)
        assert any(expected_fragment.casefold() in f.casefold() for f in flags), (
            f"expected a flag containing {expected_fragment!r} for {text!r}, got {flags}"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "I have a mild cough and a runny nose",
            "my knee has been sore after running",
            "mujhe halka bukhar hai",
            "",
            "   ",
        ],
    )
    def test_no_false_positives_on_benign_text(self, text):
        assert detect_red_flags(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "I have a headache but no chest pain",
            "there is not any chest pain",
            "the doctor denies chest pain",
        ],
    )
    def test_leading_negation_suppresses_match(self, text):
        assert detect_red_flags(text) == []

    def test_trailing_hindi_negation_is_not_suppressed(self):
        """
        'saans nahi aa raha' means 'I cannot breathe' — the negation trails the
        noun and is itself the emergency signal. It must not be filtered out.
        """
        assert detect_red_flags("saans nahi aa raha hai") != []

    def test_deduplicates_repeated_flags(self):
        flags = detect_red_flags("chest pain, chest pain, and more chest pain")
        assert len(flags) == 1


# ------------------------------------------------- Evidence grounding rules


class TestEvidenceGrounding:
    TRANSCRIPT = (
        "I have had a bad headache for three days and I am allergic to ibuprofen"
    )

    def test_verbatim_quote_is_fully_grounded(self):
        grounded, ratio = is_evidence_grounded("bad headache", self.TRANSCRIPT)
        assert grounded is True
        assert ratio == 1.0

    def test_grounding_ignores_case_and_punctuation(self):
        grounded, ratio = is_evidence_grounded("BAD HEADACHE!!!", self.TRANSCRIPT)
        assert grounded is True
        assert ratio == 1.0

    def test_fabricated_quote_is_rejected(self):
        grounded, ratio = is_evidence_grounded(
            "patient reports severe chest pain radiating to the left arm",
            self.TRANSCRIPT,
        )
        assert grounded is False
        assert ratio < 0.7

    @pytest.mark.parametrize("quote", ["", "   ", None])
    def test_empty_quote_is_never_grounded(self, quote):
        grounded, ratio = is_evidence_grounded(quote, self.TRANSCRIPT)
        assert grounded is False
        assert ratio == 0.0

    def test_empty_transcript_grounds_nothing(self):
        assert is_evidence_grounded("headache", "") == (False, 0.0)

    def test_enforce_grounding_drops_only_fabrications(self):
        real = ExtractedEntity(
            EntityKind.ALLERGY,
            "ibuprofen",
            Confidence(0.9),
            Evidence("allergic to ibuprofen"),
        )
        fake = ExtractedEntity(
            EntityKind.ALLERGY,
            "penicillin",
            Confidence(0.99),
            Evidence("I am severely allergic to penicillin"),
        )

        kept, rejected = enforce_grounding([real, fake], self.TRANSCRIPT)

        assert [e.value for e in kept] == ["ibuprofen"]
        assert [e.value for e in rejected] == ["penicillin"]

    def test_unknown_entities_bypass_grounding(self):
        unknown = ExtractedEntity.unknown(EntityKind.ALLERGY)
        kept, rejected = enforce_grounding([unknown], self.TRANSCRIPT)
        assert kept == [unknown]
        assert rejected == []

    def test_partial_grounding_reduces_confidence(self):
        partial = ExtractedEntity(
            EntityKind.SYMPTOM,
            "headache",
            Confidence(0.9),
            Evidence("a bad headache for three whole days"),
        )
        kept, rejected = enforce_grounding([partial], self.TRANSCRIPT)
        assert rejected == []
        assert kept[0].confidence.score < 0.9

    def test_normalise_collapses_whitespace_and_punctuation(self):
        assert normalise("  Hello,   WORLD!!  ") == "hello world"


# ------------------------------------------------------------- Readiness gate


class TestReadiness:
    @staticmethod
    def _session_with(*entities: ExtractedEntity) -> IntakeSession:
        session = IntakeSession(patient_user_id="u1")
        session.add_turn(TurnRole.PATIENT, "some description")
        session.merge_entities(list(entities))
        return session

    def test_empty_session_is_not_ready(self):
        verdict = evaluate_readiness(IntakeSession())
        assert verdict.is_ready is False
        assert set(verdict.missing_labels) == {"symptom", "duration", "severity"}

    def test_all_mandatory_fields_present_is_ready(self):
        verdict = evaluate_readiness(
            self._session_with(
                ExtractedEntity(EntityKind.SYMPTOM, "cough", Confidence(0.9), Evidence("cough")),
                ExtractedEntity(EntityKind.DURATION, "2 days", Confidence(0.9), Evidence("2 days")),
                ExtractedEntity(EntityKind.SEVERITY, "mild", Confidence(0.8), Evidence("mild")),
            )
        )
        assert verdict.is_ready is True
        assert verdict.forced is False

    def test_missing_duration_blocks_readiness(self):
        verdict = evaluate_readiness(
            self._session_with(
                ExtractedEntity(EntityKind.SYMPTOM, "cough", Confidence(0.9), Evidence("cough")),
                ExtractedEntity(EntityKind.SEVERITY, "mild", Confidence(0.8), Evidence("mild")),
            )
        )
        assert verdict.is_ready is False
        assert "duration" in verdict.missing_labels

    def test_low_confidence_entity_does_not_satisfy_requirement(self):
        verdict = evaluate_readiness(
            self._session_with(
                ExtractedEntity(EntityKind.SYMPTOM, "cough", Confidence(0.2), Evidence("cough")),
                ExtractedEntity(EntityKind.DURATION, "2 days", Confidence(0.9), Evidence("2 days")),
                ExtractedEntity(EntityKind.SEVERITY, "mild", Confidence(0.9), Evidence("mild")),
            )
        )
        assert verdict.is_ready is False

    def test_exhausted_followups_forces_readiness(self):
        session = IntakeSession()
        session.followup_rounds = MAX_FOLLOWUP_ROUNDS
        verdict = evaluate_readiness(session)
        assert verdict.is_ready is True
        assert verdict.forced is True
        assert verdict.missing_labels  # gaps recorded rather than invented


# -------------------------------------------------------------- Entity model


class TestEntitiesAndSession:
    def test_merge_keeps_higher_confidence_duplicate(self):
        session = IntakeSession()
        low = ExtractedEntity(EntityKind.SYMPTOM, "cough", Confidence(0.5), Evidence("cough"))
        high = ExtractedEntity(EntityKind.SYMPTOM, "Cough", Confidence(0.95), Evidence("cough"))

        session.merge_entities([low])
        session.merge_entities([high])

        assert len(session.entities) == 1
        assert session.entities[0].confidence.score == 0.95

    def test_transcript_excludes_agent_turns(self):
        session = IntakeSession()
        session.add_turn(TurnRole.PATIENT, "I have a fever")
        session.add_turn(TurnRole.AGENT, "How long have you had it?")
        session.add_turn(TurnRole.PATIENT, "two days")

        transcript = session.patient_transcript()
        assert "fever" in transcript
        assert "two days" in transcript
        assert "How long" not in transcript

    def test_unknown_entity_is_flagged(self):
        entity = ExtractedEntity.unknown(EntityKind.ALLERGY)
        assert entity.is_unknown is True
        assert entity.value == UNKNOWN_VALUE

    def test_session_round_trips_through_dict(self):
        session = IntakeSession(patient_user_id="u9")
        session.add_turn(TurnRole.PATIENT, "chest pain")
        session.merge_entities(
            [ExtractedEntity(EntityKind.SYMPTOM, "chest pain", Confidence(0.9), Evidence("chest pain"))]
        )
        session.medical_case = MedicalCase(chief_complaint="Chest pain")
        session.status = SessionStatus.AWAITING_DOCTOR_SELECTION

        restored = IntakeSession.from_dict(session.to_dict())

        assert restored.session_id == session.session_id
        assert restored.patient_user_id == "u9"
        assert restored.status is SessionStatus.AWAITING_DOCTOR_SELECTION
        assert restored.entities[0].value == "chest pain"
        assert restored.medical_case.chief_complaint == "Chest pain"

    def test_urgency_ranking(self):
        assert UrgencyLevel.CRITICAL.rank > UrgencyLevel.HIGH.rank
        assert UrgencyLevel.LOW.rank < UrgencyLevel.MEDIUM.rank

    def test_terminal_statuses(self):
        assert SessionStatus.ROUTED.is_terminal is True
        assert SessionStatus.EMERGENCY_ESCALATED.is_terminal is True
        assert SessionStatus.COLLECTING.is_terminal is False


# -------------------------------------------------------------- Specialties


class TestSpecialtyCanonicalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Cardiology", "Cardiology"),
            ("cardiology", "Cardiology"),
            ("heart doctor", "Cardiology"),
            ("Department of Cardiology", "Cardiology"),
            ("cardiac", "Cardiology"),
            ("ENT", "Otolaryngology"),
            ("skin", "Dermatology"),
            ("mental health", "Psychiatry"),
            ("internal medicine", "General Medicine"),
        ],
    )
    def test_maps_variants_to_canonical_names(self, raw, expected):
        assert canonicalise_specialty(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "Department of Interdimensional Wizardry"])
    def test_unknown_input_falls_back_to_default(self, raw):
        assert canonicalise_specialty(raw) == "General Medicine"
