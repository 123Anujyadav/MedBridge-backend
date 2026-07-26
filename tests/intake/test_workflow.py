"""
Workflow tests for the Medical Case Intake Agent.

Drives the compiled LangGraph with a scripted model, covering every input class
in the requirements: English, Hindi, Hinglish, mixed script, typos, emergencies,
incomplete descriptions, multiple simultaneous symptoms, and total model failure.
"""

from __future__ import annotations

import pytest

from app.intake.domain.enums import (
    EntityKind,
    IntentType,
    Language,
    SessionStatus,
    TurnRole,
    UrgencyLevel,
)
from app.intake.domain.policies import MAX_FOLLOWUP_ROUNDS
from app.intake.workflow.graph import LangGraphIntakeWorkflow
from tests.intake.conftest import (
    COMPLETE_TEXT,
    DeadLLM,
    FakeDoctorDirectory,
    ScriptedLLM,
    complete_extraction,
    extraction,
    make_session,
)

pytestmark = pytest.mark.asyncio


def _workflow(llm, doctors=None) -> LangGraphIntakeWorkflow:
    return LangGraphIntakeWorkflow(llm=llm, doctors=doctors or FakeDoctorDirectory())


# ------------------------------------------------------- Happy path / English


class TestCompleteEnglishIntake:
    async def test_generates_case_and_recommends_specialist(self):
        llm = ScriptedLLM(
            {
                "extraction": complete_extraction(),
                "case": {
                    "chief_complaint": "Chest discomfort on exertion",
                    "differential_considerations": ["Stable angina"],
                    "recommended_specialty": "heart doctor",
                    "urgency": "high",
                    "summary_for_doctor": "Handover summary.",
                },
            }
        )
        result = await _workflow(llm).run_detailed(make_session())

        session = result.session
        assert session.status is SessionStatus.AWAITING_DOCTOR_SELECTION
        assert session.pending_question is None

        case = session.medical_case
        assert case is not None
        assert case.symptoms == ["chest discomfort"]
        assert case.duration == "3 days"
        # "heart doctor" must be canonicalised before it reaches persistence.
        assert case.recommended_specialty == "Cardiology"
        assert case.urgency is UrgencyLevel.HIGH
        assert case.differential_considerations == ["Stable angina"]

    async def test_recommendations_reference_real_doctors(self):
        doctors = FakeDoctorDirectory()
        llm = ScriptedLLM(
            {
                "extraction": complete_extraction(),
                "case": {"recommended_specialty": "Cardiology", "urgency": "medium"},
            }
        )
        result = await _workflow(llm, doctors).run_detailed(make_session())

        recs = result.session.recommendations
        assert len(recs) == 1
        assert recs[0].doctor_id == doctors.doctors[0].doctor_id
        assert recs[0].specialty == "Cardiology"
        assert 0 < recs[0].match_score <= 99.0
        assert "Cardiology" in doctors.queried

    async def test_falls_back_to_general_medicine_when_specialty_empty(self):
        doctors = FakeDoctorDirectory(doctors=[])
        llm = ScriptedLLM(
            {
                "extraction": complete_extraction(),
                "case": {"recommended_specialty": "Cardiology", "urgency": "low"},
            }
        )
        result = await _workflow(llm, doctors).run_detailed(make_session())

        assert doctors.queried == ["Cardiology", "General Medicine"]
        assert result.session.status is SessionStatus.AWAITING_DOCTOR_SELECTION
        assert any("General Medicine" in n for n in result.notices)


# ------------------------------------------------------------- Language cases


class TestLanguageHandling:
    async def test_hindi_devanagari_is_detected(self):
        text = "मुझे तीन दिन से बहुत तेज़ सिरदर्द है"
        llm = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "headache", 0.9, "सिरदर्द"),
                    ("duration", "three days", 0.9, "तीन दिन"),
                    ("severity", "severe", 0.85, "बहुत तेज़"),
                )
            }
        )
        result = await _workflow(llm).run_detailed(make_session(text))

        assert result.session.language is Language.HINDI
        assert result.session.medical_case.patient_language is Language.HINDI

    async def test_hinglish_is_detected_without_model_call(self):
        text = "mujhe bahut tez sar dard hai aur bukhar bhi hai"
        llm = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "headache", 0.9, "sar dard"),
                    ("symptom", "fever", 0.88, "bukhar"),
                    ("duration", "2 days", 0.8, "mujhe bahut"),
                    ("severity", "severe", 0.8, "bahut tez"),
                )
            }
        )
        result = await _workflow(llm).run_detailed(make_session(text))

        assert result.session.language is Language.HINGLISH
        # Script heuristics settle this; no language call should be needed.
        assert "language" not in llm.calls

    async def test_mixed_script_is_detected(self):
        text = "I have सिरदर्द since morning and feeling weak"
        llm = ScriptedLLM(
            {"extraction": extraction(("symptom", "headache", 0.9, "सिरदर्द"))}
        )
        result = await _workflow(llm).run_detailed(make_session(text))
        assert result.session.language is Language.MIXED

    async def test_evidence_grounding_works_on_devanagari(self):
        """A Devanagari quote genuinely present must survive grounding."""
        text = "मुझे सिरदर्द है"
        llm = ScriptedLLM(
            {"extraction": extraction(("symptom", "headache", 0.9, "सिरदर्द"))}
        )
        result = await _workflow(llm).run_detailed(make_session(text))
        assert result.rejected_count == 0
        assert result.session.values_of(EntityKind.SYMPTOM) == ["headache"]


# --------------------------------------------------------------------- Typos


class TestTypoTolerance:
    async def test_typo_ridden_input_still_extracts(self):
        text = "i hav ben havng chst discomfrt for 3 dayz its modrate"
        llm = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "chest discomfort", 0.82, "chst discomfrt"),
                    ("duration", "3 days", 0.85, "for 3 dayz"),
                    ("severity", "moderate", 0.75, "modrate"),
                )
            }
        )
        result = await _workflow(llm).run_detailed(make_session(text))

        assert result.rejected_count == 0
        assert result.session.status is SessionStatus.AWAITING_DOCTOR_SELECTION
        assert result.session.medical_case.symptoms == ["chest discomfort"]


# ----------------------------------------------------------------- Emergency


class TestEmergencyEscalation:
    EMERGENCY_TEXT = "I have crushing chest pain and I cannot breathe"

    async def test_short_circuits_to_escalation(self):
        llm = ScriptedLLM({"extraction": complete_extraction()})
        result = await _workflow(llm).run_detailed(make_session(self.EMERGENCY_TEXT))
        session = result.session

        assert session.status is SessionStatus.EMERGENCY_ESCALATED
        assert session.intent is IntentType.EMERGENCY
        assert session.medical_case.urgency is UrgencyLevel.CRITICAL
        assert len(session.red_flags) == 2

    async def test_never_asks_followup_questions(self):
        """A patient describing an emergency must not be interrogated."""
        llm = ScriptedLLM({"extraction": {"entities": []}})
        result = await _workflow(llm).run_detailed(make_session(self.EMERGENCY_TEXT))

        assert result.session.pending_question is None
        assert result.session.followup_rounds == 0
        assert "followup" not in llm.calls
        assert "extraction" not in llm.calls

    async def test_surfaces_emergency_guidance_to_patient(self):
        llm = ScriptedLLM()
        result = await _workflow(llm).run_detailed(make_session(self.EMERGENCY_TEXT))

        agent_turns = [
            t.text for t in result.session.turns if t.role == TurnRole.AGENT
        ]
        assert any("emergency" in t.casefold() for t in agent_turns)
        assert any("emergency" in n.casefold() for n in result.notices)

    async def test_escalates_even_when_model_is_dead(self):
        """Red-flag detection is deterministic and must not depend on the LLM."""
        result = await _workflow(DeadLLM()).run_detailed(
            make_session(self.EMERGENCY_TEXT)
        )
        assert result.session.status is SessionStatus.EMERGENCY_ESCALATED
        assert result.session.medical_case.urgency is UrgencyLevel.CRITICAL

    async def test_hinglish_emergency_escalates(self):
        result = await _workflow(ScriptedLLM()).run_detailed(
            make_session("mujhe seene me dard hai aur saans nahi aa raha")
        )
        assert result.session.status is SessionStatus.EMERGENCY_ESCALATED

    async def test_model_cannot_downgrade_red_flag_urgency(self):
        """Even if the model says 'low', red flags force critical."""
        llm = ScriptedLLM(
            {
                "extraction": complete_extraction(),
                "case": {"urgency": "low", "recommended_specialty": "Cardiology"},
            }
        )
        result = await _workflow(llm).run_detailed(make_session(self.EMERGENCY_TEXT))
        assert result.session.medical_case.urgency is UrgencyLevel.CRITICAL


# ------------------------------------------------------ Incomplete input flow


class TestIncompleteDescription:
    async def test_asks_followup_when_information_is_missing(self):
        llm = ScriptedLLM(
            {"extraction": extraction(("symptom", "malaise", 0.6, "I feel unwell"))}
        )
        result = await _workflow(llm).run_detailed(make_session("I feel unwell"))
        session = result.session

        assert session.status is SessionStatus.COLLECTING
        assert session.pending_question == "How long have you been feeling this way?"
        assert session.followup_rounds == 1
        assert session.medical_case is None

    async def test_followup_question_recorded_as_agent_turn(self):
        llm = ScriptedLLM(
            {"extraction": extraction(("symptom", "malaise", 0.6, "I feel unwell"))}
        )
        result = await _workflow(llm).run_detailed(make_session("I feel unwell"))

        agent_turns = [t for t in result.session.turns if t.role == TurnRole.AGENT]
        assert len(agent_turns) == 1
        # Agent text must stay out of the evidence transcript.
        assert agent_turns[0].text not in result.session.patient_transcript()

    async def test_multi_turn_conversation_completes(self):
        """Second turn supplies the missing fields and the case is generated."""
        workflow = _workflow(
            ScriptedLLM(
                {"extraction": extraction(("symptom", "malaise", 0.6, "I feel unwell"))}
            )
        )
        session = await workflow.run_detailed(make_session("I feel unwell"))
        session = session.session
        assert session.status is SessionStatus.COLLECTING

        session.add_turn(TurnRole.PATIENT, "about 3 days now and it is moderate")
        session.pending_question = None

        completing = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "malaise", 0.8, "I feel unwell"),
                    ("duration", "3 days", 0.9, "about 3 days now"),
                    ("severity", "moderate", 0.85, "it is moderate"),
                ),
                "case": {"recommended_specialty": "Cardiology", "urgency": "medium"},
            }
        )
        result = await _workflow(completing).run_detailed(session)

        assert result.session.status is SessionStatus.AWAITING_DOCTOR_SELECTION
        assert result.session.medical_case.duration == "3 days"

    async def test_stops_asking_after_max_rounds_and_marks_unknown(self):
        """
        The agent must not interrogate indefinitely. Once the follow-up budget is
        spent it generates the case with gaps marked Unknown rather than guessing.
        """
        session = make_session("I feel unwell")
        session.followup_rounds = MAX_FOLLOWUP_ROUNDS

        llm = ScriptedLLM(
            {"extraction": extraction(("symptom", "malaise", 0.6, "I feel unwell"))}
        )
        result = await _workflow(llm).run_detailed(session)

        case = result.session.medical_case
        assert result.session.status is SessionStatus.AWAITING_DOCTOR_SELECTION
        assert case is not None
        assert case.duration == "Unknown"
        assert case.severity == "Unknown"
        assert "duration" in case.missing_information


# ------------------------------------------------------- Multiple symptoms


class TestMultipleSymptoms:
    async def test_captures_all_simultaneous_symptoms(self):
        text = (
            "For the past 5 days I have had a severe headache, high fever, "
            "a persistent cough and nausea"
        )
        llm = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "headache", 0.94, "severe headache"),
                    ("symptom", "fever", 0.93, "high fever"),
                    ("symptom", "cough", 0.9, "a persistent cough"),
                    ("symptom", "nausea", 0.88, "nausea"),
                    ("duration", "5 days", 0.92, "For the past 5 days"),
                    ("severity", "severe", 0.9, "severe headache"),
                )
            }
        )
        result = await _workflow(llm).run_detailed(make_session(text))
        case = result.session.medical_case

        assert set(case.symptoms) == {"headache", "fever", "cough", "nausea"}
        assert case.duration == "5 days"

    async def test_deduplicates_repeated_symptoms(self):
        llm = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "chest discomfort", 0.8, "chest discomfort"),
                    ("symptom", "Chest Discomfort", 0.93, "chest discomfort"),
                    ("duration", "3 days", 0.9, "for 3 days"),
                    ("severity", "moderate", 0.85, "moderate"),
                )
            }
        )
        result = await _workflow(llm).run_detailed(make_session())
        assert len(result.session.medical_case.symptoms) == 1


# ------------------------------------------------- Fabrication / degradation


class TestSafetyAndDegradation:
    async def test_rejects_fabricated_allergy(self):
        """
        The model invents a penicillin allergy the patient never mentioned. It
        must not reach the case.
        """
        llm = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "chest discomfort", 0.93, "chest discomfort"),
                    ("duration", "3 days", 0.91, "for 3 days"),
                    ("severity", "moderate", 0.85, "moderate"),
                    (
                        "allergy",
                        "penicillin",
                        0.99,
                        "I am severely allergic to penicillin",
                    ),
                )
            }
        )
        result = await _workflow(llm).run_detailed(make_session())

        assert result.rejected_count == 1
        assert result.session.medical_case.allergies == []
        assert any("never said" in n for n in result.notices)

    async def test_keeps_real_allergy_alongside_rejecting_fake(self):
        text = COMPLETE_TEXT + " I am allergic to ibuprofen."
        llm = ScriptedLLM(
            {
                "extraction": extraction(
                    ("symptom", "chest discomfort", 0.93, "chest discomfort"),
                    ("duration", "3 days", 0.91, "for 3 days"),
                    ("severity", "moderate", 0.85, "moderate"),
                    ("allergy", "ibuprofen", 0.95, "allergic to ibuprofen"),
                    ("allergy", "penicillin", 0.99, "severely allergic to penicillin"),
                )
            }
        )
        result = await _workflow(llm).run_detailed(make_session(text))

        assert result.session.medical_case.allergies == ["ibuprofen"]
        assert result.rejected_count == 1

    async def test_discards_unrecognised_entity_kinds(self):
        llm = ScriptedLLM(
            {
                "extraction": {
                    "entities": [
                        {
                            "kind": "astrological_sign",
                            "value": "Leo",
                            "confidence": 0.99,
                            "evidence": "chest discomfort",
                        },
                        {
                            "kind": "symptom",
                            "value": "chest discomfort",
                            "confidence": 0.9,
                            "evidence": "chest discomfort",
                        },
                    ]
                }
            }
        )
        result = await _workflow(llm).run_detailed(make_session())
        kinds = {e.kind for e in result.session.entities}
        assert kinds == {EntityKind.SYMPTOM}

    async def test_dead_model_degrades_to_deterministic_followup(self):
        result = await _workflow(DeadLLM()).run_detailed(make_session("my head hurts"))

        assert result.degraded is True
        assert result.session.status is SessionStatus.COLLECTING
        assert result.session.pending_question  # deterministic fallback text
        assert result.session.medical_case is None

    async def test_model_exception_does_not_crash_the_turn(self):
        llm = ScriptedLLM({"extraction": RuntimeError("provider exploded")})
        result = await _workflow(llm).run_detailed(make_session())

        # The graph contains the failure and leaves the session usable.
        assert result.session.status is not SessionStatus.ROUTED
        assert result.degraded is True

    async def test_malformed_confidence_is_coerced_not_crashed(self):
        llm = ScriptedLLM(
            {
                "extraction": {
                    "entities": [
                        {
                            "kind": "symptom",
                            "value": "chest discomfort",
                            "confidence": "not-a-number",
                            "evidence": "chest discomfort",
                        }
                    ]
                }
            }
        )
        result = await _workflow(llm).run_detailed(make_session())
        # Unparseable confidence becomes 0.0, so the entity cannot satisfy a
        # mandatory field and the agent asks instead of assuming.
        assert result.session.status is SessionStatus.COLLECTING

    async def test_empty_transcript_does_not_call_extraction(self):
        from app.intake.domain.entities import IntakeSession

        llm = ScriptedLLM()
        result = await _workflow(llm).run_detailed(IntakeSession(patient_user_id="u"))
        assert "extraction" not in llm.calls
        assert result.session.status is SessionStatus.COLLECTING
