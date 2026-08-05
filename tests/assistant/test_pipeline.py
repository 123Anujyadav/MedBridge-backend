"""
Pipeline and domain tests for the AI Medical Assistant.

Covers every input class in the brief: English, Hindi, Hinglish, mixed script,
context memory, emergencies, model failure, retrieval failure, and long/short
conversations.
"""

from __future__ import annotations

import pytest

from app.assistant.application.ports import RetrievedSnippet
from app.assistant.domain.entities import AssistantAnswer, Conversation
from app.assistant.domain.enums import (
    EmergencyRisk,
    IntentType,
    KnowledgeSource,
    UrgencyLevel,
)
from app.assistant.domain.language import detect_language
from app.assistant.infrastructure.guardrails import LLMGuardrails
from app.assistant.pipeline import prompts as prompts_module
from app.assistant.pipeline.graph import LangGraphAssistantPipeline
from app.intake.domain.enums import Language
from tests.assistant.conftest import (
    DeadLLM,
    ExplodingRetriever,
    FakeRetriever,
    ScriptedLLM,
    answer_payload,
    entity_payload,
)

pytestmark = pytest.mark.asyncio


def _pipeline(llm, retriever=None) -> LangGraphAssistantPipeline:
    return LangGraphAssistantPipeline(
        llm=llm, retriever=retriever or FakeRetriever()
    )


async def _ask(llm, text: str, conversation: Conversation | None = None, retriever=None):
    conversation = conversation or Conversation(patient_user_id="u1")
    conversation.add_user_message(text)
    answer = await _pipeline(llm, retriever).run(conversation, text)
    return conversation, answer


# ----------------------------------------------------------- Language


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I have a headache", Language.ENGLISH),
            ("मुझे सिरदर्द है", Language.HINDI),
            ("mujhe sar dard hai aur bukhar bhi hai", Language.HINGLISH),
            ("I have सिरदर्द since morning", Language.MIXED),
            ("", Language.UNKNOWN),
        ],
    )
    def test_detects_language(self, text, expected):
        assert detect_language(text) is expected

    async def test_pipeline_records_language(self, scripted_llm):
        _, answer = await _ask(scripted_llm, "mujhe bahut tez sar dard hai")
        assert answer.language is Language.HINGLISH

    async def test_prompt_instructs_model_to_match_language(self, scripted_llm):
        await _ask(scripted_llm, "मुझे सिरदर्द है")
        answer_prompts = [p for p in scripted_llm.prompts if "Devanagari" in p]
        assert answer_prompts, "answer prompt must carry a language directive"


# ----------------------------------------------------------- Structure


class TestStructuredOutput:
    async def test_maps_every_card_the_ui_renders(self, scripted_llm):
        _, answer = await _ask(scripted_llm, "I have a headache")
        ui = answer.to_ui_payload()

        assert ui["summary"]
        assert ui["symptoms"] == ["headache", "nausea"]
        assert ui["causes"][0] == {"cause": "Tension headache", "confidence": "High"}
        assert ui["actions"]
        assert ui["lifestyleAdvice"][0]["category"] == "Hydration"
        # camelCase is the frontend contract, not a style choice.
        assert ui["medicines"][0]["sideEffects"] == ["Nausea"]
        assert ui["medicines"][0]["type"] == "OTC"
        assert ui["specialist"]["priority"] == "Recommended"
        assert ui["urgency"] == {"level": "Low", "explanation": "Routine management."}
        assert ui["followUpQuestions"]

    async def test_omits_sections_with_no_content(self, scripted_llm):
        scripted_llm.responses["answer"] = answer_payload(
            causes=[], medicines=[], lifestyleAdvice=[]
        )
        _, answer = await _ask(scripted_llm, "hello")
        ui = answer.to_ui_payload()
        assert "causes" not in ui
        assert "medicines" not in ui
        assert "lifestyleAdvice" not in ui

    async def test_specialist_is_canonicalised(self, scripted_llm):
        scripted_llm.responses["answer"] = answer_payload(
            specialist={"name": "heart doctor", "reason": "chest symptoms"}
        )
        _, answer = await _ask(scripted_llm, "I have chest tightness on exertion")
        assert answer.specialist.name == "Cardiology"

    async def test_malformed_payload_does_not_crash(self, scripted_llm):
        scripted_llm.responses["answer"] = {
            "summary": "ok",
            "causes": "not-a-list",
            "urgency": "not-a-dict",
            "confidence": "not-a-number",
        }
        _, answer = await _ask(scripted_llm, "I feel unwell")
        assert answer.causes == []
        assert answer.urgency.level is UrgencyLevel.LOW
        assert answer.confidence == 0.7  # coerced default


# ----------------------------------------------------------- Memory


class TestConversationMemory:
    async def test_symptoms_accumulate_across_turns(self, scripted_llm):
        conversation, answer1 = await _ask(scripted_llm, "I have a headache")
        conversation.add_ai_message(answer1)

        scripted_llm.responses["answer"] = answer_payload(
            symptoms=["fever"], followUpQuestions=["Any chills?"]
        )
        conversation.add_user_message("now I also have fever")
        answer2 = await _pipeline(scripted_llm).run(conversation, "now I also have fever")
        conversation.add_ai_message(answer2)

        assert "headache" in conversation.known_symptoms
        assert "fever" in conversation.known_symptoms

    async def test_duplicate_follow_up_questions_are_suppressed(self, scripted_llm):
        conversation, answer1 = await _ask(scripted_llm, "I have a headache")
        conversation.add_ai_message(answer1)
        assert "How long have you had this?" in conversation.asked_questions

        # The model repeats itself; the pipeline must drop the repeat.
        conversation.add_user_message("two days")
        answer2 = await _pipeline(scripted_llm).run(conversation, "two days")

        assert "How long have you had this?" not in answer2.follow_up_questions

    async def test_prior_context_is_sent_to_the_model(self, scripted_llm):
        conversation, answer1 = await _ask(scripted_llm, "I have a headache")
        conversation.add_ai_message(answer1)

        conversation.add_user_message("it is worse today")
        await _pipeline(scripted_llm).run(conversation, "it is worse today")

        # The second turn's prompt is the last one issued.
        answer_prompt = [p for p in scripted_llm.prompts if "ALREADY ASKED" in p][-1]
        assert "headache" in answer_prompt
        assert "How long have you had this?" in answer_prompt

    async def test_transcript_excludes_assistant_turns(self, scripted_llm):
        conversation, answer = await _ask(scripted_llm, "I have a headache")
        conversation.add_ai_message(answer)
        transcript = conversation.transcript_for_grounding()
        assert "headache" in transcript
        assert answer.reply_text not in transcript


# ----------------------------------------------------------- Emergency


class TestEmergency:
    @pytest.mark.parametrize(
        "text",
        [
            "I have crushing chest pain and cannot breathe",
            "mujhe seene me dard hai aur saans nahi aa raha",
            "मुझे सीने में दर्द है और सांस नहीं आ रही",
        ],
    )
    async def test_escalates_across_languages(self, scripted_llm, text):
        _, answer = await _ask(scripted_llm, text)
        assert answer.urgency.level is UrgencyLevel.EMERGENCY
        assert answer.emergency is not None
        assert answer.emergency_risk is EmergencyRisk.CRITICAL

    async def test_escalates_even_when_model_is_dead(self, dead_llm):
        """Red-flag detection is deterministic and must not need the LLM."""
        _, answer = await _ask(dead_llm, "I have crushing chest pain")
        assert answer.emergency is not None
        assert answer.intent is IntentType.EMERGENCY

    async def test_emergency_skips_retrieval_and_questions(self, scripted_llm):
        retriever = FakeRetriever([RetrievedSnippet(text="x", source="s")])
        _, answer = await _ask(
            scripted_llm, "I cannot breathe at all", retriever=retriever
        )
        assert retriever.queries == []
        assert answer.follow_up_questions == []

    async def test_model_cannot_downgrade_red_flag_urgency(self, scripted_llm):
        scripted_llm.responses["answer"] = answer_payload(
            urgency={"level": "Low", "explanation": "seems fine"}
        )
        _, answer = await _ask(scripted_llm, "crushing chest pain for an hour")
        assert answer.urgency.level is UrgencyLevel.EMERGENCY

    @pytest.mark.parametrize(
        "text,marker",
        [
            ("I have crushing chest pain", "Do not drive yourself"),
            ("mujhe seene me dard hai", "Khud gaadi na chalayein"),
            ("मुझे सीने में दर्द है", "स्वयं गाड़ी न चलाएँ"),
        ],
    )
    async def test_red_flag_reply_carries_first_aid_in_the_patients_language(
        self, dead_llm, text, marker
    ):
        """First aid on this path is deterministic: no model, still guidance."""
        _, answer = await _ask(dead_llm, text)
        assert any(marker in action for action in answer.actions)

    async def test_emergency_card_names_no_hospital_it_cannot_know(
        self, scripted_llm
    ):
        """
        A guessed facility is a fabrication.

        The card's location strip falls back to a hard-coded sample hospital
        when the backend sends nothing, so this field must always be populated
        with something true.
        """
        _, answer = await _ask(scripted_llm, "I have crushing chest pain")
        hint = answer.emergency.nearest_hospital
        assert hint and "Emergency" in hint


class TestModelGradedEmergency:
    """
    The safety net for emergencies the deterministic scan does not match.

    Sepsis signs, a thunderclap headache and similar presentations have no
    red-flag pattern; before this the model could grade a turn `Emergency` and
    the patient would still see no emergency guidance.
    """

    THUNDERCLAP = "the worst headache of my life started an hour ago"

    def _emergency_payload(self, **overrides):
        return answer_payload(
            urgency={"level": "Emergency", "explanation": "Sudden severe onset."},
            **overrides,
        )

    async def test_no_red_flag_fires_on_this_text(self, scripted_llm):
        """Guards the premise: this text reaches the model path, not the node."""
        _, answer = await _ask(scripted_llm, self.THUNDERCLAP)
        assert answer.red_flags == []

    async def test_model_graded_emergency_renders_the_card(self, scripted_llm):
        scripted_llm.responses["answer"] = self._emergency_payload(
            emergency={
                "heading": "Get emergency care now",
                "description": "Call your local emergency number and do not drive.",
            }
        )
        _, answer = await _ask(scripted_llm, self.THUNDERCLAP)

        assert answer.emergency is not None
        assert answer.emergency.heading == "Get emergency care now"
        assert answer.emergency_risk is EmergencyRisk.CRITICAL
        assert "emergency" in answer.to_ui_payload()

    async def test_missing_notice_falls_back_to_deterministic_guidance(
        self, scripted_llm
    ):
        """An Emergency grading without a notice must still guide the patient."""
        scripted_llm.responses["answer"] = self._emergency_payload()
        _, answer = await _ask(scripted_llm, self.THUNDERCLAP)

        assert answer.emergency is not None
        assert "emergency" in answer.emergency.description.casefold()

    async def test_malformed_notice_falls_back(self, scripted_llm):
        scripted_llm.responses["answer"] = self._emergency_payload(
            emergency={"heading": "   ", "description": None}
        )
        _, answer = await _ask(scripted_llm, self.THUNDERCLAP)
        assert answer.emergency.heading
        assert answer.emergency.description

    @pytest.mark.parametrize("level", ["Low", "Medium", "High"])
    async def test_routine_urgency_never_renders_the_card(self, scripted_llm, level):
        """
        The card is a critical full-width alert and reports risk CRITICAL.

        A model that volunteers the key on a sore throat must not be able to
        alarm the patient.
        """
        scripted_llm.responses["answer"] = answer_payload(
            urgency={"level": level, "explanation": "routine"},
            emergency={"heading": "Panic", "description": "Call an ambulance."},
        )
        _, answer = await _ask(scripted_llm, "I have a mild sore throat")

        assert answer.emergency is None
        assert "emergency" not in answer.to_ui_payload()
        assert answer.emergency_risk is not EmergencyRisk.CRITICAL


class TestSelfCareGuidance:
    """Section 1: immediate self-care rides the existing `actions` card."""

    async def test_prompt_asks_for_immediate_self_care(self, scripted_llm):
        await _ask(scripted_llm, "I have a headache")
        prompt = [p for p in scripted_llm.prompts if "ALREADY ASKED" in p][-1]

        assert "IMMEDIATE SELF-CARE" in prompt
        assert "non-prescription" in prompt
        assert "never suggest starting or\nstopping a prescription medicine" in prompt

    async def test_prompt_scopes_the_emergency_key(self, scripted_llm):
        prompt_source = prompts_module.ANSWER_SYSTEM_PROMPT

        assert "EMERGENCY ACTIONS" in prompt_source
        assert "Do not describe invasive or advanced medical interventions" in (
            prompt_source
        )
        # Certainty and follow-up discipline are part of the same contract.
        assert "Never claim certainty" in prompt_source
        assert "pregnancy status" in prompt_source

    async def test_self_care_actions_reach_the_ui_payload(self, scripted_llm):
        scripted_llm.responses["answer"] = answer_payload(
            actions=["Sit down and rest", "Sip water slowly", "Avoid driving today"]
        )
        _, answer = await _ask(scripted_llm, "I feel dizzy when I stand up")
        assert answer.to_ui_payload()["actions"][0] == "Sit down and rest"


# ----------------------------------------------------------- Safety


class TestAntiFabrication:
    async def test_ungrounded_entity_is_rejected(self, scripted_llm):
        scripted_llm.responses["entities"] = entity_payload(
            ("symptom", "headache", 0.9, "I have a headache"),
            ("allergy", "penicillin", 0.99, "I am allergic to penicillin"),
        )
        scripted_llm.responses["answer"] = answer_payload(symptoms=[])

        _, answer = await _ask(scripted_llm, "I have a headache")

        values = {e.value for e in answer.entities}
        assert "headache" in values
        assert "penicillin" not in values

    async def test_grounded_entity_survives(self, scripted_llm):
        scripted_llm.responses["entities"] = entity_payload(
            ("allergy", "ibuprofen", 0.95, "allergic to ibuprofen")
        )
        _, answer = await _ask(
            scripted_llm, "I have a headache and I am allergic to ibuprofen"
        )
        assert "ibuprofen" in {e.value for e in answer.entities}


class TestGuardrails:
    async def test_blocks_unsafe_input(self, scripted_llm):
        scripted_llm.responses["guard_in"] = "UNSAFE: weapon instructions"
        guardrails = LLMGuardrails(llm=scripted_llm)
        allowed, message = await guardrails.check_input("how do I build a weapon")
        assert allowed is False
        assert "can't help" in message.casefold()

    async def test_allows_ordinary_health_question(self, scripted_llm):
        guardrails = LLMGuardrails(llm=scripted_llm)
        allowed, _ = await guardrails.check_input("what causes migraines?")
        assert allowed is True

    async def test_never_blocks_a_red_flag_message(self, scripted_llm):
        """A patient describing chest pain must reach the assistant."""
        scripted_llm.responses["guard_in"] = "UNSAFE: self harm"
        guardrails = LLMGuardrails(llm=scripted_llm)
        allowed, _ = await guardrails.check_input("I have crushing chest pain")
        assert allowed is True

    async def test_fails_open_when_model_unavailable(self, dead_llm):
        guardrails = LLMGuardrails(llm=dead_llm)
        allowed, _ = await guardrails.check_input("I have a headache")
        assert allowed is True

    async def test_output_review_keeps_original_on_failure(self, dead_llm):
        guardrails = LLMGuardrails(llm=dead_llm)
        original = "Your headache is likely tension-related."
        assert await guardrails.check_output(original) == original

    async def test_output_review_rejects_suspiciously_short_revision(
        self, scripted_llm
    ):
        scripted_llm.responses["guard_out"] = "No."
        guardrails = LLMGuardrails(llm=scripted_llm)
        original = "A long, detailed and appropriate clinical explanation here." * 2
        assert await guardrails.check_output(original) == original


# ----------------------------------------------------------- Resilience


class TestDegradation:
    async def test_dead_model_returns_honest_fallback(self, dead_llm):
        _, answer = await _ask(dead_llm, "I have a headache")
        assert answer.degraded is True
        assert answer.reply_text
        assert answer.confidence == 0.0
        assert answer.knowledge_source is KnowledgeSource.NONE

    async def test_fallback_speaks_the_patients_language(self, dead_llm):
        _, answer = await _ask(dead_llm, "mujhe sar dard hai aur bukhar hai")
        assert answer.language is Language.HINGLISH
        assert "Maaf kijiye" in answer.reply_text

    async def test_retrieval_failure_is_contained(self, scripted_llm):
        _, answer = await _ask(
            scripted_llm, "I have a headache", retriever=ExplodingRetriever()
        )
        assert answer.degraded is False
        assert answer.summary

    async def test_model_exception_does_not_crash_the_turn(self, scripted_llm):
        scripted_llm.responses["answer"] = RuntimeError("provider exploded")
        _, answer = await _ask(scripted_llm, "I have a headache")
        assert isinstance(answer, AssistantAnswer)
        assert answer.degraded is True

    async def test_retrieved_context_is_used_and_cited(self, scripted_llm):
        retriever = FakeRetriever(
            [RetrievedSnippet(text="Migraine guidance", source="NICE CG150")]
        )
        _, answer = await _ask(
            scripted_llm, "what helps a migraine?", retriever=retriever
        )
        assert answer.knowledge_source is KnowledgeSource.RAG
        assert "NICE CG150" in answer.references
        assert any("Migraine guidance" in p for p in scripted_llm.prompts)


class TestConversationLength:
    async def test_short_conversation_works(self, scripted_llm):
        _, answer = await _ask(scripted_llm, "hi")
        assert answer.reply_text

    async def test_long_conversation_prompt_stays_bounded(self, scripted_llm):
        """History is windowed, so a long thread cannot grow the prompt forever."""
        conversation = Conversation(patient_user_id="u1")
        for i in range(60):
            conversation.add_user_message(f"message number {i}")

        conversation.add_user_message("and now my head hurts")
        await _pipeline(scripted_llm).run(conversation, "and now my head hurts")

        answer_prompt = [p for p in scripted_llm.prompts if "ALREADY ASKED" in p][-1]
        assert "message number 0" not in answer_prompt
        assert "message number 59" in answer_prompt
