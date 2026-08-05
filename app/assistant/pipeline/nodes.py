"""
Assistant pipeline nodes.

One method per stage of the brief's pipeline:

    language -> memory -> intent -> entities -> retrieval -> LLM -> structured JSON

Every node degrades safely. A model outage yields a reduced but honest answer
rather than an exception, and emergency detection never depends on the model.
"""

from __future__ import annotations

import logging
from typing import Any

from app.ai_core.utils.validators import sanitize_prompt_input
from app.assistant.application.ports import RetrievedSnippet
from app.assistant.domain.entities import (
    AssistantAnswer,
    CauseItem,
    EmergencyNotice,
    LifestyleAdviceItem,
    MedicalEntity,
    MedicineGuidance,
    SpecialistSuggestion,
    UrgencyAssessment,
)
from app.assistant.domain.enums import (
    ConfidenceLabel,
    IntentType,
    KnowledgeSource,
    MedicineType,
    SpecialistPriority,
    UrgencyLevel,
)
from app.assistant.domain.language import detect_language, language_instruction
from app.assistant.pipeline import prompts
from app.assistant.pipeline.state import AssistantState
from app.intake.domain.policies import detect_red_flags, is_evidence_grounded
from app.intake.domain.specialties import canonicalise_specialty

logger = logging.getLogger(__name__)

_MAX_ENTITIES = 25
_MAX_CAUSES = 3
_MAX_FOLLOW_UPS = 3

_EMERGENCY_GUIDANCE = {
    "english": (
        "What you are describing may be a medical emergency. Please call your "
        "local emergency number or go to the nearest emergency department now. "
        "Do not wait for an online reply."
    ),
    "hindi": (
        "आप जो बता रहे हैं वह एक मेडिकल इमरजेंसी हो सकती है। कृपया तुरंत अपने "
        "नज़दीकी अस्पताल जाएँ या आपातकालीन नंबर पर कॉल करें। ऑनलाइन उत्तर का "
        "इंतज़ार न करें।"
    ),
    "hinglish": (
        "Aap jo bata rahe hain wo ek medical emergency ho sakti hai. Kripya "
        "turant nazdeeki hospital jaayein ya emergency number par call karein. "
        "Online reply ka intezaar na karein."
    ),
}

_EMERGENCY_HEADING = {
    "english": "Seek emergency care now",
    "hindi": "तुरंत आपातकालीन चिकित्सा सहायता लें",
    "hinglish": "Turant emergency medical help lein",
}

_EMERGENCY_ACTIONS: dict[str, list[str]] = {
    # General, non-invasive steps only. Nothing here asks an untrained patient
    # to perform a procedure, and nothing here can be wrong enough to harm:
    # the worst case is that help was already on its way.
    "english": [
        "Call your local emergency number now, or ask someone with you to call.",
        "Do not drive yourself — let someone else take you, or wait for the "
        "ambulance.",
        "Stay with another adult until help arrives, if anyone is nearby.",
        "Sit or lie down somewhere safe and try to stay still and calm.",
        "If someone becomes unresponsive, turn them on their side so the airway "
        "stays clear, and loosen tight clothing.",
        "If a doctor has already prescribed you emergency medication for a known "
        "condition, use it exactly as they instructed.",
        "Do not delay care while waiting for a reply here.",
    ],
    "hindi": [
        "अभी अपने स्थानीय आपातकालीन नंबर पर कॉल करें, या पास मौजूद किसी व्यक्ति से "
        "कॉल करने को कहें।",
        "स्वयं गाड़ी न चलाएँ — किसी और को ले जाने दें या एम्बुलेंस की प्रतीक्षा करें।",
        "यदि कोई पास हो, तो मदद आने तक किसी वयस्क के साथ रहें।",
        "किसी सुरक्षित जगह पर बैठ या लेट जाएँ और शांत रहने का प्रयास करें।",
        "यदि व्यक्ति बेहोश हो जाए, तो उसे करवट पर लिटाएँ ताकि साँस का रास्ता खुला "
        "रहे, और तंग कपड़े ढीले कर दें।",
        "यदि डॉक्टर ने किसी ज्ञात बीमारी के लिए पहले से आपातकालीन दवा दी है, तो "
        "उसे उनके निर्देशानुसार ही लें।",
        "यहाँ उत्तर की प्रतीक्षा में इलाज में देरी न करें।",
    ],
    "hinglish": [
        "Abhi apne local emergency number par call karein, ya paas mojood kisi "
        "vyakti se call karne ko kahein.",
        "Khud gaadi na chalayein — kisi aur ko le jaane dein ya ambulance ka "
        "intezaar karein.",
        "Agar koi paas ho, to madad aane tak kisi adult ke saath rahein.",
        "Kisi surakshit jagah par baith ya let jaayein aur shaant rehne ki "
        "koshish karein.",
        "Agar vyakti behosh ho jaaye, to unhe karvat par litayein taaki saans ka "
        "raasta khula rahe, aur tight kapde dheele kar dein.",
        "Agar doctor ne kisi known condition ke liye pehle se emergency dawa di "
        "hai, to use unke bataye tareeke se hi lein.",
        "Yahaan reply ka intezaar karke ilaaj mein deri na karein.",
    ],
}

_HOSPITAL_HINT = {
    # Shown in the card's location strip. Deliberately points at the platform's
    # own emergency page instead of naming a facility: this layer knows nothing
    # about where the patient is, and a guessed hospital name is a fabrication.
    "english": "Open Emergency for verified nearby hospitals and one-tap SOS",
    "hindi": "नज़दीकी सत्यापित अस्पतालों और SOS के लिए Emergency पेज खोलें",
    "hinglish": "Nazdeeki verified hospitals aur SOS ke liye Emergency page kholein",
}


def _localised(table: dict[str, Any], language: str):
    """Pick the patient's language, falling back to English."""
    return table.get(language, table["english"])


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any, limit: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def _note_soft_failure(state: AssistantState, stage: str, reason: str) -> None:
    """
    Record that an enrichment stage fell back.

    Kept out of `degraded` on purpose: these stages improve an answer, they do
    not constitute one.
    """
    state.setdefault("soft_failures", []).append(f"{stage}: {reason}")


class AssistantNodes:
    """
    Dependency-injected node collection.

    Holds only the LLM (process-wide). Knowledge retrieval is request-scoped and
    arrives through `AssistantState`, keeping the compiled graph reusable.
    """

    def __init__(self, *, llm) -> None:
        self._llm = llm

    def _llm_error(self) -> str:
        """
        The provider's reason for returning nothing.

        Node-level logs previously recorded only *that* a stage got an empty
        payload, so diagnosing a degraded reply meant guessing between a missing
        key, a revoked key and a decommissioned model. Test doubles need not
        implement `last_error`.
        """
        return getattr(self._llm, "last_error", None) or "no detail from provider"

    # -- stage 1: language + deterministic safety scan ---------------------

    async def detect_language_node(self, state: AssistantState) -> AssistantState:
        """
        Identify language and run red-flag detection.

        Red flags are computed here, before any model call, so an unavailable
        LLM cannot suppress an emergency escalation.
        """
        conversation = state["conversation"]
        raw = state.get("user_text", "")
        state["user_text"] = sanitize_prompt_input(raw, max_length=4000)
        state.setdefault("degraded", False)
        state.setdefault("soft_failures", [])
        state.setdefault("rejected_entities", 0)

        conversation.language = detect_language(state["user_text"])

        flags = detect_red_flags(conversation.transcript_for_grounding())
        state["red_flags"] = flags
        if flags:
            logger.warning(
                "[ASSISTANT_RED_FLAG] conversation=%s flags=%s",
                conversation.conversation_id,
                flags,
            )
        return state

    # -- stage 2: intent ---------------------------------------------------

    async def detect_intent_node(self, state: AssistantState) -> AssistantState:
        """Classify the message. Red flags settle it without a model call."""
        conversation = state["conversation"]
        if state.get("red_flags"):
            state["intent"] = IntentType.EMERGENCY
            return state

        payload = await self._llm.complete_json(
            system_prompt=prompts.INTENT_SYSTEM_PROMPT,
            user_content=f"PATIENT MESSAGE:\n{state['user_text']}",
            max_tokens=120,
            temperature=0.0,
        )
        raw = str(payload.get("intent", "")).strip().casefold()
        try:
            state["intent"] = IntentType(raw)
        except ValueError:
            # A missed classification is not a degraded answer: the answer stage
            # runs next and works from the transcript, not from this label.
            reason = (
                self._llm_error() if not payload else f"unrecognised intent {raw!r}"
            )
            logger.warning(
                "[ASSISTANT_INTENT_FALLBACK] conversation=%s reason=%s",
                conversation.conversation_id,
                reason,
            )
            _note_soft_failure(state, "intent", reason)
            state["intent"] = (
                IntentType.FOLLOW_UP
                if conversation.asked_questions
                else IntentType.SYMPTOM_REPORT
            )
        return state

    # -- stage 3: medical entity extraction --------------------------------

    async def extract_entities_node(self, state: AssistantState) -> AssistantState:
        """
        Extract clinical entities, then verify each against the patient's words.

        Reuses the intake agent's grounding check so a fabricated allergy is
        rejected here exactly as it is during structured intake.
        """
        conversation = state["conversation"]
        transcript = conversation.transcript_for_grounding()
        state.setdefault("entities", [])

        payload = await self._llm.complete_json(
            system_prompt=prompts.ENTITY_SYSTEM_PROMPT,
            user_content=f"PATIENT'S OWN WORDS (verbatim):\n{transcript}",
            max_tokens=1200,
            temperature=0.0,
        )
        raw_entities = payload.get("entities")
        if not isinstance(raw_entities, list):
            # Entities enrich the answer; the answer stage reads the transcript
            # directly, so losing them must not degrade a real model reply.
            reason = (
                self._llm_error() if not payload else "response had no entities list"
            )
            logger.warning(
                "[ASSISTANT_ENTITY_FALLBACK] conversation=%s reason=%s",
                conversation.conversation_id,
                reason,
            )
            _note_soft_failure(state, "entities", reason)
            return state

        kept: list[MedicalEntity] = []
        rejected = 0
        for item in raw_entities[:_MAX_ENTITIES]:
            if not isinstance(item, dict):
                continue
            entity = MedicalEntity.from_dict(item)
            if not entity.value:
                continue
            grounded, _ = is_evidence_grounded(entity.evidence, transcript)
            if grounded:
                kept.append(entity)
            else:
                rejected += 1

        state["entities"] = kept
        state["rejected_entities"] = rejected
        if rejected:
            logger.warning(
                "[ASSISTANT_UNGROUNDED_DROPPED] conversation=%s count=%d",
                conversation.conversation_id,
                rejected,
            )
        return state

    # -- stage 4: knowledge retrieval --------------------------------------

    async def retrieve_knowledge_node(self, state: AssistantState) -> AssistantState:
        """Fetch grounding context. Absence is normal, not an error."""
        state.setdefault("snippets", [])
        retriever = state.get("retriever")
        if retriever is None or not retriever.is_available:
            return state

        try:
            snippets = await retriever.retrieve(state["user_text"], limit=4)
        except Exception:
            logger.exception("[ASSISTANT_RETRIEVAL_FAILED] continuing without context")
            return state

        state["snippets"] = snippets
        logger.info("[ASSISTANT_RETRIEVED] snippets=%d", len(snippets))
        return state

    # -- stage 5: answer generation ----------------------------------------

    async def generate_answer_node(self, state: AssistantState) -> AssistantState:
        """Produce the structured answer the existing cards render."""
        conversation = state["conversation"]
        snippets: list[RetrievedSnippet] = state.get("snippets") or []

        context = "\n\n".join(
            f"[{i + 1}] {s.text[:1200]}" for i, s in enumerate(snippets)
        )

        system_prompt = (
            f"{prompts.ANSWER_SYSTEM_PROMPT}\n\n"
            f"{language_instruction(conversation.language)}"
        )
        user_content = prompts.build_answer_user_content(
            transcript=self._recent_transcript(conversation),
            known_symptoms=conversation.known_symptoms,
            asked_questions=conversation.asked_questions,
            retrieved_context=context,
            latest_message=state["user_text"],
        )

        payload = await self._llm.complete_json(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=2200,
            temperature=0.3,
        )

        if not payload:
            # Recovery attempt before degrading. The full prompt is large (a
            # transcript, known symptoms, asked questions and up to four
            # retrieved passages) and a 2200-token budget; the most common
            # non-credential failure is the model overrunning that budget and
            # returning truncated, unparseable JSON. A smaller, context-free
            # retry usually succeeds, and a real answer beats an apology.
            logger.warning(
                "[ASSISTANT_ANSWER_EMPTY] conversation=%s reason=%s — retrying "
                "with a reduced prompt",
                conversation.conversation_id,
                self._llm_error(),
            )
            payload = await self._llm.complete_json(
                system_prompt=system_prompt,
                user_content=prompts.build_answer_user_content(
                    transcript="",
                    known_symptoms=conversation.known_symptoms,
                    asked_questions=[],
                    retrieved_context="",
                    latest_message=state["user_text"],
                ),
                max_tokens=1200,
                temperature=0.3,
            )
            if payload:
                logger.info(
                    "[ASSISTANT_ANSWER_RECOVERED] conversation=%s reduced prompt "
                    "succeeded",
                    conversation.conversation_id,
                )
                _note_soft_failure(state, "answer", "recovered on reduced prompt")
                state["answer"] = self._build_answer(state, payload, [])
                return state

            # Every recovery path is exhausted: no credential worked, no model
            # worked, and the reduced prompt also failed.
            state["degraded"] = True
            logger.error(
                "[ASSISTANT_DEGRADED] conversation=%s the model produced no answer "
                "after a full and a reduced attempt. Provider reason: %s. "
                "Soft failures: %s",
                conversation.conversation_id,
                self._llm_error(),
                state.get("soft_failures") or "none",
            )
            state["answer"] = self._fallback_answer(state)
            return state

        state["answer"] = self._build_answer(state, payload, snippets)
        return state

    # -- emergency short-circuit -------------------------------------------

    async def emergency_node(self, state: AssistantState) -> AssistantState:
        """
        Return emergency guidance immediately.

        No retrieval, no follow-up questions: a patient describing crushing
        chest pain is told to seek care, not asked to rate their symptoms.

        The steps in `actions` are fixed rather than generated: this path exists
        precisely for when the model cannot be trusted or reached, so its
        first-aid guidance must not depend on one.
        """
        conversation = state["conversation"]
        flags = state.get("red_flags") or []
        language = conversation.language.value
        guidance = _localised(_EMERGENCY_GUIDANCE, language)

        answer = AssistantAnswer(
            reply_text=guidance,
            summary=guidance,
            symptoms=[e.value for e in state.get("entities", [])] or list(
                conversation.known_symptoms
            ),
            actions=list(_localised(_EMERGENCY_ACTIONS, language)),
            urgency=UrgencyAssessment(
                level=UrgencyLevel.EMERGENCY,
                explanation="Automated screening matched: " + "; ".join(flags),
            ),
            emergency=EmergencyNotice(
                heading=_localised(_EMERGENCY_HEADING, language),
                description=guidance,
                nearest_hospital=_localised(_HOSPITAL_HINT, language),
            ),
            specialist=SpecialistSuggestion(
                name="Emergency Medicine",
                reason="Red-flag symptoms require immediate in-person assessment.",
                priority=SpecialistPriority.URGENT,
            ),
            conversation_title="Emergency guidance",
            language=conversation.language,
            intent=IntentType.EMERGENCY,
            confidence=1.0,
            knowledge_source=KnowledgeSource.NONE,
            red_flags=list(flags),
            entities=list(state.get("entities", [])),
        )
        state["answer"] = answer

        logger.warning(
            "[ASSISTANT_EMERGENCY] conversation=%s flags=%s",
            conversation.conversation_id,
            flags,
        )
        return state

    # -- helpers -----------------------------------------------------------

    def _recent_transcript(self, conversation) -> str:
        """Recent turns only, to bound prompt growth on long conversations."""
        from app.assistant.config import get_assistant_settings

        limit = get_assistant_settings().max_history_messages
        lines = []
        for message in conversation.recent_messages(limit):
            speaker = "Patient" if message.role.value == "user" else "Assistant"
            lines.append(f"{speaker}: {message.text}")
        return "\n".join(lines)

    def _build_answer(
        self,
        state: AssistantState,
        payload: dict[str, Any],
        snippets: list[RetrievedSnippet],
    ) -> AssistantAnswer:
        conversation = state["conversation"]
        flags = state.get("red_flags") or []

        causes = [
            CauseItem(
                cause=str(c.get("cause", "")).strip(),
                confidence=self._as_enum(
                    ConfidenceLabel, c.get("confidence"), ConfidenceLabel.MEDIUM
                ),
            )
            for c in (payload.get("causes") or [])
            if isinstance(c, dict) and str(c.get("cause", "")).strip()
        ][:_MAX_CAUSES]

        lifestyle = [
            LifestyleAdviceItem(
                category=str(a.get("category", "General")).strip() or "General",
                description=str(a.get("description", "")).strip(),
            )
            for a in (payload.get("lifestyleAdvice") or [])
            if isinstance(a, dict) and str(a.get("description", "")).strip()
        ]

        medicines = [
            MedicineGuidance(
                name=str(m.get("name", "")).strip(),
                purpose=str(m.get("purpose", "")).strip(),
                side_effects=_as_str_list(m.get("sideEffects"), 6),
                precautions=str(m.get("precautions", "")).strip(),
                type=self._as_enum(MedicineType, m.get("type"), MedicineType.OTC),
            )
            for m in (payload.get("medicines") or [])
            if isinstance(m, dict) and str(m.get("name", "")).strip()
        ]

        specialist = None
        raw_specialist = payload.get("specialist")
        if isinstance(raw_specialist, dict) and raw_specialist.get("name"):
            specialist = SpecialistSuggestion(
                name=canonicalise_specialty(str(raw_specialist.get("name"))),
                reason=str(raw_specialist.get("reason", "")).strip(),
                priority=self._as_enum(
                    SpecialistPriority,
                    raw_specialist.get("priority"),
                    SpecialistPriority.ROUTINE,
                ),
            )

        # Urgency: never below the deterministic red-flag assessment.
        urgency_level = UrgencyLevel.LOW
        raw_urgency = payload.get("urgency")
        if isinstance(raw_urgency, dict):
            urgency_level = self._as_enum(
                UrgencyLevel, raw_urgency.get("level"), UrgencyLevel.LOW
            )
        explanation = (
            str(raw_urgency.get("explanation", "")).strip()
            if isinstance(raw_urgency, dict)
            else ""
        )
        if flags and urgency_level.rank < UrgencyLevel.EMERGENCY.rank:
            urgency_level = UrgencyLevel.EMERGENCY

        # Emergency card on the model path — the safety net for presentations
        # the deterministic red-flag scan does not match (anaphylaxis after a
        # sting, sepsis signs, "worst headache of my life"). Gated on the final
        # urgency, never on the model merely volunteering the key: this card is
        # a critical full-width alert and reports risk CRITICAL upstream, so it
        # must not fire on routine symptoms. Emergency actions themselves are
        # never left to the model alone — a missing or empty description falls
        # back to the same deterministic guidance the red-flag path uses.
        emergency = None
        if urgency_level is UrgencyLevel.EMERGENCY:
            language = conversation.language.value
            raw_emergency = payload.get("emergency")
            heading = description = ""
            if isinstance(raw_emergency, dict):
                heading = str(raw_emergency.get("heading", "")).strip()
                description = str(raw_emergency.get("description", "")).strip()
            emergency = EmergencyNotice(
                heading=heading or _localised(_EMERGENCY_HEADING, language),
                description=description or _localised(_EMERGENCY_GUIDANCE, language),
                nearest_hospital=_localised(_HOSPITAL_HINT, language),
            )
            logger.warning(
                "[ASSISTANT_MODEL_EMERGENCY] conversation=%s red_flags=%s "
                "model_supplied_notice=%s",
                conversation.conversation_id,
                flags or "none",
                isinstance(raw_emergency, dict),
            )

        # Suppress questions already asked in earlier turns.
        follow_ups = [
            q
            for q in _as_str_list(payload.get("followUpQuestions"), 6)
            if not conversation.was_already_asked(q)
        ][:_MAX_FOLLOW_UPS]

        # Prefer the model's symptom list: it is already phrased in the
        # patient's language. Grounded entities are a fallback only, so the UI
        # never shows the same symptom twice in two languages.
        symptoms = _as_str_list(payload.get("symptoms"), 12)
        if not symptoms:
            symptoms = [
                e.value for e in state.get("entities", []) if e.kind == "symptom"
            ][:12]

        source = (
            KnowledgeSource.RAG if snippets else KnowledgeSource.MODEL_ONLY
        )
        references = _as_str_list(payload.get("references"), 6)
        for snippet in snippets:
            if snippet.source and snippet.source not in references:
                references.append(snippet.source)

        return AssistantAnswer(
            reply_text=str(payload.get("replyText", "")).strip()
            or str(payload.get("summary", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            symptoms=symptoms,
            causes=causes,
            actions=_as_str_list(payload.get("actions"), 8),
            lifestyle_advice=lifestyle,
            medicines=medicines,
            specialist=specialist,
            urgency=UrgencyAssessment(level=urgency_level, explanation=explanation),
            emergency=emergency,
            references=references[:6],
            follow_up_questions=follow_ups,
            conversation_title=str(payload.get("conversationTitle", "")).strip()
            or conversation.title,
            language=conversation.language,
            intent=state.get("intent", IntentType.UNCLEAR),
            confidence=min(1.0, max(0.0, _as_float(payload.get("confidence"), 0.7))),
            knowledge_source=source,
            red_flags=list(flags),
            entities=list(state.get("entities", [])),
            degraded=bool(state.get("degraded")),
        )

    @staticmethod
    def _as_enum(enum_cls, raw: Any, default):
        """Case-insensitive enum coercion with a safe default."""
        if raw is None:
            return default
        text = str(raw).strip()
        for member in enum_cls:
            if member.value.casefold() == text.casefold():
                return member
        return default

    def _fallback_answer(self, state: AssistantState) -> AssistantAnswer:
        """
        Deterministic reply used when the model is unavailable.

        Says plainly that it could not analyse the message rather than emitting
        empty cards that would look like a confident non-answer.
        """
        conversation = state["conversation"]
        message = {
            "hindi": (
                "क्षमा करें, मैं अभी आपके संदेश का विश्लेषण नहीं कर पा रहा हूँ। "
                "कृपया थोड़ी देर बाद पुनः प्रयास करें। यदि यह आपात स्थिति है तो "
                "तुरंत डॉक्टर से संपर्क करें।"
            ),
            "hinglish": (
                "Maaf kijiye, main abhi aapke message ka analysis nahi kar pa raha "
                "hoon. Kripya thodi der baad dobara koshish karein. Agar yeh "
                "emergency hai to turant doctor se sampark karein."
            ),
        }.get(
            conversation.language.value,
            "Sorry — I could not analyse your message just now. Please try again "
            "shortly. If this is urgent, contact a doctor directly.",
        )

        return AssistantAnswer(
            reply_text=message,
            summary=message,
            symptoms=list(conversation.known_symptoms),
            language=conversation.language,
            intent=state.get("intent", IntentType.UNCLEAR),
            confidence=0.0,
            knowledge_source=KnowledgeSource.NONE,
            conversation_title=conversation.title,
            red_flags=list(state.get("red_flags") or []),
            entities=list(state.get("entities", [])),
            degraded=True,
        )
