"""
Safety guardrails, ported from the source project's `LocalGuardrails`.

The original builds LCEL chains (`PromptTemplate | llm | StrOutputParser`) against
LangChain 0.3 and calls them synchronously. This port keeps the two-stage
input/output design and the prompt intent, but runs on the assistant's async LLM
port — so it adds no `langchain-groq` dependency and never blocks the event loop.

The input prompt was also tightened. The original blocked ~47 categories
including "asks for the source of the information" and any non-medical content,
which would refuse legitimate patient questions; this version blocks genuine
safety violations and explicitly permits mental-health distress, which needs a
careful answer rather than a refusal.
"""

from __future__ import annotations

import logging

from app.assistant.application.ports import AssistantLLMPort
from app.assistant.config import AssistantSettings, get_assistant_settings
from app.assistant.pipeline import prompts
from app.intake.domain.policies import detect_red_flags

logger = logging.getLogger(__name__)

_SAFE_TOKEN = "SAFE"
_UNSAFE_TOKEN = "UNSAFE"

_BLOCKED_MESSAGE = (
    "I can't help with that request. If you have a health concern, "
    "describe your symptoms and I'll do my best to help."
)


class LLMGuardrails:
    """Two-stage input/output safety filter."""

    def __init__(
        self,
        *,
        llm: AssistantLLMPort,
        settings: AssistantSettings | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings or get_assistant_settings()

    async def check_input(self, text: str) -> tuple[bool, str]:
        """
        Screen a patient message.

        Fails **open** on model unavailability: refusing every message during a
        provider outage would be a worse failure for a patient than letting an
        edge case through, and the output stage still runs. A message containing
        clinical red flags is always allowed straight through — someone
        describing chest pain must never be blocked by a safety filter.
        """
        if not self._settings.enable_guardrails:
            return True, text

        if detect_red_flags(text):
            return True, text

        verdict = await self._llm.complete_text(
            system_prompt="You are a strict content safety classifier.",
            user_content=prompts.GUARDRAIL_INPUT_PROMPT.format(input=text),
            max_tokens=60,
            temperature=0.0,
        )

        if not verdict:
            logger.warning("[ASSISTANT_GUARDRAIL_INPUT_UNAVAILABLE] failing open")
            return True, text

        normalised = verdict.strip().upper()
        if normalised.startswith(_UNSAFE_TOKEN):
            reason = (
                verdict.split(":", 1)[1].strip()
                if ":" in verdict
                else "content policy violation"
            )
            logger.warning("[ASSISTANT_GUARDRAIL_BLOCKED] reason=%s", reason[:120])
            return False, _BLOCKED_MESSAGE

        if not normalised.startswith(_SAFE_TOKEN):
            # Unrecognised verdict: treat as safe rather than silently refusing.
            logger.warning(
                "[ASSISTANT_GUARDRAIL_UNCLEAR] verdict=%r", verdict[:80]
            )
        return True, text

    async def check_output(self, output: str, *, user_input: str = "") -> str:
        """
        Review the assistant's reply.

        Returns the original text unchanged on any failure — a guardrail outage
        must not blank out a valid clinical answer.
        """
        if not output or not output.strip():
            return output
        if not self._settings.enable_guardrails:
            return output

        revised = await self._llm.complete_text(
            system_prompt="You are a medical response safety reviewer.",
            user_content=prompts.GUARDRAIL_OUTPUT_PROMPT.format(
                user_input=user_input, output=output
            ),
            max_tokens=900,
            temperature=0.0,
        )

        if not revised:
            return output

        # Guard against a reviewer that returns a refusal or an empty shell
        # instead of a revision.
        if len(revised) < max(24, len(output) // 4):
            logger.warning(
                "[ASSISTANT_GUARDRAIL_OUTPUT_SUSPECT] keeping original "
                "(orig=%d revised=%d)",
                len(output),
                len(revised),
            )
            return output

        return revised


class NullGuardrails:
    """No-op guardrails, for tests and for explicitly disabled deployments."""

    async def check_input(self, text: str) -> tuple[bool, str]:
        return True, text

    async def check_output(self, output: str, *, user_input: str = "") -> str:
        return output
