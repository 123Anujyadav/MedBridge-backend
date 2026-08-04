"""
Plain-language summariser for safety findings.

Reuses the platform's single Groq client rather than opening its own, so model
selection, fallback and credential handling stay in one place.

The prompt is written to keep the model inside its lane. It receives findings
that have already been gathered from labels and rules, and its only job is to
phrase them for a reader. It is told explicitly not to add findings, not to
change the prescription, and to say so plainly when nothing was found. Anything
it produces beyond the supplied evidence is unciteable, which is why the summary
is stored separately from the findings and never becomes one.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

from app.core.ai_provider import get_ai_provider_config, get_groq_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical pharmacist assistant for MedBridge.

You will be given a list of safety findings that have ALREADY been established \
from FDA drug labels, RxNorm ingredient data and deterministic rules, plus the \
patient's context.

Your ONLY job is to write a short, calm, plain-language summary of those \
findings for a patient and their doctor to read together.

Hard rules:
- Do NOT invent findings. Describe only what you are given.
- Do NOT suggest changing, adding, removing or re-dosing any medicine. The \
doctor's prescription stands. You may say a finding is "worth discussing with \
your doctor".
- Do NOT contradict a finding's severity.
- If the findings list is empty, say that no known issues were detected from \
the sources checked, and mention any medicines listed as unchecked.
- Never state or imply that an unchecked medicine is safe.
- 2 to 4 sentences. No lists, no headings, no markdown.
- Write for a worried adult with no medical training."""


class GroqSafetyExplainer:
    """Implements `SafetyExplainer`."""

    name = "groq"

    async def summarise(
        self,
        *,
        medications: Sequence[str],
        findings_payload: Sequence[dict],
        patient_context: dict,
    ) -> tuple[str, str | None]:
        config = get_ai_provider_config()
        if not config.is_configured():
            logger.info("[RXSAFETY_EXPLAIN_SKIPPED] Groq is not configured")
            return "", None

        payload = {
            "medicines": list(medications),
            "findings": [
                {
                    "category": f.get("category"),
                    "severity": f.get("severity"),
                    "title": f.get("title"),
                    "detail": (f.get("detail") or "")[:400],
                    "medicines": f.get("medications_involved") or [],
                }
                for f in findings_payload
            ],
            "unchecked_medicines": patient_context.get("unchecked", []),
            "patient": {
                "age": patient_context.get("age"),
                "is_pregnant": patient_context.get("is_pregnant"),
                "allergies": patient_context.get("allergies", []),
                "conditions": patient_context.get("conditions", []),
            },
        }

        try:
            text = await get_groq_client().complete_text(
                system_prompt=SYSTEM_PROMPT,
                user_content=json.dumps(payload, ensure_ascii=False),
                max_tokens=400,
                temperature=0.2,
            )
        except Exception as exc:
            # complete_text is documented never to raise, but a summary is the
            # most disposable part of a safety review: if it fails the findings
            # still stand on their own and are shown without prose.
            logger.warning("[RXSAFETY_EXPLAIN_FAILED] %s", exc)
            return "", None

        return (text or "").strip(), (config.model if text else None)
