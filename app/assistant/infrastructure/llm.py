"""
`AssistantLLMPort` adapter for the AI Medical Assistant.

Delegates to the shared `app.core.ai_provider.GroqClient`. Previously this was a
separate Groq client with its own credential resolution; consolidating removes
the duplicate implementation and guarantees the assistant, the intake agent and
the legacy `ai_core` services all authenticate with the same Groq account.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.ai_provider import AIProviderConfig, GroqClient

logger = logging.getLogger(__name__)


class AssistantGroqAdapter(GroqClient):
    """Async Groq access (JSON + text) for the assistant pipeline and guardrails."""

    def __init__(self, config: AIProviderConfig | None = None) -> None:
        super().__init__(config=config)

    async def health(self, *, probe: bool = False) -> dict[str, Any]:
        payload = await super().health(probe=probe)
        payload.setdefault("component", "assistant")
        return payload
