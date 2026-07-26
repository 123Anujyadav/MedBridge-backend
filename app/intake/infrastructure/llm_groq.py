"""
`LLMPort` adapter for the Medical Case Intake Agent.

This used to be a standalone Groq client with its own credential lookup and its
own tier-fallback logic — a near-duplicate of the assistant's adapter, reading a
different key. Both now share `app.core.ai_provider.GroqClient`, so there is one
client implementation and one credential for the whole platform.

The class is kept as a thin, named subclass rather than replaced by a bare
alias: it documents intake's dependency at the DI boundary and leaves room for
intake-specific tuning without reintroducing a second client.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.ai_provider import AIProviderConfig, GroqClient

logger = logging.getLogger(__name__)


class GroqJSONAdapter(GroqClient):
    """Async, JSON-mode Groq access for the intake workflow."""

    def __init__(
        self,
        api_key: str | None = None,
        config: AIProviderConfig | None = None,
    ) -> None:
        # `api_key` is retained for tests and explicit overrides. When omitted —
        # the production path — the centralised provider supplies the credential.
        if api_key:
            override = AIProviderConfig()
            override._file_values = {  # noqa: SLF001 - deliberate injection point
                **override._file_values,
                "GROQ_API_KEY": api_key,
            }
            config = override
        super().__init__(config=config)

    async def health(self) -> dict[str, Any]:
        payload = await super().health()
        payload.setdefault("component", "intake")
        return payload

    @staticmethod
    def _parse(content: str) -> dict[str, Any] | None:
        """Backwards-compatible alias for the shared JSON parser."""
        return GroqClient.parse_json(content)
