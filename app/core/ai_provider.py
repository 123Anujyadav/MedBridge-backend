"""
Centralised Groq configuration and client for every AI service.

Groq is the platform's single AI provider. Before this module the credential was
resolved independently in five places — `ai_core.model_manager`,
`ai_core.chat_agent`, `services.ai_service`, the intake adapter and the
assistant adapter — and they did not agree: three read the platform `.env` key
while only the assistant read the working one, which is why `/ai/chat` and
`/ai/analyze-report` returned 401 while the assistant worked.

Everything now resolves through `get_ai_provider_config()`, and every async
caller shares `get_groq_client()`.

Credential resolution order (first non-empty wins):

  1. `GROQ_API_KEY` in `.env.ai-assistant`  — the AI provider environment
  2. `GROQ_API_KEY` in the process environment
  3. `settings.GROQ_API_KEY` from the platform `.env`

Nothing is hardcoded and the key is never logged: `describe()` reports only
whether a key is present and a short fingerprint, never the value.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from app.core.config import settings

logger = logging.getLogger(__name__)

AI_ENV_FILENAME = ".env.ai-assistant"

# Backend/ — three levels up from Backend/app/core/ai_provider.py
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
AI_ENV_PATH = _BACKEND_ROOT / AI_ENV_FILENAME

DEFAULT_MODEL = "llama-3.3-70b-versatile"

FALLBACK_MODELS: tuple[str, ...] = ("llama-3.1-8b-instant", "llama3-70b-8192")
"""Tried in order after the configured model when a request fails."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class AIProviderConfig:
    """Single source of truth for Groq credentials and model selection."""

    def __init__(self, env_path: Path | None = None) -> None:
        self._path = env_path or AI_ENV_PATH
        self._file_values: dict[str, str] = {}

        if self._path.exists():
            # `dotenv_values` rather than `load_dotenv`: this must not mutate
            # os.environ and leak the AI credential into unrelated modules.
            self._file_values = {
                k: v for k, v in dotenv_values(self._path).items() if v
            }

    def _resolve(self, key: str, default: str = "") -> str:
        for candidate in (
            self._file_values.get(key),
            os.getenv(key),
            getattr(settings, key, None),
        ):
            if candidate:
                return str(candidate).strip()
        return default

    @property
    def api_key(self) -> str:
        return self._resolve("GROQ_API_KEY")

    @property
    def model(self) -> str:
        return self._resolve("GROQ_MODEL", DEFAULT_MODEL)

    @property
    def models(self) -> tuple[str, ...]:
        """Configured model first, then fallbacks, without duplicates."""
        primary = self.model
        return (primary, *(m for m in FALLBACK_MODELS if m != primary))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def source(self) -> str:
        """Which layer supplied the key. Diagnostic only, never the value."""
        if self._file_values.get("GROQ_API_KEY"):
            return AI_ENV_FILENAME
        if os.getenv("GROQ_API_KEY"):
            return "process environment"
        if getattr(settings, "GROQ_API_KEY", ""):
            return "platform .env"
        return "unconfigured"

    @property
    def fingerprint(self) -> str:
        """
        Short, non-reversible digest of the key.

        Lets two services be compared for "same credential?" in logs without
        ever printing the secret.
        """
        key = self.api_key
        if not key:
            return "none"
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def describe(self) -> dict[str, Any]:
        """Non-secret snapshot for health endpoints."""
        return {
            "provider": "groq",
            "configured": self.is_configured,
            "key_source": self.source,
            "key_fingerprint": self.fingerprint,
            "model": self.model,
            "fallback_models": list(self.models[1:]),
            "ai_env_file": AI_ENV_FILENAME,
            "ai_env_present": self._path.exists(),
        }


@lru_cache(maxsize=1)
def get_ai_provider_config() -> AIProviderConfig:
    """Process-wide AI provider configuration (env files are read once)."""
    config = AIProviderConfig()
    logger.info(
        "[AI_PROVIDER] groq configured=%s source=%s fingerprint=%s model=%s",
        config.is_configured,
        config.source,
        config.fingerprint,
        config.model,
    )
    return config


class GroqClient:
    """
    Shared async Groq client used by every AI service.

    Replaces the near-identical adapters that previously existed in the intake
    and assistant packages. Returns empty results instead of raising, because
    every caller is required to degrade rather than fail the user's request.
    """

    def __init__(self, config: AIProviderConfig | None = None) -> None:
        self._config = config or get_ai_provider_config()
        self._client: Any = None

        if self._config.is_configured:
            try:
                from groq import AsyncGroq

                self._client = AsyncGroq(api_key=self._config.api_key, timeout=40.0)
            except Exception:
                logger.exception("[AI_PROVIDER] Groq async client init failed")

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def _complete(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        if self._client is None:
            logger.warning("[AI_PROVIDER] no Groq credential available")
            return ""

        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for model in self._config.models:
            try:
                completion = await self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                content = completion.choices[0].message.content or ""

                # A syntactically successful call that returns prose instead of
                # a JSON object is still a failure for a JSON caller, so fall
                # through to the next model rather than surfacing nothing.
                if json_mode and self.parse_json(content) is None:
                    logger.warning(
                        "[AI_PROVIDER] model=%s returned unparseable JSON (%d chars)",
                        model,
                        len(content),
                    )
                    continue

                return content
            except Exception as exc:
                logger.warning("[AI_PROVIDER] model=%s failed: %s", model, exc)
                continue

        logger.error("[AI_PROVIDER] all Groq models failed")
        return ""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Parsed JSON object, or `{}` when unavailable. Never raises."""
        content = await self._complete(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
        return self.parse_json(content) or {}

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 600,
        temperature: float = 0.2,
    ) -> str:
        """Plain text, or `""` when unavailable. Never raises."""
        return (
            await self._complete(
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=False,
            )
        ).strip()

    @staticmethod
    def parse_json(content: str) -> dict[str, Any] | None:
        """
        Parse a model reply into a dict.

        JSON mode makes this usually trivial, but models still occasionally wrap
        output in prose or fences, so the first balanced-looking object is
        retried before giving up.
        """
        text = (content or "").strip()
        if not text:
            return None
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            pass

        match = _JSON_BLOCK_RE.search(text)
        if not match:
            return None
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None

    async def health(self) -> dict[str, Any]:
        """Configuration-level reachability. Makes no billable call."""
        config = self._config
        if not config.is_configured:
            return {
                "status": "unhealthy",
                "provider": "groq",
                "error": (
                    f"No GROQ_API_KEY found in {AI_ENV_FILENAME}, the process "
                    "environment, or the platform .env"
                ),
            }
        if self._client is None:
            return {
                "status": "unhealthy",
                "provider": "groq",
                "error": "Groq async client failed to initialise",
            }
        return {"status": "healthy", **config.describe()}


@lru_cache(maxsize=1)
def get_groq_client() -> GroqClient:
    """
    Process-wide async Groq client.

    Cached so the underlying HTTP connection pool is reused rather than rebuilt
    per request, and so every service demonstrably shares one credential.
    """
    return GroqClient()


def get_groq_api_key() -> str:
    """
    Centralised key accessor for the synchronous `ai_core` components.

    Those use the blocking `groq.Groq` client and are not being rewritten here;
    routing their credential through this function is what puts them on the same
    configuration as everything else.
    """
    return get_ai_provider_config().api_key


def get_groq_model() -> str:
    """Centralised model accessor for the synchronous `ai_core` components."""
    return get_ai_provider_config().model
