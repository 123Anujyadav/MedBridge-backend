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

import asyncio
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

FALLBACK_MODELS: tuple[str, ...] = ("llama-3.1-8b-instant", "openai/gpt-oss-20b")
"""
Tried in order after the configured model when a request fails.

Every entry is verified to serve `response_format={"type": "json_object"}` on a
current Groq account. `llama3-70b-8192` and `gemma2-9b-it` used to sit here and
are now **decommissioned** — Groq answers them with
`400 model_decommissioned`, so they burned a retry each and could never
succeed. Check https://console.groq.com/docs/deprecations before editing.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_MAX_ATTEMPTS_PER_MODEL = 3
"""Attempts per model for *transient* failures (timeout, 429, 5xx)."""

_RETRY_BASE_DELAY = 0.5
"""Seconds; doubled per attempt. Keeps total latency inside the 40s timeout."""

# Failure taxonomy. The distinction matters: retrying a 401 against three
# models is three guaranteed failures and ~3x the latency before the caller
# degrades, whereas a 429 genuinely does succeed on a second attempt.
_AUTH = "auth"
_MODEL_UNUSABLE = "model_unusable"
_TRANSIENT = "transient"
_UNKNOWN = "unknown"


def _classify_error(exc: BaseException) -> str:
    """
    Bucket a Groq SDK exception so the retry policy can act on it.

    Falls back to message inspection when the SDK's typed exceptions are not
    importable, so classification never itself becomes a failure point.
    """
    try:
        from groq import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            InternalServerError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
        )

        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return _AUTH
        if isinstance(exc, NotFoundError):
            return _MODEL_UNUSABLE
        if isinstance(
            exc,
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError),
        ):
            return _TRANSIENT
    except ImportError:  # pragma: no cover - groq is a hard dependency
        pass

    text = str(exc).lower()
    if "invalid api key" in text or "401" in text or "unauthorized" in text:
        return _AUTH
    if "decommissioned" in text or "does not exist" in text or "model_not_found" in text:
        return _MODEL_UNUSABLE
    if any(t in text for t in ("timeout", "rate limit", "429", "502", "503", "504")):
        return _TRANSIENT
    return _UNKNOWN


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
    def api_key_candidates(self) -> list[tuple[str, str]]:
        """
        Every distinct credential available, best-first, as `(source, key)`.

        Deployments routinely end up with more than one Groq key — this repo has
        a live key in `.env.ai-assistant` and a **revoked** one in the platform
        `.env`. Resolution used to stop at the first non-empty value, so whether
        the assistant worked depended on which file happened to be present. The
        client now walks this list on an authentication failure, so one stale key
        can no longer take the assistant down while a valid one sits unused.
        """
        primary = self.api_key
        if not primary:
            return []

        ordered: list[tuple[str, str]] = [(self.source, primary)]
        seen = {primary}
        for label, value in (
            (AI_ENV_FILENAME, self._file_values.get("GROQ_API_KEY")),
            ("process environment", os.getenv("GROQ_API_KEY")),
            ("platform .env", getattr(settings, "GROQ_API_KEY", None)),
        ):
            key = str(value).strip() if value else ""
            if key and key not in seen:
                ordered.append((label, key))
                seen.add(key)
        return ordered

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

        self._candidates = self._config.api_key_candidates
        self._candidate_index = 0
        self._last_error: str | None = None
        """
        Why the most recent call produced nothing.

        Callers return empty results by contract, which used to make "no
        credential", "revoked credential" and "model decommissioned"
        indistinguishable from "the model had nothing to say". Health endpoints
        surface this instead of guessing.
        """

        if not self._candidates:
            self._last_error = (
                f"No GROQ_API_KEY found in {AI_ENV_FILENAME}, the process "
                "environment, or the platform .env"
            )
            logger.error(
                "[AI_PROVIDER] NOT CONFIGURED — %s. Every AI feature will return "
                "degraded responses until a key is set.",
                self._last_error,
            )
            return

        self._open_client()

    def _open_client(self) -> bool:
        """Build the async client for the current candidate credential."""
        try:
            from groq import AsyncGroq
        except Exception:
            self._last_error = "the groq package is not importable"
            logger.exception("[AI_PROVIDER] cannot import groq.AsyncGroq")
            return False

        source, key = self._candidates[self._candidate_index]
        try:
            self._client = AsyncGroq(api_key=key, timeout=40.0)
        except Exception as exc:
            self._last_error = f"Groq client init failed: {exc}"
            logger.exception(
                "[AI_PROVIDER] Groq async client init failed for credential "
                "source=%s",
                source,
            )
            return False

        logger.info(
            "[AI_PROVIDER] client ready source=%s fingerprint=%s models=%s "
            "(%d credential(s) available)",
            source,
            hashlib.sha256(key.encode()).hexdigest()[:8],
            list(self._config.models),
            len(self._candidates),
        )
        return True

    def _advance_credential(self) -> bool:
        """
        Rotate to the next distinct credential after an authentication failure.

        Returns False once every candidate has been rejected, which is the only
        point at which an auth problem is genuinely unrecoverable.
        """
        if self._candidate_index + 1 >= len(self._candidates):
            return False

        dead_source, _ = self._candidates[self._candidate_index]
        self._candidate_index += 1
        next_source, _ = self._candidates[self._candidate_index]
        logger.warning(
            "[AI_PROVIDER] credential from %s was rejected; retrying with the key "
            "from %s",
            dead_source,
            next_source,
        )
        return self._open_client()

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    @property
    def last_error(self) -> str | None:
        """Reason the most recent call returned nothing, or None on success."""
        return self._last_error

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
            logger.error(
                "[AI_PROVIDER] call skipped — no usable Groq credential (%s)",
                self._last_error,
            )
            return ""

        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        failures: list[str] = []

        for model in self._config.models:
            attempt = 0
            while attempt < _MAX_ATTEMPTS_PER_MODEL:
                attempt += 1
                try:
                    completion = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    content = completion.choices[0].message.content or ""

                    # A syntactically successful call that returns prose instead
                    # of a JSON object is still a failure for a JSON caller, so
                    # fall through to the next model rather than surfacing
                    # nothing. Retrying the same model is pointless here — the
                    # model simply does not honour the format.
                    if json_mode and self.parse_json(content) is None:
                        logger.warning(
                            "[AI_PROVIDER] model=%s returned unparseable JSON "
                            "(%d chars): %.200r",
                            model,
                            len(content),
                            content,
                        )
                        failures.append(f"{model}: unparseable JSON")
                        break

                    if not content.strip():
                        logger.warning(
                            "[AI_PROVIDER] model=%s returned empty content", model
                        )
                        failures.append(f"{model}: empty content")
                        break

                    self._last_error = None
                    return content

                except Exception as exc:
                    kind = _classify_error(exc)
                    detail = f"{type(exc).__name__}: {exc}"
                    failures.append(f"{model}: {detail}")

                    # Full traceback at every failure — the old code logged only
                    # str(exc) at WARNING, which made a revoked key look
                    # identical to a network blip in production logs.
                    logger.warning(
                        "[AI_PROVIDER] model=%s attempt=%d/%d kind=%s failed: %s",
                        model,
                        attempt,
                        _MAX_ATTEMPTS_PER_MODEL,
                        kind,
                        detail,
                        exc_info=True,
                    )

                    if kind == _AUTH:
                        # The key, not the model, is wrong. Rotate credentials
                        # and retry this same model before giving up.
                        if self._advance_credential():
                            attempt = 0
                            continue
                        self._last_error = (
                            f"Groq rejected every configured credential ({detail}). "
                            "Set a valid GROQ_API_KEY."
                        )
                        logger.error(
                            "[AI_PROVIDER] AUTHENTICATION FAILED for all %d "
                            "credential(s) — fingerprint=%s source=%s. %s",
                            len(self._candidates),
                            self._config.fingerprint,
                            self._config.source,
                            self._last_error,
                        )
                        return ""

                    if kind == _MODEL_UNUSABLE:
                        logger.error(
                            "[AI_PROVIDER] model=%s is decommissioned or unavailable "
                            "on this account — remove it from GROQ_MODEL/"
                            "FALLBACK_MODELS. Trying the next model.",
                            model,
                        )
                        break

                    if kind == _TRANSIENT and attempt < _MAX_ATTEMPTS_PER_MODEL:
                        delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        logger.info(
                            "[AI_PROVIDER] transient failure on %s; retrying in "
                            "%.1fs",
                            model,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    break

        self._last_error = "all Groq models failed: " + "; ".join(failures)
        logger.error(
            "[AI_PROVIDER] ALL MODELS FAILED after exhausting %d model(s) and %d "
            "credential(s). Callers will degrade. Failures: %s",
            len(self._config.models),
            len(self._candidates),
            " | ".join(failures),
        )
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

    async def health(self, *, probe: bool = False) -> dict[str, Any]:
        """
        Report provider readiness.

        Config-level by default, which makes no billable call. Pass `probe=True`
        for a real round trip: a *revoked* key is indistinguishable from a valid
        one at configuration level, and reporting "healthy" while every
        completion 401s is precisely how a credential outage stayed invisible
        here. The probe spends a handful of tokens to tell the truth.
        """
        config = self._config
        if not config.is_configured or not self._candidates:
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
                "error": self._last_error or "Groq async client failed to initialise",
            }

        payload: dict[str, Any] = {"status": "healthy", **config.describe()}
        payload["credentials_available"] = len(self._candidates)
        if self._last_error:
            payload["last_error"] = self._last_error

        if probe:
            reply = await self.complete_text(
                system_prompt="Reply with the single word OK.",
                user_content="Health check.",
                max_tokens=5,
                temperature=0.0,
            )
            if reply:
                payload["status"] = "healthy"
                payload["probe"] = "ok"
                payload.pop("last_error", None)
            else:
                payload["status"] = "unhealthy"
                payload["probe"] = "failed"
                payload["error"] = self._last_error or "live probe returned nothing"

        return payload


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
