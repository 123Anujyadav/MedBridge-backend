"""
Isolated configuration for the AI Medical Assistant.

Reads `.env.ai-assistant` and nothing else. The rest of the platform continues
to use `app.core.config.settings` backed by `.env`; the two never mix.

`dotenv_values` is used rather than `load_dotenv` deliberately: `load_dotenv`
mutates `os.environ` process-wide, which would leak the assistant's credentials
into every other module and let the main `.env` shadow them. Reading the file
into a private mapping keeps the isolation the integration brief requires.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

ENV_FILENAME = ".env.ai-assistant"

# Backend/ — three levels up from Backend/app/assistant/config.py
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = _BACKEND_ROOT / ENV_FILENAME


class AssistantSettings:
    """
    Configuration for the assistant, sourced only from `.env.ai-assistant`.

    Values from the file take precedence over `os.environ` so that a stale or
    invalid key in the platform's main environment cannot override the
    assistant's working credentials.
    """

    def __init__(self, env_path: Path | None = None) -> None:
        self._path = env_path or ENV_PATH
        self._values: dict[str, Any] = {}

        if self._path.exists():
            self._values = {
                k: v for k, v in dotenv_values(self._path).items() if v is not None
            }
            logger.info(
                "[ASSISTANT_CONFIG] loaded %d keys from %s",
                len(self._values),
                self._path.name,
            )
        else:
            logger.warning(
                "[ASSISTANT_CONFIG] %s not found at %s — assistant will run in "
                "degraded mode",
                ENV_FILENAME,
                self._path,
            )

    def _get(self, key: str, default: str = "") -> str:
        value = self._values.get(key)
        if value is None or value == "":
            value = os.getenv(key, default)
        return (value or "").strip()

    # -- LLM ---------------------------------------------------------------

    @property
    def groq_api_key(self) -> str:
        """Delegates to the centralised provider so there is one credential path."""
        from app.core.ai_provider import get_groq_api_key

        return get_groq_api_key()

    @property
    def groq_model(self) -> str:
        from app.core.ai_provider import get_groq_model

        return get_groq_model()

    # -- Knowledge retrieval ----------------------------------------------

    @property
    def tavily_api_key(self) -> str:
        return self._get("TAVILY_API_KEY")

    @property
    def qdrant_url(self) -> str:
        return self._get("QDRANT_URL")

    @property
    def qdrant_api_key(self) -> str:
        return self._get("QDRANT_API_KEY")

    @property
    def qdrant_collection(self) -> str:
        return self._get("QDRANT_COLLECTION", "medical_assistance_rag")

    @property
    def qdrant_local_path(self) -> str:
        """
        Filesystem path to a local Qdrant store, from `QDRANT_LOCAL_PATH`.

        Configuration-only: there is deliberately no bundled fallback path. An
        implicit default pointing outside the project would silently re-break
        the moment that directory moved or was deleted. Empty means retrieval is
        disabled, which the pipeline handles by answering from model knowledge.
        """
        configured = self._get("QDRANT_LOCAL_PATH")
        if not configured:
            return ""

        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (_BACKEND_ROOT / path).resolve()
        if not path.exists():
            logger.warning(
                "[ASSISTANT_CONFIG] QDRANT_LOCAL_PATH does not exist: %s — "
                "retrieval disabled",
                path,
            )
            return ""
        return str(path)

    # -- Behaviour ---------------------------------------------------------

    @property
    def max_history_messages(self) -> int:
        """Turns of prior conversation replayed into the model (10 Q&A pairs)."""
        try:
            return int(self._get("ASSISTANT_MAX_HISTORY", "20"))
        except ValueError:
            return 20

    @property
    def enable_guardrails(self) -> bool:
        return self._get("ASSISTANT_ENABLE_GUARDRAILS", "true").lower() != "false"

    @property
    def enable_retrieval(self) -> bool:
        return self._get("ASSISTANT_ENABLE_RETRIEVAL", "true").lower() != "false"

    @property
    def is_configured(self) -> bool:
        return bool(self.groq_api_key)

    def describe(self) -> dict[str, Any]:
        """Non-secret snapshot for health endpoints and logs."""
        return {
            "env_file": self._path.name,
            "env_file_present": self._path.exists(),
            "llm_configured": bool(self.groq_api_key),
            "model": self.groq_model,
            "web_search_configured": bool(self.tavily_api_key),
            "retrieval_enabled": self.enable_retrieval,
            "retrieval_corpus_present": bool(self.qdrant_local_path),
            "guardrails_enabled": self.enable_guardrails,
            "max_history_messages": self.max_history_messages,
        }


@lru_cache(maxsize=1)
def get_assistant_settings() -> AssistantSettings:
    """Process-wide assistant settings (file is read once)."""
    return AssistantSettings()
