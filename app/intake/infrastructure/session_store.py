"""
Redis-backed session store implementing `SessionStorePort`.

Built on the project's existing `ResilientRedisClient`, so an intake
conversation survives a Redis outage via the same in-memory fallback the rest of
the platform already relies on.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.intake.domain.entities import IntakeSession

logger = logging.getLogger(__name__)

_KEY_PREFIX = "intake_session:"
DEFAULT_TTL_SECONDS = 60 * 60


class RedisSessionStore:
    """Serialises `IntakeSession` aggregates to JSON in Redis."""

    def __init__(self, redis: Any, *, key_prefix: str = _KEY_PREFIX) -> None:
        self._redis = redis
        self._prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def get(self, session_id: str) -> IntakeSession | None:
        """
        Load a session, or None if absent, expired or corrupt.

        Corrupt payloads are treated as absent rather than raised: a bad cache
        entry should force a fresh intake, not hand the patient a 500.
        """
        try:
            raw = await self._redis.get(self._key(session_id))
        except Exception:
            logger.exception("[INTAKE_SESSION_READ_FAILED] session=%s", session_id)
            return None

        if not raw:
            return None

        try:
            payload = json.loads(raw)
            return IntakeSession.from_dict(payload)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            logger.exception(
                "[INTAKE_SESSION_CORRUPT] session=%s — treating as missing", session_id
            )
            return None

    async def save(
        self, session: IntakeSession, ttl_seconds: int | None = None
    ) -> None:
        """
        Persist a session.

        A write failure is logged and swallowed. The caller has already computed
        a valid response; losing the cache write costs the patient continuity,
        while raising would cost them the whole turn.
        """
        ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_TTL_SECONDS
        try:
            payload = json.dumps(session.to_dict(), ensure_ascii=False)
            await self._redis.set(self._key(session.session_id), payload, ex=ttl)
        except Exception:
            logger.exception(
                "[INTAKE_SESSION_WRITE_FAILED] session=%s", session.session_id
            )

    async def delete(self, session_id: str) -> None:
        try:
            await self._redis.delete(self._key(session_id))
        except Exception:
            logger.exception("[INTAKE_SESSION_DELETE_FAILED] session=%s", session_id)
