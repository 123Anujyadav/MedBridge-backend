import logging
import time
from typing import Any, AsyncGenerator, Optional
import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)

class InMemoryFallbackCache:
    """
    Thread-safe in-memory cache fallback used when Redis server is unreachable.
    Supports TTL expiration for tokens, rate limits, and user sessions.
    """
    def __init__(self):
        self._store: dict[str, tuple[Any, Optional[float]]] = {}

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        expires_at = (time.time() + ex) if ex else None
        self._store[key] = (value, expires_at)
        return True

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        val, expires_at = self._store[key]
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return str(val)

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                count += 1
        return count

    def incr(self, key: str, amount: int = 1) -> int:
        val = self.get(key)
        new_val = (int(val) if val and val.isdigit() else 0) + amount
        self.set(key, str(new_val))
        return new_val

    def expire(self, key: str, seconds: int) -> bool:
        if key in self._store:
            val, _ = self._store[key]
            self._store[key] = (val, time.time() + seconds)
            return True
        return False


class ResilientRedisClient:
    """
    Resilient Redis Client Wrapper.
    Attempts async operations against configured Redis instance.
    If Redis is down or encounters network timeouts/connection errors,
    it automatically falls back to an in-memory TTL store.
    """
    COOLDOWN_SECONDS = 15.0
    """
    How long to use the fallback exclusively after a failure.

    Without this every operation re-attempted the dead connection and paid the
    full `socket_connect_timeout` again — 1.5s per call. An endpoint making two
    Redis writes therefore took three seconds while Redis was down, which is not
    what "degrades gracefully" should mean. After the cooldown one operation is
    allowed through to discover that Redis has come back.
    """

    def __init__(self):
        self.client: Optional[aioredis.Redis] = None
        self.fallback = InMemoryFallbackCache()
        self.is_redis_available = False
        self._retry_after = 0.0

    def _usable(self) -> bool:
        """True when the client exists and is not inside a failure cooldown."""
        return self.client is not None and time.time() >= self._retry_after

    def _mark_down(self) -> None:
        self.is_redis_available = False
        self._retry_after = time.time() + self.COOLDOWN_SECONDS

    def _mark_up(self) -> None:
        self.is_redis_available = True
        self._retry_after = 0.0

    def init_redis(self) -> None:
        """
        Initializes the async Redis client instance.
        """
        logger.info(f"Initializing Redis client for {settings.REDIS_URL}...")
        try:
            self.client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                max_connections=20,
                socket_timeout=1.5,
                socket_connect_timeout=1.5
            )
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. Fallback cache active.")
            self.is_redis_available = False

    async def ping(self) -> bool:
        """
        Explicit health probe.

        Deliberately ignores the cooldown: this is how a recovered Redis gets
        noticed promptly, and a successful ping clears the breaker for everyone.
        """
        if not self.client:
            return False
        try:
            res = await self.client.ping()
            self._mark_up()
            return res
        except Exception as e:
            logger.warning(f"Redis ping failed ({e}). Operating in Fallback Mode.")
            self._mark_down()
            return False

    async def get(self, name: str) -> Optional[str]:
        if self._usable():
            try:
                val = await self.client.get(name)
                self._mark_up()
                return val
            except Exception as e:
                logger.warning(f"Redis GET failed ({e}). Using in-memory fallback for key '{name}'.")
                self._mark_down()
        return self.fallback.get(name)

    async def set(
        self,
        name: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        if self._usable():
            try:
                res = await self.client.set(name, value, ex=ex, px=px, nx=nx, xx=xx)
                self._mark_up()
                return res
            except Exception as e:
                logger.warning(f"Redis SET failed ({e}). Using in-memory fallback for key '{name}'.")
                self._mark_down()
        return self.fallback.set(name, value, ex=ex)

    async def delete(self, *names: str) -> int:
        if self._usable():
            try:
                res = await self.client.delete(*names)
                self._mark_up()
                return res
            except Exception as e:
                logger.warning(f"Redis DELETE failed ({e}). Using in-memory fallback.")
                self._mark_down()
        return self.fallback.delete(*names)

    async def incr(self, name: str, amount: int = 1) -> int:
        if self._usable():
            try:
                res = await self.client.incr(name, amount=amount)
                self._mark_up()
                return res
            except Exception as e:
                logger.warning(f"Redis INCR failed ({e}). Using in-memory fallback for key '{name}'.")
                self._mark_down()
        return self.fallback.incr(name, amount=amount)

    async def expire(self, name: str, time_sec: int) -> bool:
        if self._usable():
            try:
                res = await self.client.expire(name, time_sec)
                self._mark_up()
                return res
            except Exception as e:
                logger.warning(f"Redis EXPIRE failed ({e}). Using in-memory fallback for key '{name}'.")
                self._mark_down()
        return self.fallback.expire(name, time_sec)

    async def close(self) -> None:
        # Not `_usable()`: shutdown must release the connection even mid-cooldown.
        if self.client:
            try:
                await self.client.aclose()
            except Exception:
                pass
            logger.info("Redis client connection closed.")

redis_manager = ResilientRedisClient()

async def get_redis() -> AsyncGenerator[Any, None]:
    """
    Dependency yielding the resilient Redis client instance.
    """
    if redis_manager.client is None:
        redis_manager.init_redis()
    yield redis_manager
