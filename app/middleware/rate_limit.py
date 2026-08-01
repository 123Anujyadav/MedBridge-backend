import base64
import binascii
import json
import logging
import time
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings
from app.core.redis import redis_manager

logger = logging.getLogger(__name__)

# Sensitive paths and their corresponding max limits per 60-second window.
#
# Matched in order, first match wins, so the specific AI read routes are listed
# before the `/ai` catch-all that covers everything generative. The catch-all is
# deliberate: a route added under `/ai` later is rate limited from the moment it
# exists, rather than from the moment somebody remembers to add it here.
PROTECTED_PATHS = {
    "/auth/login": settings.RATE_LIMIT_LOGIN,
    "/auth/signup/patient": settings.RATE_LIMIT_REGISTER,
    "/auth/signup/doctor": settings.RATE_LIMIT_REGISTER,
    "/auth/forgot-password": settings.RATE_LIMIT_LOGIN,
    "/auth/reset-password": settings.RATE_LIMIT_LOGIN,
    "/auth/verify-account": settings.RATE_LIMIT_LOGIN,
    "/patient/emergency": settings.RATE_LIMIT_EMERGENCY,
    # Reads: a database query, not a model call.
    "/ai/health": settings.RATE_LIMIT_AI_READ,
    "/ai/intake/health": settings.RATE_LIMIT_AI_READ,
    "/ai/assistant/health": settings.RATE_LIMIT_AI_READ,
    "/ai/assistant/conversations": settings.RATE_LIMIT_AI_READ,
    # Everything else under /ai reaches an LLM. One shared budget, so the total
    # spend a single caller can cause is bounded rather than being bounded once
    # per route: separate per-route budgets would let one caller run the
    # assistant, the intake agent and report analysis concurrently, each at the
    # full allowance.
    "/ai": settings.RATE_LIMIT_AI,
}

AI_PATH_PREFIX = "/ai"

_SUBJECT_CLAIMS = ("sub", "user_id", "uid")


def _token_subject(request: Request) -> str | None:
    """
    The account a bearer token names, read without verifying it.

    Used only to label a rate-limit bucket, never to decide access — the route's
    own dependencies still authenticate the request, and they run after this
    middleware. That ordering is what makes an unverified read safe here: a
    forged subject buys a fresh bucket for requests that are about to be
    rejected with 401 anyway, and never reaches the model call the budget
    exists to protect.

    Bucketing authenticated AI traffic by account rather than by address is
    what stops one credential spending without limit simply by moving between
    addresses, and equally stops a clinic behind one NAT address from sharing a
    single allowance between everybody in the building.
    """
    header = request.headers.get("Authorization") or ""
    if not header.lower().startswith("bearer "):
        return None

    token = header[7:].strip()
    parts = token.split(".")
    if len(parts) != 3:
        return None

    try:
        payload_segment = parts[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        claims = json.loads(decoded)
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(claims, dict):
        return None

    for claim in _SUBJECT_CLAIMS:
        value = claims.get(claim)
        if isinstance(value, str) and value.strip():
            return value.strip()[:128]
    return None

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate Limiting Middleware targeting sensitive authentication,
    verification, and emergency API endpoints.
    Uses ResilientRedisClient (falling back to in-memory store if Redis is offline).
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        # Disable rate limiting during automated testing
        import os
        if settings.ENVIRONMENT.lower() in ("testing", "test") or os.getenv("TESTING", "0") in ("1", "true"):
            return await call_next(request)

        path = request.url.path

        
        # Strip API prefix if needed to normalize route matching
        normalized_path = path
        if path.startswith(settings.API_V1_STR):
            normalized_path = path[len(settings.API_V1_STR):]

        limit = None
        bucket = normalized_path
        for route_pattern, max_limit in PROTECTED_PATHS.items():
            # Match whole path segments, not bare prefixes. A plain `startswith`
            # also catches every *sibling* route whose name merely begins the
            # same way: `/patient/emergency-profile` — editing a stored address
            # — was being throttled at the panic-button's ten-per-minute
            # because it starts with `/patient/emergency`. Requiring the next
            # character to be `/` keeps `/auth/login/doctor` sharing the
            # `/auth/login` budget, which is deliberate, while leaving
            # `-profile` alone.
            if (normalized_path == route_pattern
                    or normalized_path.startswith(route_pattern + "/")):
                limit = max_limit
                # Count against the *pattern*, not the exact path. Otherwise
                # `/auth/login` and `/auth/login/doctor` keep separate counters
                # and an attacker gets the budget twice over for one account —
                # they are two doors into the same credential check.
                bucket = route_pattern
                break

        if limit is not None:
            client_ip = request.client.host if request.client else "unknown"

            # Who the budget belongs to. Address everywhere, except on the AI
            # routes, where an authenticated caller is charged against their
            # account: those are the only routes that cost money per request,
            # and they are always authenticated, so the account is both the
            # more accurate identity and the one an attacker cannot change by
            # moving address. Anonymous traffic — and every non-AI route —
            # keeps the existing per-address behaviour unchanged.
            caller = client_ip
            if normalized_path.startswith(AI_PATH_PREFIX):
                subject = _token_subject(request)
                if subject:
                    caller = f"user:{subject}"

            window = int(time.time()) // 60
            key = f"rate_limit:{caller}:{bucket}:{window}"

            try:
                count = await redis_manager.incr(key)
                if count == 1:
                    await redis_manager.expire(key, 60)
                
                if count > limit:
                    logger.warning(
                        f"Rate limit exceeded for {caller} (IP {client_ip}) "
                        f"on path {path}"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "success": False,
                            "message": "Too many requests. Please try again later.",
                            "code": "RATE_LIMIT_EXCEEDED",
                            "details": f"Maximum limit of {limit} requests per minute reached.",
                            "retry_after_seconds": 60 - (int(time.time()) % 60)
                        },
                        headers={"Retry-After": "60"}
                    )
            except Exception as e:
                logger.warning(f"Rate limiter exception, allowing request: {str(e)}")

        return await call_next(request)
