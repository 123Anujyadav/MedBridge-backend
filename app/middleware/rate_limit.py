import logging
import time
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings
from app.core.redis import redis_manager

logger = logging.getLogger(__name__)

# Sensitive paths and their corresponding max limits per 60-second window
PROTECTED_PATHS = {
    "/auth/login": settings.RATE_LIMIT_LOGIN,
    "/auth/signup/patient": settings.RATE_LIMIT_REGISTER,
    "/auth/signup/doctor": settings.RATE_LIMIT_REGISTER,
    "/auth/forgot-password": settings.RATE_LIMIT_LOGIN,
    "/auth/reset-password": settings.RATE_LIMIT_LOGIN,
    "/auth/verify-account": settings.RATE_LIMIT_LOGIN,
    "/patient/emergency": settings.RATE_LIMIT_EMERGENCY,
}

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
            if normalized_path == route_pattern or normalized_path.startswith(route_pattern):
                limit = max_limit
                # Count against the *pattern*, not the exact path. Otherwise
                # `/auth/login` and `/auth/login/doctor` keep separate counters
                # and an attacker gets the budget twice over for one account —
                # they are two doors into the same credential check.
                bucket = route_pattern
                break

        if limit is not None:
            client_ip = request.client.host if request.client else "unknown"
            window = int(time.time()) // 60
            key = f"rate_limit:{client_ip}:{bucket}:{window}"

            try:
                count = await redis_manager.incr(key)
                if count == 1:
                    await redis_manager.expire(key, 60)
                
                if count > limit:
                    logger.warning(f"Rate limit exceeded for IP {client_ip} on path {path}")
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
