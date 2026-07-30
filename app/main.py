import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.redis import redis_manager
from app.core.prometheus import PrometheusMiddleware, get_metrics_response
from app.middleware.exceptions import register_exception_handlers
from app.middleware.logging import LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security import SecurityHeadersMiddleware
from app.api.v1.endpoints.health import router as health_router

# Setup logging configuration first
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown hooks for resource lifecycle mapping.
    """
    # Startup actions
    redis_manager.init_redis()

    # The emergency-communication retry sweeper.
    #
    # Celery runs the same sweep on a beat schedule, but Celery needs a
    # reachable broker and an emergency alert must not depend on one. Both
    # claim rows with the same atomic conditional update, so running both is
    # safe and running either alone is sufficient. Skipped under pytest, where
    # a background loop would outlive the test that started it.
    sweeper = None
    if "pytest" not in sys.modules:
        import asyncio

        from app.services.emergency_comms import retry_sweep_loop

        sweeper = asyncio.create_task(retry_sweep_loop())

    yield

    # Shutdown actions
    if sweeper is not None:
        sweeper.cancel()
        try:
            await sweeper
        except (asyncio.CancelledError, Exception):
            pass
    await redis_manager.close()

is_prod = settings.ENVIRONMENT == "production"

# Instantiate FastAPI Application with Swagger Security in Production
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise HIPAA-compliant Backend for MedBridge Healthcare System.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if is_prod else "/docs",
    redoc_url=None if is_prod else "/redoc",
    openapi_url=None if is_prod else f"{settings.API_V1_STR}/openapi.json"
)

# Apply CORS configurations with resilient fallback defaults
cors_origins = [str(origin) for origin in settings.BACKEND_CORS_ORIGINS] if settings.BACKEND_CORS_ORIGINS else []
default_dev_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"]
for default_origin in default_dev_origins:
    if default_origin not in cors_origins:
        cors_origins.append(default_origin)

if is_prod and "*" in cors_origins:
    cors_origins = [o for o in cors_origins if o != "*"]

app.state.cors_origins = cors_origins
"""
The resolved allow-list, published for the 500 handler.

Starlette treats a handler registered for bare `Exception` specially: it is
invoked by `ServerErrorMiddleware`, which wraps the *entire* application and
therefore sits outside CORS. That one response can never pick the headers up on
its way out, so the handler has to attach them itself — and to do that it needs
the same list configured here, without importing this module and creating a
cycle.
"""

# Apply Hardening, Observability, and Monitoring Middlewares
app.add_middleware(SecurityHeadersMiddleware)


app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(PrometheusMiddleware)

# CORS is added LAST, which in Starlette makes it the OUTERMOST middleware —
# `add_middleware` prepends, so the last one added is the first one entered and
# the last one left.
#
# It has to be outermost because it is the only thing that attaches
# `Access-Control-Allow-Origin`, and a response that never reaches it is a
# response the browser refuses to hand to the page. Previously CORS was added
# first and so sat *innermost*: any middleware outside it that answered on its
# own — most importantly `RateLimitMiddleware` returning 429 — produced a reply
# with no CORS headers at all. To the browser that is not "rate limited", it is
# a CORS failure, which surfaces in axios as an opaque "Network Error" with no
# status code. That is the deployed sign-in failure this ordering fixes.
#
# Two consequences worth knowing about:
#   * Every response now carries the headers, including the ones produced by
#     the exception handlers (401/403/404/422/500) and by the rate limiter.
#   * Preflight `OPTIONS` requests are answered by CORS before they reach the
#     rate limiter, so a preflight no longer consumes part of the caller's
#     per-minute budget. A sign-in attempt used to cost two.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Custom Exception Map Handlers
register_exception_handlers(app)

# Mount Health & Probes Router
app.include_router(health_router, tags=["Monitoring"])

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics scraping endpoint.
    """
    return get_metrics_response()

from app.api.v1.router import api_router

# Mount V1 APIs Router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get(f"{settings.API_V1_STR}/", tags=["Core"])
async def root():
    return {"message": "Welcome to MedBridge API. Secure Gateway V1."}
