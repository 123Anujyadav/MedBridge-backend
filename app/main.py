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
    yield
    # Shutdown actions
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply Hardening, Observability, and Monitoring Middlewares
app.add_middleware(SecurityHeadersMiddleware)


app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(PrometheusMiddleware)

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
