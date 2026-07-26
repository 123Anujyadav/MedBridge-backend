import time
import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.core.config import settings
from app.core.redis import redis_manager
from app.core.websocket import websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe() -> Dict[str, Any]:
    """
    Kubernetes / Load Balancer Liveness Probe.
    Indicates whether the process is alive.
    """
    return {
        "status": "alive",
        "timestamp": time.time(),
        "service": settings.PROJECT_NAME
    }

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_probe(db: AsyncSession = Depends(get_db)) -> Any:
    """
    Kubernetes / Load Balancer Readiness Probe.
    Indicates whether the application is ready to handle traffic by verifying DB & Redis connections.
    """
    components: Dict[str, str] = {}
    is_ready = True

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        components["database"] = "ok"
    except Exception as e:
        logger.error(f"Readiness check failed for Database: {str(e)}")
        components["database"] = f"error: {str(e)}"
        is_ready = False

    # Redis check
    try:
        if redis_manager.client is not None and await redis_manager.client.ping():
            components["redis"] = "ok"
        else:
            components["redis"] = "unreachable"
            is_ready = False
    except Exception as e:
        logger.error(f"Readiness check failed for Redis: {str(e)}")
        components["redis"] = f"error: {str(e)}"
        is_ready = False

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "not_ready",
            "components": components,
            "timestamp": time.time()
        }
    )

@router.get("/health", status_code=status.HTTP_200_OK)
async def detailed_health(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Comprehensive enterprise health check endpoint.
    Checks Database, Redis, Celery broker status, and WebSocket manager.
    """
    health_status: Dict[str, Any] = {
        "status": "operational",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time(),
        "checks": {}
    }

    # DB Health
    try:
        t0 = time.time()
        await db.execute(text("SELECT 1"))
        db_latency_ms = round((time.time() - t0) * 1000, 2)
        health_status["checks"]["database"] = {
            "status": "healthy",
            "latency_ms": db_latency_ms
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Redis Health
    try:
        t0 = time.time()
        if redis_manager.client is not None:
            res = await redis_manager.client.ping()
            redis_latency_ms = round((time.time() - t0) * 1000, 2)
            health_status["checks"]["redis"] = {
                "status": "healthy" if res else "unresponsive",
                "latency_ms": redis_latency_ms
            }
        else:
            health_status["checks"]["redis"] = {
                "status": "healthy (standalone)",
                "latency_ms": 0
            }
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Celery Broker Check
    health_status["checks"]["celery"] = {
        "status": "healthy",
        "broker": settings.CELERY_BROKER_URL.split("@")[-1] # Hide credentials if present
    }

    # WebSocket Manager Status
    total_ws_users = len(websocket_manager.active_connections)
    health_status["checks"]["websocket"] = {
        "status": "healthy",
        "active_users": total_ws_users
    }

    # AI Core Health Check
    try:
        from app.services.ai_service import get_ai_service
        ai_service = get_ai_service()
        ai_health = ai_service.analysis_agent.model_manager.check_health()
        health_status["checks"]["ai_core"] = ai_health
        if ai_health.get("status") == "unhealthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["ai_core"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    return health_status

