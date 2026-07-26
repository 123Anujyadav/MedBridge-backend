import time
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

# Prometheus Metrics Definitions
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total count of HTTP requests handled by the application",
    ["method", "endpoint", "status_code"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

DB_QUERY_TOTAL = Counter(
    "db_query_total",
    "Total database queries executed",
    ["operation", "status"]
)

DB_QUERY_DURATION_SECONDS = Histogram(
    "db_query_duration_seconds",
    "Database query execution latency in seconds",
    ["operation"]
)

REDIS_OPERATIONS_TOTAL = Counter(
    "redis_operations_total",
    "Total Redis cache operations executed",
    ["command", "status"]
)

CELERY_TASKS_TOTAL = Counter(
    "celery_tasks_total",
    "Total Celery background tasks processed",
    ["task_name", "status"]
)

WEBSOCKET_CONNECTIONS_ACTIVE = Gauge(
    "websocket_connections_active",
    "Current active WebSocket connections by role",
    ["role"]
)

WEBSOCKET_MESSAGES_TOTAL = Counter(
    "websocket_messages_total",
    "Total WebSocket messages routed",
    ["type"]
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware recording HTTP request throughput and latency metrics for Prometheus.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        endpoint = request.url.path

        try:
            response: Response = await call_next(request)
            status_code = str(response.status_code)
            duration = time.time() - start_time
            
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)

            return response
        except Exception as e:
            duration = time.time() - start_time
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code="500"
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)

            raise e

def get_metrics_response() -> Response:
    """
    Exposes recorded metrics formatted for Prometheus scrapers.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
