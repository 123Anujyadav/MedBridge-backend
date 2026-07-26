import logging
import time
import uuid
from contextvars import ContextVar
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Context variables for tracing request lifecycle
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")

class TracingFilter(logging.Filter):
    """
    Logging filter injecting Correlation ID, Request ID, and Trace ID into all log records.
    """
    def filter(self, record):
        record.correlation_id = correlation_id_ctx.get() or "no-correlation-id"
        record.request_id = request_id_ctx.get() or "no-request-id"
        record.trace_id = trace_id_ctx.get() or "no-trace-id"
        return True

def register_tracing_filter():
    root_logger = logging.getLogger()
    filter_exists = any(isinstance(f, TracingFilter) for f in root_logger.filters)
    if not filter_exists:
        root_logger.addFilter(TracingFilter())

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured HTTP Observability & Tracing Middleware.
    Extracts or generates request headers and logs execution metrics.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        register_tracing_filter()

        corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())

        token_corr = correlation_id_ctx.set(corr_id)
        token_req = request_id_ctx.set(req_id)
        token_trace = trace_id_ctx.set(trace_id)

        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            f"Incoming request: {request.method} {request.url.path} from {client_ip}",
            extra={
                "http_method": request.method,
                "path": request.url.path,
                "client_ip": client_ip,
            }
        )

        try:
            response: Response = await call_next(request)
            process_time_ms = round((time.time() - start_time) * 1000, 2)
            
            response.headers["X-Correlation-ID"] = corr_id
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Trace-ID"] = trace_id

            logger.info(
                f"Completed request: {request.method} {request.url.path} - {response.status_code} ({process_time_ms}ms)",
                extra={
                    "status_code": response.status_code,
                    "duration_ms": process_time_ms
                }
            )
            return response
        except Exception as e:
            process_time_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(
                f"Failed request: {request.method} {request.url.path} - Exception: {str(e)} ({process_time_ms}ms)",
                exc_info=True,
                extra={
                    "duration_ms": process_time_ms,
                    "error": str(e)
                }
            )
            raise e
        finally:
            correlation_id_ctx.reset(token_corr)
            request_id_ctx.reset(token_req)
            trace_id_ctx.reset(token_trace)
