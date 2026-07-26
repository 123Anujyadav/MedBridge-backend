import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jwt.exceptions import PyJWTError
from sqlalchemy.exc import DBAPIError, IntegrityError, SQLAlchemyError
import redis.exceptions as redis_exceptions
from app.core.exceptions import (
    AronofyException,
    AuthenticationException,
    AuthorizationException,
    BusinessRuleValidationException,
    ConsentRequiredException,
    EntityNotFoundException,
)

logger = logging.getLogger(__name__)

def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers enterprise global exception handlers intercepting validation,
    authentication, permission, database, redis, and system-level errors.
    Always returns structured JSON error responses.
    """
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Request validation failed for {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "message": "Request validation failed.",
                "code": "VALIDATION_ERROR",
                "details": str(exc.errors())
            }
        )

    @app.exception_handler(EntityNotFoundException)
    async def entity_not_found_handler(request: Request, exc: EntityNotFoundException):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "success": False,
                "message": exc.message,
                "code": "ENTITY_NOT_FOUND",
                "details": str(exc)
            }
        )

    @app.exception_handler(AuthenticationException)
    async def authentication_handler(request: Request, exc: AuthenticationException):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
            content={
                "success": False,
                "message": exc.message,
                "code": "UNAUTHENTICATED",
                "details": "Invalid email or password."
            }
        )

    @app.exception_handler(PyJWTError)
    async def jwt_exception_handler(request: Request, exc: PyJWTError):
        logger.warning(f"JWT verification failure on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
            content={
                "success": False,
                "message": "Invalid or expired authentication token.",
                "code": "INVALID_TOKEN",
                "details": str(exc)
            }
        )

    @app.exception_handler(AuthorizationException)
    async def authorization_handler(request: Request, exc: AuthorizationException):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "success": False,
                "message": exc.message,
                "code": "UNAUTHORIZED_ROLE",
                "details": "Access restricted for current user role."
            }
        )

    @app.exception_handler(ConsentRequiredException)
    async def consent_required_handler(request: Request, exc: ConsentRequiredException):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "success": False,
                "message": exc.message,
                "code": "CONSENT_REQUIRED",
                "details": "Patient consent required for this action."
            }
        )

    @app.exception_handler(BusinessRuleValidationException)
    async def business_rule_handler(request: Request, exc: BusinessRuleValidationException):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "message": exc.message,
                "code": "BUSINESS_RULE_VALIDATION_FAILED",
                "details": str(exc)
            }
        )

    @app.exception_handler(redis_exceptions.RedisError)
    async def redis_exception_handler(request: Request, exc: redis_exceptions.RedisError):
        logger.warning(f"Intercepted Redis Connection/Command Exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "message": "Cache/Session storage degraded. In-memory fallback engaged.",
                "code": "REDIS_UNAVAILABLE",
                "details": str(exc)
            }
        )

    @app.exception_handler(IntegrityError)
    async def db_integrity_handler(request: Request, exc: IntegrityError):
        logger.error(f"Database constraint violation on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "success": False,
                "message": "Database resource conflict or unique constraint violation.",
                "code": "RESOURCE_CONFLICT",
                "details": str(exc.orig) if hasattr(exc, 'orig') else str(exc)
            }
        )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
        logger.error(f"Database operational error on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "Database operation failed. Transaction rolled back.",
                "code": "DATABASE_ERROR",
                "details": str(exc)
            }
        )

    @app.exception_handler(AronofyException)
    async def base_aronofy_handler(request: Request, exc: AronofyException):
        logger.error(f"Domain exception intercepted: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": exc.message,
                "code": "INTERNAL_DOMAIN_ERROR",
                "details": str(exc)
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.critical(f"Unhandled system error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "An unexpected server error occurred.",
                "code": "INTERNAL_SERVER_ERROR",
                "details": str(exc)
            }
        )
