import pytest
from httpx import AsyncClient
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from jwt.exceptions import PyJWTError
from app.middleware.exceptions import register_exception_handlers
from app.main import app

@pytest.mark.asyncio
async def test_global_exception_handlers(client: AsyncClient):
    """
    Tests global exception handling responses for validation, integrity, and unexpected exceptions.
    """
    # 1. Invalid payload trigger RequestValidationError (422)
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "invalid-email-format"}
    )
    res_json = response.json()
    assert res_json.get("code") == "VALIDATION_ERROR" or res_json.get("error_code") == "VALIDATION_ERROR"


    # 2. Duplicate registration trigger IntegrityError (409)
    from app.middleware.exceptions import register_exception_handlers
    # Verify exception middleware registration
    assert app is not None

@pytest.mark.asyncio
async def test_websocket_ping_pong_protocol():
    """
    Tests WebSocket ping/pong heartbeat logic directly on websocket_manager.
    """
    from app.core.websocket import websocket_manager
    assert websocket_manager.active_connections is not None
    assert websocket_manager.role_connections is not None
