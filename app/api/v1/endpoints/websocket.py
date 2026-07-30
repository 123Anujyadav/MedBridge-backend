import logging
import time
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.repositories.user import user_repository
from app.core.exceptions import AuthorizationException
from app.core.identity import InvalidTokenError, get_token_verifier
from app.core.websocket import websocket_manager
from app.services.doctor_access import assert_doctor_may_practise

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Persistent bi-directional WebSocket connection.
    Authenticates using JWT token passed as query parameter.
    Enforces connection limits and heartbeat checks.
    """
    try:
        # Same verifier the HTTP dependencies use, so the socket can never
        # accept a token the REST API would reject.
        identity = get_token_verifier().verify(token).require_access_token()

        # Resolve the token's subject the same way the HTTP dependency does.
        #
        # `identity.subject` is only the *local* user id under the built-in
        # provider. Under Supabase it is the provider's own id, so looking it up
        # as a primary key never matched and every socket was refused — realtime
        # was silently dead in any Supabase deployment. `resolve_local_user`
        # handles both, and is the same function `get_current_user` uses, so the
        # socket can never accept an identity the REST API would reject.
        from app.services.identity_link import resolve_local_user

        user = await resolve_local_user(db, identity)
        if not user or not user.is_active:
            logger.warning(
                "Rejected WebSocket connection: subject %s not active or not found.",
                identity.subject,
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # The local id, not the provider's: `send_personal_message` is keyed by
        # it, and every payload carries local ids.
        user_id = str(user.id)

        # An unapproved or rejected clinician must not receive the live
        # clinical broadcasts either — the socket carries case and appointment
        # events for the doctor role.
        if user.role == "doctor":
            try:
                await assert_doctor_may_practise(db, user)
            except AuthorizationException:
                logger.warning(
                    f"Rejected WebSocket connection: doctor {user_id} not approved."
                )
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

        role = user.role

    except InvalidTokenError as e:
        logger.warning(f"Rejected WebSocket connection: Token validation failed: {str(e)}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    except ValueError as e:
        logger.warning(f"Rejected WebSocket connection: {str(e)}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Accept and register connection with connection limits
    connected = await websocket_manager.connect(websocket, user_id, role)
    if not connected:
        return

    try:
        # Loop listening for incoming messages (ping/pong heartbeats or commands)
        while True:
            data = await websocket.receive_text()
            if data.lower() == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                continue
            elif data == "trigger_emergency":
                await websocket_manager.broadcast_to_role(
                    {"type": "emergency", "message": "Emergency alert code blue!"},
                    "doctor"
                )
            
            logger.debug(f"Received text from client {user_id}: {data}")
            await websocket.send_json({"status": "acknowledged", "received": data})

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, user_id, role)
        logger.info(f"WebSocket disconnected gracefully for user {user_id}.")
    except Exception as e:
        websocket_manager.disconnect(websocket, user_id, role)
        logger.error(f"WebSocket connection error for user {user_id}: {str(e)}")
