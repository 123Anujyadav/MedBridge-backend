import logging
from typing import Dict, List
from fastapi import WebSocket, status
from app.core.prometheus import WEBSOCKET_CONNECTIONS_ACTIVE, WEBSOCKET_MESSAGES_TOTAL

logger = logging.getLogger(__name__)

MAX_CONNECTIONS_PER_USER = 5

class ConnectionManager:
    """
    Enterprise WebSocket Connection Manager providing role-based broadcasting,
    private messaging, connection limits, and active socket tracking.
    """
    def __init__(self):
        # Maps user_id -> List of active WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Maps role -> List of active WebSocket connections
        self.role_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str, role: str) -> bool:
        """
        Accepts connection, enforces max connection limits, and registers socket.
        """
        # Check connection limits
        user_conns = self.active_connections.get(user_id, [])
        if len(user_conns) >= MAX_CONNECTIONS_PER_USER:
            logger.warning(f"WebSocket connection rejected for user {user_id}: Exceeded max connection limit ({MAX_CONNECTIONS_PER_USER}).")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Max concurrent connections exceeded.")
            return False

        await websocket.accept()
        
        # User-specific mapping
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

        # Role-specific mapping
        if role not in self.role_connections:
            self.role_connections[role] = []
        self.role_connections[role].append(websocket)

        WEBSOCKET_CONNECTIONS_ACTIVE.labels(role=role).inc()
        logger.info(f"WebSocket connected for user {user_id} with role {role}.")
        return True

    def disconnect(self, websocket: WebSocket, user_id: str, role: str):
        """
        Deregisters and removes socket from active connection registries cleanly.
        """
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        if role in self.role_connections:
            if websocket in self.role_connections[role]:
                self.role_connections[role].remove(websocket)
            if not self.role_connections[role]:
                del self.role_connections[role]

        WEBSOCKET_CONNECTIONS_ACTIVE.labels(role=role).dec()
        logger.info(f"WebSocket disconnected for user {user_id} with role {role}.")

    async def send_personal_message(self, message: dict, user_id: str):
        """
        Sends JSON payload to all active sockets of a specific user.
        """
        WEBSOCKET_MESSAGES_TOTAL.labels(type="private").inc()
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send personal socket message to user {user_id}: {str(e)}")

    async def broadcast_to_role(self, message: dict, role: str):
        """
        Sends JSON payload to all connected sockets matching a specific role.
        """
        WEBSOCKET_MESSAGES_TOTAL.labels(type="role_broadcast").inc()
        if role in self.role_connections:
            for connection in self.role_connections[role]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to broadcast socket message to role {role}: {str(e)}")

    async def broadcast(self, message: dict):
        """
        Sends JSON payload to all connected sockets on the platform.
        """
        WEBSOCKET_MESSAGES_TOTAL.labels(type="global_broadcast").inc()
        for user_conns in self.active_connections.values():
            for connection in user_conns:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to broadcast socket message: {str(e)}")

websocket_manager = ConnectionManager()
