"""
Where an emergency gets announced — and the seam Phase 3 plugs into.

Phase 2 delivers one channel: the live WebSocket already carrying the rest of
the platform's realtime traffic. Phase 3 adds SMS, WhatsApp, voice and push.
The point of this module is that adding them changes *this file and nothing
else* — the SOS service calls an interface, never a transport.

Two rules the implementations must keep, because an emergency is exactly where
they matter:

* **Announce after the commit, never before.** A responder told about an
  emergency that then rolled back has been sent to a patient who did not raise
  one.
* **A delivery failure must not fail the emergency.** If a channel is down, the
  record still exists and the other channels still fire. `notify` therefore
  never raises; it logs and returns.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class EmergencyNotifier(Protocol):
    """
    A channel an emergency can be announced on.

    Phase 3 implements this for SMS, WhatsApp, voice and push. Each new channel
    is a class here plus one line in `get_emergency_notifier`; no caller
    changes, because callers only ever hold this Protocol.
    """

    async def emergency_raised(self, payload: dict) -> None:
        """A patient has raised an emergency."""
        ...

    async def emergency_updated(self, payload: dict) -> None:
        """Its status changed — assigned, dispatched, resolved, cancelled."""
        ...

    async def communications_updated(self, payload: dict) -> None:
        """A call or message about it was queued, sent, retried or failed."""
        ...


class WebSocketEmergencyNotifier:
    """
    Delivery over the platform's existing live socket.

    Audience is decided per message rather than broadcast to everyone:

    * the **patient** gets their own emergency, and only theirs, over the
      user-scoped channel;
    * **administrators** get every emergency, as a role broadcast;
    * **doctors** get a role broadcast too, because an unclaimed emergency has
      to reach a clinician who can accept it. The socket is only ever opened by
      an approved, active clinician — the WebSocket endpoint re-checks that on
      connect — and what a doctor may then *fetch* is narrowed again in
      `sos_service`.

    Nothing here uses `broadcast()`. A global broadcast would put a patient's
    name, blood group and home address on every open socket in the platform,
    including other patients'.
    """

    async def emergency_raised(self, payload: dict) -> None:
        await self._fan_out("EMERGENCY_SOS_CREATED", payload)

    async def emergency_updated(self, payload: dict) -> None:
        await self._fan_out("EMERGENCY_SOS_UPDATED", payload)

    async def communications_updated(self, payload: dict) -> None:
        await self._fan_out("EMERGENCY_COMMS_UPDATED", payload)

    async def _fan_out(self, event_type: str, payload: dict) -> None:
        from fastapi.encoders import jsonable_encoder

        from app.core.websocket import websocket_manager

        # Coerced to JSON primitives before it goes near a socket. The service
        # builds this payload from ORM values, so it carries `UUID` and
        # `datetime` objects that `send_json` cannot encode — and because a
        # delivery failure is deliberately swallowed below, the result was a
        # silent one: every emergency was recorded correctly and announced to
        # nobody.
        message = jsonable_encoder({"type": event_type, **payload})
        patient_id = str(payload.get("patient_id") or "")

        try:
            if patient_id:
                await websocket_manager.send_personal_message(message, patient_id)
            await websocket_manager.broadcast_to_role(message, "admin")
            await websocket_manager.broadcast_to_role(message, "doctor")
        except Exception as exc:
            # An emergency that is recorded but not announced is recoverable;
            # one that fails to record because a socket was closed is not.
            logger.error("[SOS_NOTIFY_FAILED] %s: %s", event_type, exc)


class NullEmergencyNotifier:
    """Announces nothing. Used by tests that assert on the record, not delivery."""

    async def emergency_raised(self, payload: dict) -> None:
        return None

    async def emergency_updated(self, payload: dict) -> None:
        return None

    async def communications_updated(self, payload: dict) -> None:
        return None


_notifier: EmergencyNotifier = WebSocketEmergencyNotifier()


def get_emergency_notifier() -> EmergencyNotifier:
    """
    The active notifier.

    Phase 3 replaces this with a composite that fans out to SMS, WhatsApp and
    voice alongside the socket. Every call site already goes through here, so
    that is the only edit required.
    """
    return _notifier


def set_emergency_notifier(notifier: EmergencyNotifier) -> None:
    """Install a different notifier — tests, and the Phase 3 rollout."""
    global _notifier
    _notifier = notifier
