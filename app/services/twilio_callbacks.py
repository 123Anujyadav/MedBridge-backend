"""
What Twilio tells us after it has taken a message or a call.

Creating a message returns `queued` or `accepted` — an acknowledgement, not a
delivery. Everything after that arrives asynchronously as a status callback, and
this module is where those land: the vendor's vocabulary is translated into
ours, the row is advanced, the callback itself is kept, and the dashboards are
told.

Three rules:

* **Nothing is synthesised.** A status exists in the record only because the
  provider reported it. There is no inference from elapsed time, and no
  optimistic promotion of `accepted` to `delivered`.
* **The lifecycle only moves forward.** Callbacks can arrive out of order — a
  `sent` after a `delivered` is common enough — and a later-but-earlier-stage
  callback must not walk the status backwards.
* **Every callback is persisted**, in `provider_events`, even one that changes
  nothing. That list is the evidence of how an attempt reached its outcome.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.communication import CommunicationLog

logger = logging.getLogger(__name__)

MESSAGE_STATUS_MAP = {
    "queued": "queued",
    "accepted": "accepted",
    "scheduled": "accepted",
    "sending": "sending",
    "sent": "sent",
    "receiving": "sent",
    "received": "delivered",
    "delivered": "delivered",
    "read": "delivered",          # WhatsApp read receipt — still delivered
    "undelivered": "undelivered",
    "failed": "failed",
    "canceled": "canceled",
}

CALL_STATUS_MAP = {
    "queued": "queued",
    "initiated": "accepted",
    "ringing": "sent",            # the handset is ringing: it left the network
    "in-progress": "sent",
    "answered": "sent",
    "completed": "delivered",     # the call connected and finished
    "busy": "undelivered",
    "no-answer": "undelivered",
    "canceled": "canceled",
    "failed": "failed",
}

STATUS_RANK = {
    "queued": 0,
    "sending": 1,
    "accepted": 2,
    "sent": 3,
    "delivered": 4,
    "undelivered": 4,
    "canceled": 4,
    "failed": 4,
    "skipped": 4,
}
"""
How far through the lifecycle each status is.

Used to refuse a backwards move. Twilio does not guarantee callback ordering,
and a `sent` arriving after a `delivered` would otherwise un-deliver a message
that had already reached the handset.
"""


def callback_url(channel: str) -> Optional[str]:
    """
    Where Twilio should report to, or None if it cannot reach us.

    Returning None is what keeps the platform working without a public
    hostname: the message is still sent, and delivery tracking honestly stops
    at the provider's acknowledgement rather than pointing Twilio at a URL it
    will fail to call and retry.
    """
    base = (settings.PUBLIC_BASE_URL or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}{settings.API_V1_STR}/webhooks/twilio/{channel}-status"


def translate(channel: str, provider_status: str) -> Optional[str]:
    """Map a provider status onto ours, or None if we do not recognise it."""
    table = CALL_STATUS_MAP if channel == "voice" else MESSAGE_STATUS_MAP
    return table.get((provider_status or "").strip().lower())


class TwilioCallbackService:
    def verify_signature(
        self, url: str, params: dict[str, Any], signature: Optional[str]
    ) -> bool:
        """
        Whether this really came from Twilio.

        The endpoint is public — Twilio cannot present a bearer token — so the
        signature is the only thing standing between the callback handler and
        anyone who can guess a message SID. Without a configured auth token
        there is nothing to verify against, and the request is refused rather
        than trusted.
        """
        token = (settings.TWILIO_AUTH_TOKEN or "").strip()
        if not token or not signature:
            return False
        try:
            from twilio.request_validator import RequestValidator

            return RequestValidator(token).validate(url, params, signature)
        except Exception as exc:
            logger.warning("[TWILIO_CALLBACK_VALIDATION_ERROR] %s", exc)
            return False

    async def record(
        self, db: AsyncSession, channel: str, params: dict[str, Any]
    ) -> Optional[CommunicationLog]:
        """
        Apply one callback to the attempt it belongs to.

        Returns the row when something was recorded, or None when the callback
        refers to a SID this platform did not send — which is not an error worth
        failing the request over, since Twilio would only retry it.
        """
        sid = (params.get("MessageSid") or params.get("CallSid") or "").strip()
        provider_status = (
            params.get("MessageStatus") or params.get("CallStatus") or ""
        ).strip()
        if not sid or not provider_status:
            return None

        row = (await db.execute(
            select(CommunicationLog).where(CommunicationLog.provider_sid == sid)
        )).scalars().first()
        if row is None:
            logger.info("[TWILIO_CALLBACK_UNKNOWN_SID] %s", sid[:12])
            return None

        mapped = translate(channel, provider_status)
        now = datetime.now(timezone.utc)

        # Persisted whether or not it advances anything: the sequence is the
        # evidence, and a callback that changed nothing is still a fact.
        events = list(row.provider_events or [])
        events.append({
            "at": now.isoformat(),
            "provider_status": provider_status,
            "mapped_status": mapped,
            "error_code": params.get("ErrorCode"),
            "duration": params.get("CallDuration"),
        })
        row.provider_events = events
        row.provider_status = provider_status

        if params.get("ErrorCode"):
            row.error_code = str(params["ErrorCode"])[:40]
        if params.get("ErrorMessage"):
            row.error_message = str(params["ErrorMessage"])[:500]

        duration = params.get("CallDuration")
        if duration:
            try:
                row.duration_seconds = int(duration)
            except (TypeError, ValueError):
                pass

        if mapped is None:
            logger.info("[TWILIO_CALLBACK_UNMAPPED] %s -> %s", channel,
                        provider_status)
            await db.flush()
            return row

        # Forward only. An out-of-order callback must not walk it backwards.
        if STATUS_RANK.get(mapped, 0) >= STATUS_RANK.get(row.status, 0):
            row.status = mapped
            if mapped == "sent" and row.sent_at is None:
                row.sent_at = now
            if mapped in ("delivered", "undelivered", "canceled", "failed"):
                row.completed_at = now
                # A terminal provider outcome ends the attempt: there is nothing
                # for the retry sweep to pick up.
                row.next_attempt_at = None

        await db.flush()
        logger.info(
            "[TWILIO_CALLBACK] sid=%s %s -> %s", sid[:12], provider_status,
            row.status,
        )
        return row


twilio_callback_service = TwilioCallbackService()
