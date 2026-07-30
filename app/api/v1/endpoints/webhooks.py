"""
Provider callbacks.

Public by necessity — Twilio cannot present a bearer token — so authenticity
comes from the request signature instead. An unsigned or wrongly-signed request
is refused with 403 before anything is read from it.

Answers 204 for anything it accepts *or* recognises as not ours. Twilio retries
a non-2xx, so returning an error for a SID this platform never sent would earn a
stream of retries about a message that does not exist here.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.services.twilio_callbacks import twilio_callback_service

logger = logging.getLogger(__name__)

router = APIRouter()


async def _handle(channel: str, request: Request, db: AsyncSession) -> Response:
    form = await request.form()
    params = {key: str(value) for key, value in form.items()}

    # Validated against the URL Twilio actually called. Behind a proxy the
    # scheme must be the forwarded one, which is why `--proxy-headers` matters
    # here as well: signing over `http` when Twilio called `https` fails.
    signature = request.headers.get("X-Twilio-Signature")
    if not twilio_callback_service.verify_signature(
        str(request.url), params, signature
    ):
        logger.warning("[TWILIO_CALLBACK_REJECTED] bad or missing signature")
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    row = await twilio_callback_service.record(db, channel, params)
    if row is not None:
        await db.commit()

        # Tell the open dashboards. Failure to announce must not fail the
        # callback, or Twilio will retry a status already recorded.
        try:
            from app.services.emergency_comms import emergency_comms_service

            await emergency_comms_service.announce(db, row.emergency_id)
        except Exception as exc:
            logger.warning("[TWILIO_CALLBACK_ANNOUNCE_FAILED] %s", exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/twilio/sms-status", include_in_schema=False)
async def twilio_sms_status(
    request: Request, db: AsyncSession = Depends(get_db)
) -> Any:
    """Delivery lifecycle for an SMS."""
    return await _handle("sms", request, db)


@router.post("/twilio/whatsapp-status", include_in_schema=False)
async def twilio_whatsapp_status(
    request: Request, db: AsyncSession = Depends(get_db)
) -> Any:
    """Delivery lifecycle for a WhatsApp message."""
    return await _handle("whatsapp", request, db)


@router.post("/twilio/voice-status", include_in_schema=False)
async def twilio_voice_status(
    request: Request, db: AsyncSession = Depends(get_db)
) -> Any:
    """Lifecycle for a voice call, including its duration."""
    return await _handle("voice", request, db)
