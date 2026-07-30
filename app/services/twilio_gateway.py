"""
The Twilio transport: one place that knows how to place a call or send a message.

Everything above this module deals in `DeliveryResult` and never imports
`twilio`, so swapping vendor or adding a second one is a change here and
nowhere else.

Two rules, both because of what this is used for:

* **Nothing raises.** Every failure — missing credentials, a bad number, an
  outage, a timeout — comes back as a `DeliveryResult` with `success=False` and
  a reason. A vendor being down must never be the reason an emergency fails to
  be *recorded*, and the caller needs the reason in order to decide whether
  retrying is worth it.
* **Permanent failures are named.** A malformed number will fail identically
  forever; retrying it four times wastes the window in which somebody could
  have been told the number is wrong. `retryable` distinguishes the two.

Credentials come from the environment through `settings` and are never logged.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

PERMANENT_TWILIO_CODES = {
    "21211",  # invalid 'To' number
    "21214",  # 'To' number is not a valid mobile number
    "21217",  # phone number does not appear to be valid
    "21219",  # 'To' number is not verified (trial accounts)
    "21408",  # permission to send to this region not enabled
    "21606",  # 'From' number is not a valid, SMS-capable number
    "21610",  # recipient has unsubscribed
    "21612",  # cannot route to this number
    "63003",  # WhatsApp channel could not find the recipient
    "63007",  # WhatsApp sender not found / not joined
}
"""
Errors that will fail identically on every retry.

Retrying a number Twilio has already said is unroutable does not eventually
work; it just delays the moment a human notices the number is wrong.
"""


@dataclass
class DeliveryResult:
    """What happened on one attempt, in vendor-neutral terms."""

    success: bool
    sid: Optional[str] = None
    provider_status: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = True
    skipped: bool = False
    """True when nothing was attempted — no credentials, or no number to try."""

    duration_seconds: Optional[int] = None
    extra: dict = field(default_factory=dict)


def _skipped(reason: str) -> DeliveryResult:
    return DeliveryResult(
        success=False, skipped=True, retryable=False,
        error_code="not_configured", error_message=reason,
    )


class TwilioGateway:
    """Voice, SMS and WhatsApp over Twilio."""

    # ── configuration ────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """
        Whether calls and SMS can be attempted at all.

        Read from settings per call, not captured at import, so credentials can
        be added to the environment and picked up on restart without a code
        change — the same deferred-activation property the Maps service has.
        """
        return bool(
            settings.EMERGENCY_COMMS_ENABLED
            and (settings.TWILIO_ACCOUNT_SID or "").strip()
            and (settings.TWILIO_AUTH_TOKEN or "").strip()
            and (settings.TWILIO_PHONE_NUMBER or "").strip()
        )

    def is_whatsapp_configured(self) -> bool:
        """
        WhatsApp needs its own approved sender, so it is configured separately.

        Absent means the WhatsApp attempt is recorded as `skipped`; the voice
        call and the SMS still go.
        """
        return bool(
            settings.EMERGENCY_COMMS_ENABLED
            and (settings.TWILIO_ACCOUNT_SID or "").strip()
            and (settings.TWILIO_AUTH_TOKEN or "").strip()
            and (settings.TWILIO_WHATSAPP_NUMBER or "").strip()
        )

    def _client(self):
        from twilio.rest import Client

        return Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

    @staticmethod
    def _whatsapp_address(number: str) -> str:
        number = (number or "").strip()
        return number if number.startswith("whatsapp:") else f"whatsapp:{number}"

    # ── sending ──────────────────────────────────────────────────────────

    async def _run(self, fn, *args, **kwargs) -> DeliveryResult:
        """
        Call the blocking Twilio SDK off the event loop.

        The SDK is synchronous. Called directly it would block the whole
        server for the duration of an HTTP round trip to Twilio, once per
        recipient, while other requests — including other emergencies — waited.
        """
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:  # nothing may escape into the caller
            logger.error("[TWILIO_UNEXPECTED] %s", exc)
            return DeliveryResult(
                success=False, error_code="unexpected",
                error_message=str(exc)[:400], retryable=True,
            )

    def _translate(self, exc: Exception) -> DeliveryResult:
        """Turn a Twilio exception into a result, preserving the vendor code."""
        code = getattr(exc, "code", None)
        message = getattr(exc, "msg", None) or str(exc)
        code_str = str(code) if code is not None else None
        return DeliveryResult(
            success=False,
            error_code=code_str,
            error_message=str(message)[:400],
            retryable=code_str not in PERMANENT_TWILIO_CODES,
        )

    async def place_call(self, to: str, say_text: str) -> DeliveryResult:
        """
        Ring a number and read a message when it is answered.

        TwiML is built inline rather than pointing at a hosted URL: the
        alternative needs a publicly reachable callback endpoint, which is
        infrastructure this phase does not have and which would fail closed the
        moment the app was unreachable — exactly when an emergency is running.
        """
        if not self.is_configured():
            return _skipped("Twilio voice is not configured.")
        if not (to or "").strip():
            return _skipped("No number on record for this recipient.")

        def _send() -> DeliveryResult:
            from twilio.base.exceptions import TwilioRestException
            from twilio.twiml.voice_response import VoiceResponse

            response = VoiceResponse()
            # Read twice: an emergency call is often answered mid-sentence.
            response.pause(length=1)
            response.say(say_text, voice="alice")
            response.pause(length=1)
            response.say(say_text, voice="alice")

            from app.services.twilio_callbacks import callback_url

            kwargs = {
                "to": to,
                "from_": settings.TWILIO_PHONE_NUMBER,
                "twiml": str(response),
            }
            # Only when there is a public URL to call back to. Pointing Twilio
            # at an unreachable host makes it retry and fills the log with
            # failures that say nothing about the call.
            url = callback_url("voice")
            if url:
                kwargs["status_callback"] = url
                kwargs["status_callback_event"] = [
                    "initiated", "ringing", "answered", "completed",
                ]
                kwargs["status_callback_method"] = "POST"

            try:
                call = self._client().calls.create(**kwargs)
            except TwilioRestException as exc:
                return self._translate(exc)
            except Exception as exc:
                return DeliveryResult(success=False, error_code="transport",
                                      error_message=str(exc)[:400], retryable=True)

            return DeliveryResult(
                success=True, sid=call.sid, provider_status=call.status,
            )

        return await self._run(_send)

    async def send_sms(self, to: str, body: str) -> DeliveryResult:
        if not self.is_configured():
            return _skipped("Twilio SMS is not configured.")
        if not (to or "").strip():
            return _skipped("No number on record for this recipient.")

        def _send() -> DeliveryResult:
            from twilio.base.exceptions import TwilioRestException

            from app.services.twilio_callbacks import callback_url

            kwargs = {
                "to": to, "from_": settings.TWILIO_PHONE_NUMBER, "body": body,
            }
            url = callback_url("sms")
            if url:
                kwargs["status_callback"] = url

            try:
                message = self._client().messages.create(**kwargs)
            except TwilioRestException as exc:
                return self._translate(exc)
            except Exception as exc:
                return DeliveryResult(success=False, error_code="transport",
                                      error_message=str(exc)[:400], retryable=True)

            return DeliveryResult(
                success=True, sid=message.sid, provider_status=message.status,
            )

        return await self._run(_send)

    async def send_whatsapp(
        self, to: str, body: str, media_urls: Optional[list[str]] = None
    ) -> DeliveryResult:
        """
        Send over WhatsApp.

        `media_urls` is accepted now and passed straight through, so attaching a
        map image or a photo later is a caller change rather than a change here.
        """
        if not self.is_whatsapp_configured():
            return _skipped("Twilio WhatsApp is not configured.")
        if not (to or "").strip():
            return _skipped("No number on record for this recipient.")

        def _send() -> DeliveryResult:
            from twilio.base.exceptions import TwilioRestException

            from app.services.twilio_callbacks import callback_url

            kwargs = {
                "to": self._whatsapp_address(to),
                "from_": self._whatsapp_address(settings.TWILIO_WHATSAPP_NUMBER),
                "body": body,
            }
            if media_urls:
                kwargs["media_url"] = media_urls
            url = callback_url("whatsapp")
            if url:
                kwargs["status_callback"] = url

            try:
                message = self._client().messages.create(**kwargs)
            except TwilioRestException as exc:
                return self._translate(exc)
            except Exception as exc:
                return DeliveryResult(success=False, error_code="transport",
                                      error_message=str(exc)[:400], retryable=True)

            return DeliveryResult(
                success=True, sid=message.sid, provider_status=message.status,
            )

        return await self._run(_send)


_gateway: TwilioGateway | None = None


def get_twilio_gateway() -> TwilioGateway:
    global _gateway
    if _gateway is None:
        _gateway = TwilioGateway()
    return _gateway


def set_twilio_gateway(gateway: TwilioGateway | None) -> None:
    """Install a different gateway — tests, and any future second vendor."""
    global _gateway
    _gateway = gateway
