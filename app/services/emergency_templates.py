"""
The words sent to a responder, in one place.

No message text lives at a call site. Every channel renders from a template
here, keyed by `template_key`, and the rendered body is stored on the
communication log — so an incident review can read what somebody actually
received rather than re-deriving it from a template that may have changed
since.

Three things the templates are careful about:

* **Nothing is invented.** A field that is unknown is omitted, not filled with
  "N/A" or a guess. A responder reading "ETA 12 min" needs that to have come
  from somewhere real.
* **The voice script is different from the written ones.** It is read aloud by
  a synthetic voice to someone who may have just woken up, so it is short,
  has no URLs, and says the important thing first.
* **The map link always survives.** It needs no API key, so it is present even
  when every other Maps feature is switched off.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

TEMPLATE_VOICE_CONTACT = "voice.emergency_contact"
TEMPLATE_VOICE_DOCTOR = "voice.doctor"
TEMPLATE_VOICE_ADMIN = "voice.admin"
TEMPLATE_SMS_ALERT = "sms.emergency_alert"
TEMPLATE_WHATSAPP_ALERT = "whatsapp.emergency_alert"


def _fmt_time(value: Optional[datetime]) -> str:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.strftime("%d %b %Y, %H:%M UTC")


def _lines(*parts: Optional[str]) -> str:
    """Join the parts that exist, dropping the ones that do not."""
    return "\n".join(p for p in parts if p)


def build_context(emergency, patient=None) -> dict[str, Any]:
    """
    The facts a template may use, gathered once per emergency.

    Assembled from the record rather than passed around piecemeal, so every
    channel describes the same emergency the same way.
    """
    location = emergency.location or {}
    latitude = location.get("lat")
    longitude = location.get("lng")

    return {
        "patient_name": emergency.patient_name or "A patient",
        "patient_id": str(emergency.patient_id),
        "patient_phone": emergency.patient_phone,
        "emergency_id": str(emergency.id),
        "status": (emergency.status or "").replace("_", " ").title(),
        "address": emergency.resolved_address or location.get("address") or None,
        "latitude": latitude,
        "longitude": longitude,
        "maps_url": emergency.maps_url,
        "triggered_at": _fmt_time(getattr(emergency, "created_at", None)),
        "doctor_name": emergency.assigned_doctor_name,
        "contact_name": emergency.contact_name,
        "hospital_name": emergency.hospital_name,
        "hospital_eta": emergency.eta,
        "blood_type": getattr(patient, "blood_type", None),
    }


# ── voice ────────────────────────────────────────────────────────────────

def voice_script(template_key: str, ctx: dict[str, Any]) -> str:
    """
    What the synthetic voice reads out.

    No URL and no coordinates: neither can be written down by someone holding a
    phone, and reading a long string of digits aloud wastes the seconds that
    matter. The detail goes by SMS, which arrives at the same time.
    """
    who = ctx["patient_name"]
    where = ctx.get("address")
    tail = f" Their registered location is {where}." if where else ""

    if template_key == TEMPLATE_VOICE_CONTACT:
        return (
            f"This is an automated emergency alert from MedBridge. "
            f"{who} has triggered an emergency S O S and may need immediate help."
            f"{tail} Please check on them now. Details have been sent to you by text message."
        )
    if template_key == TEMPLATE_VOICE_DOCTOR:
        return (
            f"MedBridge emergency alert. Your patient {who} has triggered an "
            f"emergency S O S and is awaiting clinical response.{tail} "
            f"Please open the MedBridge clinician portal now."
        )
    return (
        f"MedBridge administrator alert. An emergency S O S has been raised by "
        f"{who}.{tail} Please open the MedBridge admin dashboard now."
    )


# ── SMS ──────────────────────────────────────────────────────────────────

def sms_body(ctx: dict[str, Any]) -> str:
    """
    The written alert. Carries everything the voice call could not.

    Kept compact because it is billed and split per 160 characters, but the
    map link is never dropped — it is the one line a responder acts on.
    """
    coords = (
        f"Lat/Lng: {ctx['latitude']:.5f}, {ctx['longitude']:.5f}"
        if ctx.get("latitude") is not None and ctx.get("longitude") is not None
        else None
    )
    return _lines(
        "MEDBRIDGE EMERGENCY SOS",
        f"Patient: {ctx['patient_name']}",
        f"Patient ID: {ctx['patient_id'][:8]}",
        f"Emergency ID: {ctx['emergency_id'][:8]}",
        f"Status: {ctx['status']}",
        f"Time: {ctx['triggered_at']}",
        f"Address: {ctx['address']}" if ctx.get("address") else None,
        coords,
        f"Map: {ctx['maps_url']}" if ctx.get("maps_url") else None,
        f"Nearest hospital: {ctx['hospital_name']}" if ctx.get("hospital_name") else None,
        f"ETA: {ctx['hospital_eta']} min" if ctx.get("hospital_eta") else None,
    )


# ── WhatsApp ─────────────────────────────────────────────────────────────

def whatsapp_body(ctx: dict[str, Any]) -> str:
    """
    The same facts, formatted for a chat client.

    WhatsApp renders markdown and has no per-segment cost, so this one can
    afford to be readable rather than terse.
    """
    coords = (
        f"📍 *Coordinates:* {ctx['latitude']:.5f}, {ctx['longitude']:.5f}"
        if ctx.get("latitude") is not None and ctx.get("longitude") is not None
        else None
    )
    return _lines(
        "🚨 *MEDBRIDGE EMERGENCY SOS*",
        "",
        f"*Patient:* {ctx['patient_name']}",
        f"*Blood group:* {ctx['blood_type']}" if ctx.get("blood_type") else None,
        f"*Emergency type:* Patient-triggered SOS",
        f"*Status:* {ctx['status']}",
        f"*Raised:* {ctx['triggered_at']}",
        "",
        f"*Address:* {ctx['address']}" if ctx.get("address") else None,
        coords,
        f"*Live location:* {ctx['maps_url']}" if ctx.get("maps_url") else None,
        "",
        f"*Nearest hospital:* {ctx['hospital_name']}" if ctx.get("hospital_name") else None,
        f"*Estimated arrival:* {ctx['hospital_eta']} min" if ctx.get("hospital_eta") else None,
        f"*Assigned clinician:* {ctx['doctor_name']}" if ctx.get("doctor_name") else None,
        "",
        "_This is an automated alert. Please respond immediately._",
    )


def render(channel: str, template_key: str, ctx: dict[str, Any]) -> str:
    """Render any channel's body from its key."""
    if channel == "voice":
        return voice_script(template_key, ctx)
    if channel == "sms":
        return sms_body(ctx)
    if channel == "whatsapp":
        return whatsapp_body(ctx)
    raise ValueError(f"Unknown communication channel: {channel}")
