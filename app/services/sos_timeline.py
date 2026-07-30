"""
The whole history of an emergency, on one axis.

Status changes live in `emergency_status_events` and communication attempts in
`communication_logs`, because they are different things: one is clinical state
that authorisation depends on, the other is delivery. They are *read* together,
though — the question a patient and an incident review both ask is "what
happened, in what order", and that answer spans both tables plus a couple of
facts held on the emergency row itself.

Merging on read rather than writing to a single table keeps each table's
constraints meaningful and means neither has to learn the other's vocabulary.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.communication import CommunicationLog
from app.models.emergency import EmergencyRequest, EmergencyStatusEvent
from app.services.emergency_comms import mask_phone
from app.services.google_maps import get_google_maps_service

logger = logging.getLogger(__name__)

STATUS_LABELS = {
    "pending": "Emergency created",
    "accepted": "Emergency accepted",
    "doctor_assigned": "Doctor assigned",
    "ambulance_dispatched": "Ambulance dispatched",
    "hospital_reached": "Hospital reached",
    "resolved": "Resolved",
    "cancelled": "Cancelled",
}

CHANNEL_LABELS = {"voice": "Call", "sms": "SMS", "whatsapp": "WhatsApp"}

ROLE_LABELS = {
    "emergency_contact": "emergency contact",
    "doctor": "doctor",
    "admin": "admin",
}

COMMS_STATUS_VERBS = {
    "queued": "queued for",
    "sending": "in progress to",
    "sent": "sent to",
    "delivered": "delivered to",
    "failed": "failed to",
    "skipped": "skipped for",
}


class SOSTimelineService:
    async def build(
        self, db: AsyncSession, emergency_id: uuid.UUID,
        emergency_payload: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Every event for one emergency, oldest first.

        Each entry carries a machine-readable `kind` and `key` alongside its
        label, so the UI can style them without parsing prose.
        """
        emergency = await db.get(EmergencyRequest, emergency_id)
        if emergency is None:
            return {"emergency_id": str(emergency_id), "entries": []}

        entries: list[dict[str, Any]] = []

        status_events = (await db.execute(
            select(EmergencyStatusEvent)
            .where(EmergencyStatusEvent.emergency_id == emergency_id)
            .order_by(EmergencyStatusEvent.created_at)
        )).scalars().all()

        for event in status_events:
            entries.append({
                "kind": "status",
                "key": event.status,
                "label": STATUS_LABELS.get(event.status, event.status),
                "detail": event.note,
                "actor_name": event.actor_name,
                "actor_role": event.actor_role,
                "at": event.created_at,
            })

        from app.services.emergency_comms import _CHANNEL_RANK, _RECIPIENT_RANK

        comms = (await db.execute(
            select(CommunicationLog)
            .where(CommunicationLog.emergency_id == emergency_id)
            .order_by(_RECIPIENT_RANK, _CHANNEL_RANK, CommunicationLog.id)
        )).scalars().all()

        for row in comms:
            channel = CHANNEL_LABELS.get(row.channel, row.channel)
            who = ROLE_LABELS.get(row.recipient_role, row.recipient_role)
            verb = COMMS_STATUS_VERBS.get(row.status, row.status)

            # A skipped attempt says *why* — "no number on record" and "WhatsApp
            # not configured" need different action from whoever reads this.
            detail = row.error_message if row.status in ("failed", "skipped") else None

            entries.append({
                "kind": "communication",
                "key": f"{row.channel}.{row.recipient_role}.{row.status}",
                "label": f"{channel} {verb} {who}",
                "detail": detail,
                "actor_name": row.recipient_name,
                "actor_role": row.recipient_role,
                "channel": row.channel,
                "status": row.status,
                "attempts": row.attempts,
                "at": row.completed_at or row.sent_at or row.created_at,
            })

        if emergency.hospital_name:
            entries.append({
                "kind": "hospital",
                "key": "hospital_found",
                "label": f"Nearest hospital identified: {emergency.hospital_name}",
                "detail": (
                    f"{emergency.hospital_distance_km} km away"
                    if emergency.hospital_distance_km is not None else None
                ),
                "at": emergency.updated_at,
            })

        entries.sort(key=lambda e: (e["at"] is None, e["at"]))
        return {
            "emergency_id": str(emergency_id),
            "patient_id": str(emergency.patient_id),
            "status": emergency.status,
            "entries": entries,
        }

    async def hospital_summary(
        self, db: AsyncSession, emergency_id: uuid.UUID
    ) -> dict[str, Any]:
        """
        The nearest facility, or an honest statement that none is known.

        `available` is false whenever there is nothing real to report, and
        `reason` says which of the two causes it was — no key configured, or a
        search that found nothing. Neither case produces a placeholder
        hospital: an invented address on an emergency screen is somewhere an
        ambulance gets sent.
        """
        emergency = await db.get(EmergencyRequest, emergency_id)
        if emergency is None:
            return {"available": False, "reason": "Emergency not found."}

        if emergency.hospital_name:
            maps = get_google_maps_service()
            location = emergency.location or {}
            directions_url = None
            if (
                location.get("lat") is not None
                and emergency.hospital_latitude is not None
            ):
                directions_url = maps.build_directions_url(
                    (location["lat"], location["lng"]),
                    (emergency.hospital_latitude, emergency.hospital_longitude),
                )
            return {
                "available": True,
                "hospital_id": emergency.hospital_id,
                "hospital_name": emergency.hospital_name,
                "latitude": emergency.hospital_latitude,
                "longitude": emergency.hospital_longitude,
                "distance_km": emergency.hospital_distance_km,
                "eta_minutes": emergency.eta,
                "directions_url": directions_url,
            }

        if not get_google_maps_service().is_enabled():
            return {
                "available": False,
                "reason": (
                    "Hospital search is not enabled on this deployment. "
                    "Coordinates and the map link are still available."
                ),
            }
        return {
            "available": False,
            "reason": "No nearby hospital has been identified for this emergency.",
        }


sos_timeline_service = SOSTimelineService()
