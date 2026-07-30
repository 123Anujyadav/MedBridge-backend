"""
Request and response shapes for the SOS emergency workflow.

The response deliberately carries everything a responder needs on one screen —
patient identity, blood group, the emergency contact, the address, the map link
and the timeline — because the alternative is a dashboard that fires five
requests while somebody is waiting for an ambulance.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.emergency import SOS_STATUSES


class SOSTriggerRequest(BaseModel):
    """
    What the browser sends when the countdown finishes.

    Coordinates are optional. The patient may have refused the location prompt,
    in which case the position stored on their emergency profile is used
    instead — refusing to share a live position is not a reason to refuse them
    an ambulance.
    """

    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    @field_validator("latitude", "longitude")
    @classmethod
    def _reject_nan(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v != v:  # NaN is the only value unequal to itself
            raise ValueError("Coordinates must be real numbers.")
        return v


class SOSStatusUpdateRequest(BaseModel):
    """A responder moving the emergency to its next state."""

    status: str
    note: Optional[str] = Field(default=None, max_length=500)

    @field_validator("status")
    @classmethod
    def _known_status(cls, v: str) -> str:
        value = (v or "").strip().lower()
        if value not in SOS_STATUSES:
            raise ValueError(
                "Unknown emergency status. Expected one of: "
                + ", ".join(SOS_STATUSES)
            )
        return value


class SOSCancelRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=255)


class SOSAssignDoctorRequest(BaseModel):
    doctor_id: uuid.UUID


class SOSTimelineEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: str
    note: Optional[str] = None
    actor_role: Optional[str] = None
    actor_name: Optional[str] = None
    created_at: Optional[datetime] = None


class SOSEmergencyResponse(BaseModel):
    """
    One emergency, as every portal sees it.

    The same shape is returned to the patient, the assigned clinician and an
    administrator. What differs between them is *which* emergencies they can
    reach, which is decided in the service layer — not which fields they get,
    because a responder who is entitled to the emergency at all needs all of
    it.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    status: str

    # ── who ──────────────────────────────────────────────────────────────
    patient_name: str
    patient_phone: Optional[str] = None
    patient_age: Optional[int] = None
    blood_type: Optional[str] = None

    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_relationship: Optional[str] = None

    assigned_doctor_id: Optional[uuid.UUID] = None
    assigned_doctor_name: Optional[str] = None

    # ── where ────────────────────────────────────────────────────────────
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    maps_url: Optional[str] = None

    # ── when ─────────────────────────────────────────────────────────────
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancel_reason: Optional[str] = None

    created_by: Optional[str] = None
    is_active: bool = True

    timeline: List[SOSTimelineEntry] = Field(default_factory=list)


class SOSActiveResponse(BaseModel):
    """
    Whether this patient currently has an emergency open.

    A dedicated shape so the SOS button can answer "is one already running?"
    in a single request, without the page having to fetch a list and reason
    about it.
    """

    active: bool
    emergency: Optional[SOSEmergencyResponse] = None


# ── Phase 3: communication layer ─────────────────────────────────────────

class SOSCommunicationEntry(BaseModel):
    """
    One attempt to reach one person on one channel.

    The telephone number is masked. The full value stays in the row for an
    incident review; the API does not hand a third party's number to every
    screen that lists an emergency.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    channel: str
    recipient_role: str
    recipient_name: Optional[str] = None
    recipient_phone_masked: Optional[str] = None

    status: str
    provider: Optional[str] = None
    provider_sid: Optional[str] = None
    provider_status: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    attempts: int = 0
    next_attempt_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    created_at: Optional[datetime] = None


class SOSCommunicationsResponse(BaseModel):
    emergency_id: str
    patient_id: Optional[str] = None
    communications: List[SOSCommunicationEntry] = Field(default_factory=list)


class SOSTimelineItem(BaseModel):
    """
    One event, whatever kind it is.

    `kind` and `key` are machine-readable so the UI can style an entry without
    parsing its label.
    """

    kind: str
    key: str
    label: str
    detail: Optional[str] = None
    actor_name: Optional[str] = None
    actor_role: Optional[str] = None
    channel: Optional[str] = None
    status: Optional[str] = None
    attempts: Optional[int] = None
    at: Optional[datetime] = None


class SOSTimelineResponse(BaseModel):
    emergency_id: str
    patient_id: Optional[str] = None
    status: Optional[str] = None
    entries: List[SOSTimelineItem] = Field(default_factory=list)


class SOSHospitalResponse(BaseModel):
    """
    The nearest facility, or an honest statement that none is known.

    `available` is false whenever there is nothing real to report, and `reason`
    says why. No field is ever populated with an estimate.
    """

    available: bool
    reason: Optional[str] = None

    hospital_id: Optional[uuid.UUID] = None
    hospital_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_km: Optional[float] = None
    eta_minutes: Optional[int] = None
    directions_url: Optional[str] = None
