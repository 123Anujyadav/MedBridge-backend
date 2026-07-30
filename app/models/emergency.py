import uuid
from datetime import datetime
from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

SOS_STATUSES = (
    "pending",
    "accepted",
    "doctor_assigned",
    "ambulance_dispatched",
    "hospital_reached",
    "resolved",
    "cancelled",
)
"""
The SOS lifecycle, in the order it normally runs.

`cancelled` is reachable from anywhere before a terminal state; the rest move
forward only.
"""

LEGACY_STATUSES = ("active", "dispatched", "arrived", "completed")
"""
The vocabulary this table used before the SOS workflow existed.

Still accepted by the constraint, because rows written under it are real
records of real events and a migration must not invalidate them. Nothing new
is ever written with these values.
"""

ACTIVE_SOS_STATUSES = (
    "pending",
    "accepted",
    "doctor_assigned",
    "ambulance_dispatched",
    "hospital_reached",
    "active",
    "dispatched",
    "arrived",
)
"""
An emergency that is still running.

Used to decide whether a patient already has one open — the check that stops a
second SOS being raised — and by the dashboards' "active emergencies" count, so
both agree on what "active" means.
"""

TERMINAL_SOS_STATUSES = ("resolved", "cancelled", "completed")

_ALL_STATUSES = SOS_STATUSES + LEGACY_STATUSES
_STATUS_SQL = ", ".join(f"'{s}'" for s in _ALL_STATUSES)


class EmergencyRequest(Base):
    """
    One emergency, from the moment a patient raises it to the moment it closes.

    Extended in Phase 2 rather than replaced. A second emergency table would
    have meant two answers to "is this patient in trouble right now" and two
    places for a dashboard to look; there is one table, and the pre-existing
    rows keep their meaning.

    This model records state. It sends nothing: dispatch, messaging and
    external integrations arrive in Phase 3 behind the notifier interface in
    `app.services.sos_notifications`.
    """
    __tablename__ = "emergency_requests"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_SQL})", name="emergency_status_check"),
        # A resolved emergency has to say when it resolved; a timeline with a
        # missing timestamp cannot be reconstructed afterwards.
        CheckConstraint(
            "status <> 'resolved' OR resolved_at IS NOT NULL",
            name="emergency_resolved_at_check",
        ),
        CheckConstraint(
            "status <> 'cancelled' OR cancelled_at IS NOT NULL",
            name="emergency_cancelled_at_check",
        ),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    patient_phone: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Real-time coordinates and location strings
    location: Mapped[dict] = mapped_column(
        JSON,

        default=lambda: {"lat": 0.0, "lng": 0.0, "address": ""},
        nullable=False
    )
    
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    hospital_name: Mapped[str] = mapped_column(String(150), nullable=True)
    ambulance_dispatched: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ambulance_id: Mapped[str] = mapped_column(String(50), nullable=True)
    
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    eta: Mapped[int] = mapped_column(Integer, nullable=True) # estimated minutes

    # ── Phase 2: SOS workflow ────────────────────────────────────────────

    assigned_doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.id", ondelete="SET NULL"), nullable=True, index=True
    )
    """
    The clinician responsible for this emergency.

    Indexed because it is the filter behind every doctor-facing query: a doctor
    may read the emergencies assigned to them and the unclaimed queue, and
    nothing else.
    """

    assigned_doctor_name: Mapped[str] = mapped_column(String(200), nullable=True)

    contact_name: Mapped[str] = mapped_column(String(120), nullable=True)
    contact_phone: Mapped[str] = mapped_column(String(20), nullable=True)
    contact_relationship: Mapped[str] = mapped_column(String(60), nullable=True)
    """
    The emergency contact, copied from the profile when the SOS was raised.

    A snapshot on purpose. Responders need the number that was current at the
    moment of the emergency, and a patient editing their profile afterwards
    must not rewrite the record of who was called for.
    """

    maps_url: Mapped[str] = mapped_column(String(255), nullable=True)

    created_by: Mapped[str] = mapped_column(String(50), nullable=True, default="patient")
    """Who raised it. `patient` today; Phase 3 may add automated triggers."""

    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    cancel_reason: Mapped[str] = mapped_column(String(255), nullable=True)

    # ── Phase 3: resolved location and nearest facility ──────────────────

    resolved_address: Mapped[str] = mapped_column(String(400), nullable=True)
    """
    The coordinates turned into a street address by reverse geocoding.

    Null until a Maps key is configured, and null forever if the lookup fails.
    The stored `location["address"]` — assembled from the patient's own
    registered address — remains the fallback, so a responder always has
    something to go on and never a guess presented as a geocoded fact.
    """

    hospital_distance_km: Mapped[float] = mapped_column(Float, nullable=True)
    hospital_latitude: Mapped[float] = mapped_column(Float, nullable=True)
    hospital_longitude: Mapped[float] = mapped_column(Float, nullable=True)
    """
    The nearest facility found, and how far away it is.

    All null unless a Maps key is present and the search succeeded. Nothing
    here is ever estimated: an invented ETA on an emergency screen is a number
    somebody will plan around.
    """

    # Relationships
    patient = relationship("Patient", back_populates="emergency_requests")
    communications = relationship(
        "CommunicationLog",
        back_populates="emergency",
        cascade="all, delete-orphan",
        order_by="CommunicationLog.created_at",
    )
    status_events = relationship(
        "EmergencyStatusEvent",
        back_populates="emergency",
        cascade="all, delete-orphan",
        order_by="EmergencyStatusEvent.created_at",
    )


class EmergencyStatusEvent(Base):
    """
    One entry in an emergency's timeline.

    Written for every status change, including the first. The current status
    lives on `emergency_requests` because that is what the guards read on every
    request; this table is the history behind it — who changed what, when, and
    why — which is what the patient's live status screen renders and what an
    audit of a clinical incident actually needs.
    """

    __tablename__ = "emergency_status_events"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_SQL})", name="emergency_event_status_check"),
    )

    emergency_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("emergency_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(String(50), nullable=False)
    note: Mapped[str] = mapped_column(String(500), nullable=True)

    actor_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    actor_role: Mapped[str] = mapped_column(String(50), nullable=True)
    actor_name: Mapped[str] = mapped_column(String(200), nullable=True)

    emergency = relationship("EmergencyRequest", back_populates="status_events")
