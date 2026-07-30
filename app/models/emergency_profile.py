import uuid
from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from app.db.base_class import Base


class EmergencyProfile(Base):
    """
    The standing emergency record for one patient: who to call, where they live,
    and where they last were.

    Kept in its own table rather than in `patients.emergency_contact`, which is a
    three-key JSON blob. A blob cannot be indexed, cannot be constrained, and
    cannot say whether a field was never filled in or was filled in with an
    empty string — and this is the record something will one day read while
    somebody is unconscious. Every field here is a real column with a real
    constraint.

    One row per patient: the primary key *is* the patient's id, the same shape
    `patients` and `doctors` already use against `users`. That makes a second
    profile for the same patient unrepresentable rather than merely discouraged.

    Nothing in this module notifies anybody. It is the data foundation the SOS
    system is built on later; dispatch, messaging and alerting are not here.
    """

    __tablename__ = "emergency_profiles"
    __table_args__ = (
        CheckConstraint(
            "latitude IS NULL OR (latitude >= -90 AND latitude <= 90)",
            name="emergency_profile_latitude_check",
        ),
        CheckConstraint(
            "longitude IS NULL OR (longitude >= -180 AND longitude <= 180)",
            name="emergency_profile_longitude_check",
        ),
        # Coordinates are meaningless individually — a latitude with no
        # longitude points nowhere. They are written and cleared as a pair.
        CheckConstraint(
            "(latitude IS NULL) = (longitude IS NULL)",
            name="emergency_profile_coordinate_pair_check",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # ── Emergency contact ────────────────────────────────────────────────
    contact_name: Mapped[str] = mapped_column(String(120), nullable=False)
    contact_phone: Mapped[str] = mapped_column(String(20), nullable=False)
    contact_relationship: Mapped[str] = mapped_column(String(60), nullable=False)

    alternate_phone: Mapped[str] = mapped_column(String(20), nullable=True)
    """A second number to try. Optional, and never equal to `contact_phone`."""

    # ── Registered address ───────────────────────────────────────────────
    house_number: Mapped[str] = mapped_column(String(60), nullable=False)
    street: Mapped[str] = mapped_column(String(150), nullable=False)
    landmark: Mapped[str] = mapped_column(String(150), nullable=True)
    locality: Mapped[str] = mapped_column(String(120), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    district: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False, default="India")
    pincode: Mapped[str] = mapped_column(String(12), nullable=False)

    # ── Last known coordinates ───────────────────────────────────────────
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=True)

    maps_url: Mapped[str] = mapped_column(String(255), nullable=True)
    """
    A Google Maps link for the stored coordinates.

    Derived on the server from the latitude and longitude rather than accepted
    from the browser. A URL a client can choose is a URL a client can point
    anywhere, and this one is rendered as a link for someone to follow in an
    emergency.
    """

    location_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """
    When the coordinates were captured — not when the row was last written.

    A position is only as useful as it is recent, so the reader needs to know
    the age of the fix independently of any later edit to the contact or
    address.
    """

    # Relationships
    patient = relationship("Patient", back_populates="emergency_profile")
