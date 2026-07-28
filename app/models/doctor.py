import uuid
from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class Doctor(Base):
    """
    Doctor profile mapping user identity to specialties and hospital networks.
    """
    __tablename__ = "doctors"
    __table_args__ = (
        CheckConstraint("availability IN ('available', 'busy', 'offline', 'on_leave')", name="doctor_availability_check"),
        CheckConstraint("verification_status IN ('verified', 'pending', 'rejected', 'expired', 'under_review')", name="doctor_verification_status_check"),
        CheckConstraint("rating >= 0.0 AND rating <= 5.0", name="doctor_rating_check"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )

    doctor_code: Mapped[str] = mapped_column(
        String(8),
        unique=True,
        index=True,
        nullable=True,
        default=None,
    )
    """
    The clinician's 8-character Doctor ID — the third sign-in factor.

    Named `doctor_code` rather than `doctor_id` deliberately: `doctor_id` is
    already the UUID foreign key on appointments, cases and prescriptions, and
    two columns of that name meaning different things is the kind of collision
    that eventually routes a record to the wrong clinician. It is presented to
    people as "Doctor ID" everywhere in the UI and the API documentation.

    Null until an administrator approves the account. The workflow is
    signup → pending → approval → Doctor ID issued, so a clinician who has not
    been approved has no ID to be told, to leak, or to sign in with. It is
    allocated exactly once, by `admin_service.verify_doctor`, and never
    reissued afterwards — a clinician who has been given their ID keeps it
    through any later unverify/re-approve cycle.

    The database enforces the other half of that rule: a row may not be
    `verified` while this column is null (`doctor_verified_requires_code`).
    """

    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    specialty: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    sub_specialties: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    hospital_name: Mapped[str] = mapped_column(String(150), nullable=True)
    license_number: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    years_of_experience: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[float] = mapped_column(Float, default=5.0)
    
    total_patients: Mapped[int] = mapped_column(Integer, default=0)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    
    availability: Mapped[str] = mapped_column(String(50), default="available") # available, busy, offline, on_leave
    next_available: Mapped[str] = mapped_column(String(100), nullable=True)
    consultation_fee: Mapped[float] = mapped_column(Float, default=0.0)
    
    education: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    certifications: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    languages: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    avatar_url: Mapped[str] = mapped_column(String(255), nullable=True)
    
    verification_status: Mapped[str] = mapped_column(String(50), default="pending") # verified, pending, rejected, under_review
    verified_date: Mapped[str] = mapped_column(String(100), nullable=True)
    bio: Mapped[str] = mapped_column(String(1000), nullable=True)

    # Relationships
    user = relationship("User", back_populates="doctor")
    hospital = relationship("Hospital", back_populates="doctors")
    cases = relationship("Case", back_populates="doctor")
    appointments = relationship("Appointment", back_populates="doctor")
    prescriptions = relationship("Prescription", back_populates="doctor")
