import uuid
from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class Appointment(Base):
    """
    Physician Consultation Appointment model.
    """
    __tablename__ = "appointments"
    __table_args__ = (
        Index("idx_appointment_slot", "doctor_id", "date", "time"),
        # One live booking per clinician slot, enforced by the database.
        #
        # The service checks for a conflict before inserting, but check-then-
        # insert cannot hold under concurrency: two requests can both read "free"
        # before either writes, and both then book the same slot. That is not
        # theoretical — duplicate rows for one doctor/date/time were observed
        # during verification. Cancelled, completed and no-show appointments are
        # excluded so a released slot can be rebooked.
        Index(
            "uq_appointment_active_slot",
            "doctor_id", "date", "time",
            unique=True,
            postgresql_where=text(
                "deleted_at IS NULL AND status IN "
                "('scheduled', 'confirmed', 'in_progress')"
            ),
            sqlite_where=text(
                "deleted_at IS NULL AND status IN "
                "('scheduled', 'confirmed', 'in_progress')"
            ),
        ),
        CheckConstraint("status IN ('scheduled', 'confirmed', 'in_progress', 'completed', 'cancelled', 'no_show')", name="appointment_status_check"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    patient_avatar_url: Mapped[str] = mapped_column(String(255), nullable=True)
    
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    doctor_name: Mapped[str] = mapped_column(String(200), nullable=False)
    doctor_avatar_url: Mapped[str] = mapped_column(String(255), nullable=True)
    
    specialty: Mapped[str] = mapped_column(String(100), nullable=False)
    hospital_name: Mapped[str] = mapped_column(String(150), nullable=False)
    
    date: Mapped[str] = mapped_column(String(50), nullable=False)  # ISO Date String YYYY-MM-DD
    time: Mapped[str] = mapped_column(String(50), nullable=False)  # HH:MM format
    duration: Mapped[int] = mapped_column(Integer, default=30)  # minutes
    
    type: Mapped[str] = mapped_column(String(50), default="in_person") # in_person, video, phone, ai_triage
    status: Mapped[str] = mapped_column(String(50), default="scheduled") # scheduled, confirmed, in_progress, completed, cancelled, no_show
    
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    
    room_number: Mapped[str] = mapped_column(String(50), nullable=True)
    video_call_link: Mapped[str] = mapped_column(String(255), nullable=True)
    
    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Relationships
    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")
    case = relationship("Case", back_populates="appointments")

