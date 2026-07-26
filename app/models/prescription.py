import uuid
from datetime import datetime
from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime
from app.db.base_class import Base

class Prescription(Base):
    """
    Medical Prescription (Rx) model.
    """
    __tablename__ = "prescriptions"
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'ai_parsing', 'parsed', 'verified', 'active', 'completed', 'cancelled')", name="prescription_status_check"),
    )

    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    doctor_name: Mapped[str] = mapped_column(String(200), nullable=False)
    diagnosis: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    
    status: Mapped[str] = mapped_column(String(50), default="draft") # draft, ai_parsing, parsed, verified, active, completed, cancelled
    ai_parsed: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_parse_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    follow_up_date: Mapped[str] = mapped_column(String(100), nullable=True)
    attachment_url: Mapped[str] = mapped_column(String(255), nullable=True)

    # Relationships
    case = relationship("Case", back_populates="prescriptions")
    patient = relationship("Patient", back_populates="prescriptions")
    doctor = relationship("Doctor", back_populates="prescriptions")
    medications = relationship("Medication", back_populates="prescription", cascade="all, delete-orphan")


class Medication(Base):
    """
    Medication line items attached to a Prescription.
    """
    __tablename__ = "medications"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'taken', 'missed', 'snoozed', 'active')", name="medication_status_check"),
    )

    prescription_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prescriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    generic_name: Mapped[str] = mapped_column(String(150), nullable=True)
    dosage: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "500mg"
    frequency: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "once daily"
    duration: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "7 days"
    special_instructions: Mapped[str] = mapped_column(Text, default="", nullable=False)
    
    status: Mapped[str] = mapped_column(String(50), default="active") # pending, taken, missed, snoozed, active
    scheduled_times: Mapped[list] = mapped_column(JSON, default=list, nullable=False) # ["08:00", "20:00"]
    
    taken_doses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_doses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    start_date: Mapped[str] = mapped_column(String(100), nullable=False)
    end_date: Mapped[str] = mapped_column(String(100), nullable=False)
    
    side_effects: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    interactions: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # Relationships
    prescription = relationship("Prescription", back_populates="medications")


