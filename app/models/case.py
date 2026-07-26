import uuid
from datetime import datetime
from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime
from app.db.base_class import Base

class Case(Base):
    """
    Medical Consultation Case (Ticket) model.
    Tracks a patient case from intake, through AI triage, routing, and completion.
    """
    __tablename__ = "cases"
    __table_args__ = (
        Index("idx_case_status_urgency", "status", "urgency_level"),
        CheckConstraint("urgency_level IN ('low', 'medium', 'high', 'critical')", name="case_urgency_level_check"),
        CheckConstraint("status IN ('intake', 'ai_processing', 'routed', 'in_consultation', 'prescribed', 'report_generated', 'completed', 'archived')", name="case_status_check"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    patient_avatar_url: Mapped[str] = mapped_column(String(255), nullable=True)
    patient_age: Mapped[int] = mapped_column(Integer, nullable=False)
    patient_gender: Mapped[str] = mapped_column(String(50), nullable=False)
    
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("doctors.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    doctor_name: Mapped[str] = mapped_column(String(200), nullable=True)
    specialty: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    symptom_summary: Mapped[str] = mapped_column(Text, nullable=False)
    urgency_level: Mapped[str] = mapped_column(String(50), default="low", index=True) # low, medium, high, critical
    status: Mapped[str] = mapped_column(String(50), default="intake", index=True) # intake, ai_processing, routed, in_consultation, prescribed, etc.
    
    # AI generated triage analysis
    ai_extracted_symptoms: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    ai_specialty_recommendation: Mapped[str] = mapped_column(String(100), nullable=True)
    ai_confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Files/Documents attachments
    attachments: Mapped[list] = mapped_column(JSON, default=list, nullable=False) # list of {name, type, url}

    
    patient_history: Mapped[str] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    patient = relationship("Patient", back_populates="cases")
    doctor = relationship("Doctor", back_populates="cases")
    symptoms = relationship("Symptom", back_populates="case", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="case", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="case", cascade="all, delete-orphan")


class Symptom(Base):
    """
    Structured symptoms attached to a Consultation Case.
    """
    __tablename__ = "symptoms"
    __table_args__ = (
        CheckConstraint("severity IN ('mild', 'moderate', 'severe')", name="symptom_severity_check"),
    )

    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(50), nullable=False) # mild, moderate, severe
    duration: Mapped[str] = mapped_column(String(100), nullable=False)
    body_part: Mapped[str] = mapped_column(String(100), nullable=True)

    # Relationships
    case = relationship("Case", back_populates="symptoms")

