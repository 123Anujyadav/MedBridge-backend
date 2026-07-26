import uuid
from sqlalchemy import CheckConstraint, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class VitalReading(Base):
    """
    Time-series Patient Vital Measurement Reading model.
    Tracks vitals over time for health scoring and alarm triggers.
    """
    __tablename__ = "vital_readings"
    __table_args__ = (
        CheckConstraint("status IN ('normal', 'warning', 'critical')", name="vital_reading_status_check"),
    )


    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    type: Mapped[str] = mapped_column(String(50), nullable=False, index=True) # e.g. blood_pressure_systolic, heart_rate, blood_sugar
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False) # e.g. mmHg, bpm, mg/dL
    
    timestamp: Mapped[str] = mapped_column(String(50), nullable=False) # ISO timestamp
    status: Mapped[str] = mapped_column(String(50), default="normal") # normal, warning, critical

    # Relationships
    patient = relationship("Patient", back_populates="vital_readings")
