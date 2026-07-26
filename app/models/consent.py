import uuid
from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class ConsentRecord(Base):
    """
    HIPAA Privacy and Data Processing Consent Record model.
    Tracks historical agreements and grants of patient access permissions.
    """
    __tablename__ = "consent_records"

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    consent_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True) # e.g. DATA_SHARING, AI_PROCESSING
    
    granted: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    granted_at: Mapped[str] = mapped_column(String(50), nullable=False) # ISO timestamp
    expires_at: Mapped[str] = mapped_column(String(50), nullable=True) # ISO timestamp or null
    
    version: Mapped[str] = mapped_column(String(20), default="1.0", nullable=False)
    details: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="consent_records")
