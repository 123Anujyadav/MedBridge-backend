import uuid
from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class EmergencyRequest(Base):
    """
    Emergency Dispatch Request model.
    Facilitates low-latency GPS capture and routing coordinates to local hospitals.
    """
    __tablename__ = "emergency_requests"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'dispatched', 'arrived', 'completed', 'cancelled')", name="emergency_status_check"),
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
    
    status: Mapped[str] = mapped_column(String(50), default="active", index=True) # active, dispatched, arrived, completed, cancelled
    eta: Mapped[int] = mapped_column(Integer, nullable=True) # estimated minutes

    # Relationships
    patient = relationship("Patient", back_populates="emergency_requests")
