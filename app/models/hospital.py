from sqlalchemy import Boolean, CheckConstraint, Float, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class Hospital(Base):
    """
    Hospital clinical facility model.
    Contains structural details, ambulance linking, and compliance levels.
    """
    __tablename__ = "hospitals"
    __table_args__ = (
        CheckConstraint("emergency_capacity IN ('available', 'limited', 'full')", name="hospital_emergency_capacity_check"),
        CheckConstraint("verification_status IN ('verified', 'pending', 'rejected', 'under_review')", name="hospital_verification_status_check"),
        CheckConstraint("rating >= 0.0 AND rating <= 5.0", name="hospital_rating_check"),
    )

    name: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    address: Mapped[str] = mapped_column(String(255), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    
    services: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    
    ambulance_linked: Mapped[bool] = mapped_column(Boolean, default=False)
    ambulance_count: Mapped[int] = mapped_column(Integer, default=0)
    emergency_capacity: Mapped[str] = mapped_column(String(50), default="available") # available, limited, full
    
    total_doctors: Mapped[int] = mapped_column(Integer, default=0)
    total_beds: Mapped[int] = mapped_column(Integer, default=0)
    available_beds: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[float] = mapped_column(Float, default=5.0)
    
    # Geolocational coordinates
    coordinates: Mapped[dict] = mapped_column(
        JSON,
        default=lambda: {"lat": 0.0, "lng": 0.0},
        nullable=False
    )
    
    logo_url: Mapped[str] = mapped_column(String(255), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(50), default="pending") # verified, pending, rejected, under_review


    # Relationships
    doctors = relationship("Doctor", back_populates="hospital")
