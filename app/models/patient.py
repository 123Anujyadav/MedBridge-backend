import uuid
from sqlalchemy import CheckConstraint, ForeignKey, String, Float, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class Patient(Base):
    """
    Patient Clinical Profile model.
    Inherits primary key ID from User model.
    """
    __tablename__ = "patients"
    __table_args__ = (
        CheckConstraint("gender IN ('male', 'female', 'other')", name="patient_gender_check"),
        CheckConstraint("health_score >= 0 AND health_score <= 100", name="patient_health_score_check"),
        CheckConstraint("height IS NULL OR height > 0", name="patient_height_check"),
        CheckConstraint("weight IS NULL OR weight > 0", name="patient_weight_check"),
    )


    id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    date_of_birth: Mapped[str] = mapped_column(String(50), nullable=False)
    gender: Mapped[str] = mapped_column(String(50), nullable=False)
    blood_type: Mapped[str] = mapped_column(String(20), nullable=True)
    height: Mapped[float] = mapped_column(Float, nullable=True)  # cm
    weight: Mapped[float] = mapped_column(Float, nullable=True)  # kg
    address: Mapped[str] = mapped_column(String(255), nullable=True)
    city: Mapped[str] = mapped_column(String(100), nullable=True)
    state: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Complex JSON fields
    emergency_contact: Mapped[dict] = mapped_column(
        JSON,
        default=lambda: {"name": "", "phone": "", "relationship": ""},
        nullable=False
    )
    
    allergies: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    
    chronic_conditions: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    
    medications: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    
    insurance_provider: Mapped[str] = mapped_column(String(150), nullable=True)
    insurance_number: Mapped[str] = mapped_column(String(100), nullable=True)
    avatar_url: Mapped[str] = mapped_column(String(255), nullable=True)
    health_score: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    
    # HIPAA and Data consent flags
    consent_flags: Mapped[dict] = mapped_column(
        JSON,

        default=lambda: {
            "dataSharing": True,
            "researchParticipation": False,
            "emergencyAccess": True,
            "aiProcessing": True
        },
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="patient")
    cases = relationship("Case", back_populates="patient", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="patient", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="patient", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="patient", cascade="all, delete-orphan")
    emergency_requests = relationship("EmergencyRequest", back_populates="patient", cascade="all, delete-orphan")
    consent_records = relationship("ConsentRecord", back_populates="patient", cascade="all, delete-orphan")
    vital_readings = relationship("VitalReading", back_populates="patient", cascade="all, delete-orphan")
