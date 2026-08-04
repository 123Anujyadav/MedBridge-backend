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

    # --- Prescriber snapshot -------------------------------------------------
    # A prescription is a legal record of what was ordered, by whom, on what
    # authority, on a given date. The prescriber's details are therefore copied
    # here at issue time rather than read live through `doctor`: a clinician who
    # later changes hospital, gains a qualification or renews a licence must not
    # retroactively rewrite prescriptions they signed years earlier.
    # `doctor_id` still points at the live profile for anyone who needs it.
    doctor_specialty: Mapped[str] = mapped_column(String(100), nullable=True)
    doctor_qualification: Mapped[str] = mapped_column(String(255), nullable=True)
    doctor_hospital: Mapped[str] = mapped_column(String(150), nullable=True)
    doctor_registration_number: Mapped[str] = mapped_column(String(100), nullable=True)
    doctor_experience_years: Mapped[int] = mapped_column(Integer, nullable=True)
    doctor_avatar_url: Mapped[str] = mapped_column(String(255), nullable=True)

    consultation_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """When the consultation behind this prescription took place."""

    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    """Set when the clinician signs. Null means issued but not yet signed."""

    doctor_signature_url: Mapped[str] = mapped_column(String(255), nullable=True)
    prescription_image_url: Mapped[str] = mapped_column(String(255), nullable=True)
    """A scan or photograph of a paper prescription, when one exists."""

    pdf_url: Mapped[str] = mapped_column(String(255), nullable=True)
    """Rendered printable PDF. Generated on demand and cached here."""

    @property
    def is_signed(self) -> bool:
        return self.signed_at is not None

    # Relationships
    case = relationship("Case", back_populates="prescriptions")
    patient = relationship("Patient", back_populates="prescriptions")
    doctor = relationship("Doctor", back_populates="prescriptions")
    medications = relationship("Medication", back_populates="prescription", cascade="all, delete-orphan")
    verifications = relationship(
        "PrescriptionVerification",
        back_populates="prescription",
        cascade="all, delete-orphan",
        order_by="PrescriptionVerification.created_at.desc()",
    )


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
    brand_name: Mapped[str] = mapped_column(String(150), nullable=True)
    """The branded product, when the clinician specified one."""

    strength: Mapped[str] = mapped_column(String(100), nullable=True)
    """Amount of active ingredient per unit, e.g. "500 mg"."""

    dosage: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "500mg"
    frequency: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "once daily"
    duration: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "7 days"

    food_instruction: Mapped[str] = mapped_column(String(50), nullable=True)
    """before_food | after_food | with_food | empty_stomach | anytime."""

    route: Mapped[str] = mapped_column(String(50), nullable=True)
    """oral, topical, intravenous, and so on."""

    quantity: Mapped[int] = mapped_column(Integer, nullable=True)
    """Units to dispense. Drives pharmacy stock checks and order totals."""

    rxcui: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    """
    RxNorm concept identifier resolved from `name`.

    Free-text drug names do not join reliably — "Crocin", "Paracetamol" and
    "Acetaminophen" are one ingredient under three labels. The RxCUI is the
    stable key that safety checks and pharmacy inventory lookups match on.
    Null when the name could not be normalised; callers must treat that as
    "unknown", never as "no interactions".
    """

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


