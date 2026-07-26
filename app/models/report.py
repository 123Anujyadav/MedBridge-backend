import uuid
from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class Report(Base):
    """
    Clinical Medical Report model.
    Contains results, vitals, AI insights, and documents references.
    """
    __tablename__ = "reports"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'ready', 'reviewed', 'shared', 'pending_review', "
            "'approved', 'rejected', 'needs_revision', 'archived')",
            name="report_status_check",
        ),
    )


    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    """
    Consultation case this report was issued from, when there is one.

    Nullable because lab results and imaging reports arrive without a case.
    `SET NULL` rather than `CASCADE`: an issued clinical report is part of the
    patient's permanent record and must outlive the case that produced it.
    """

    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    
    type: Mapped[str] = mapped_column(String(100), nullable=False, index=True) # lab_result, ai_report, imaging, discharge_summary, vital_signs, etc.
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    doctor_name: Mapped[str] = mapped_column(String(200), nullable=True)
    hospital_name: Mapped[str] = mapped_column(String(150), nullable=True)
    
    date: Mapped[str] = mapped_column(String(50), nullable=False)  # ISO Date YYYY-MM-DD
    status: Mapped[str] = mapped_column(String(50), default="ready") # pending, ready, reviewed, shared
    
    file_url: Mapped[str] = mapped_column(String(255), nullable=True)
    file_size: Mapped[str] = mapped_column(String(50), nullable=True)
    
    flagged_for_follow_up: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    """
    Clinician follow-up marker, orthogonal to `status`.

    A report can be approved *and* need follow-up, so this cannot be folded into
    the status vocabulary without losing one of the two facts.
    """

    ai_generated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ai_confidence_score: Mapped[float] = mapped_column(Float, nullable=True)
    
    tags: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    
    # Custom Vitals snapshot
    vitals: Mapped[dict] = mapped_column(
        JSON,
        nullable=True

    )

    current_version: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    """
    Highest version number issued for this report. 0 for legacy rows that
    predate version tracking — those are surfaced as a derived "current state"
    entry rather than being back-dated into a version they never had.
    """

    # Relationships
    patient = relationship("Patient", back_populates="reports")
    versions = relationship(
        "ReportVersion",
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="ReportVersion.version_number",
    )
