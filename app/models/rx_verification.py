import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

from app.db.base_class import Base


# Verdicts, worst-first. Ordering matters: a prescription's overall verdict is
# the worst verdict among its findings.
VERDICT_CRITICAL = "critical"
VERDICT_WARNING = "warning"
VERDICT_SAFE = "safe"
VERDICT_UNKNOWN = "unknown"

VERDICT_SEVERITY = {
    VERDICT_CRITICAL: 3,
    VERDICT_WARNING: 2,
    VERDICT_UNKNOWN: 1,
    VERDICT_SAFE: 0,
}

FINDING_CATEGORIES = (
    "drug_interaction",
    "duplicate_therapy",
    "contraindication",
    "max_dosage",
    "allergy",
    "renal",
    "hepatic",
    "pregnancy",
    "elderly",
    "food_interaction",
)


class PrescriptionVerification(Base):
    """
    One AI-assisted safety review of a prescription.

    Reviews are append-only: re-running verification writes a new row rather
    than mutating the previous one, so what the clinician and patient were
    shown at any point stays recoverable. `Prescription.verifications` is
    ordered newest-first.

    This record never alters the prescription. It is advisory: findings are
    surfaced to the clinician and the patient, and the medication rows are
    left exactly as the prescriber wrote them.
    """

    __tablename__ = "prescription_verifications"
    __table_args__ = (
        CheckConstraint(
            "verdict IN ('safe', 'warning', 'critical', 'unknown')",
            name="rx_verification_verdict_check",
        ),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="rx_verification_confidence_check",
        ),
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed', 'degraded')",
            name="rx_verification_status_check",
        ),
    )

    prescription_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prescriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    """
    pending   — queued or running
    completed — every source answered
    degraded  — produced findings, but at least one source was unreachable
    failed    — nothing usable was produced

    `degraded` exists so a partial result is never presented as a clean bill of
    health. If openFDA was unreachable for two of five drugs, the patient is
    told which drugs were not checked.
    """

    verdict: Mapped[str] = mapped_column(String(20), default=VERDICT_UNKNOWN, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    """Plain-language explanation. LLM-written, grounded in the findings."""

    checked_medication_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    unchecked_medications: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """Drug names no source could resolve. Surfaced verbatim to the reader."""

    sources_used: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """e.g. ["rxnorm", "openfda", "groq"] — what actually answered."""

    engine_version: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=True)

    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    error: Mapped[str] = mapped_column(Text, nullable=True)

    prescription = relationship("Prescription", back_populates="verifications")
    findings = relationship(
        "VerificationFinding",
        back_populates="verification",
        cascade="all, delete-orphan",
    )


class VerificationFinding(Base):
    """
    A single safety observation, tied to the evidence that produced it.

    `evidence` is what separates this from a model guess: it holds the source
    document excerpt (an openFDA label section, an RxNorm interaction pair) the
    finding was drawn from. A finding with an empty `evidence` list and
    `source='groq'` is model-generated and is presented as such.
    """

    __tablename__ = "verification_findings"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('safe', 'warning', 'critical', 'unknown')",
            name="rx_finding_severity_check",
        ),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="rx_finding_confidence_check",
        ),
    )

    verification_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prescription_verifications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    category: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    detail: Mapped[str] = mapped_column(Text, default="", nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    """Advisory only — what a clinician might consider. Never auto-applied."""

    medications_involved: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    source: Mapped[str] = mapped_column(String(30), default="", nullable=False)
    """rxnorm | openfda | groq | rules"""

    evidence: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    """
    [{"source": "openfda", "section": "drug_interactions",
      "excerpt": "...", "reference": "https://api.fda.gov/..."}]
    """

    verification = relationship("PrescriptionVerification", back_populates="findings")
