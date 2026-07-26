"""
Immutable version history for clinical reports.

A `Report` row holds the *current* state of a document. Each `ReportVersion`
holds a complete, frozen snapshot of the document as it stood when that version
was created — not a diff, because a diff chain is only as recoverable as every
link in it, and clinical documentation must survive a corrupted link.

Immutability is enforced by a database trigger, not convention:

* DELETE is rejected outright.
* Every content column is frozen once written.
* Lifecycle columns (status, approval note, rejection reason, and the file
  produced for the version) may change only on the newest version of a report.
  A historical version is completely read-only.

`content_hash` is what stops the system regenerating an identical PDF: an
edit that changes nothing produces the same hash, and the existing version and
its rendered file are reused instead of a near-duplicate being written.
"""

import uuid

from sqlalchemy import (
    CheckConstraint, Float, ForeignKey, Index, Integer, String, Text, JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base

AUTHOR_TYPES = ("doctor", "ai", "system")

VERSION_STATUSES = (
    "draft",
    "ai_draft",
    "under_review",
    "approved",
    "rejected",
    "shared",
    "archived",
)


class ReportVersion(Base):
    """One frozen revision of a clinical report."""

    __tablename__ = "report_versions"
    __table_args__ = (
        UniqueConstraint("report_id", "version_number", name="uq_report_version_number"),
        Index("idx_report_version_lookup", "report_id", "version_number"),
        CheckConstraint(
            "author_type IN ('doctor', 'ai', 'system')",
            name="report_version_author_type_check",
        ),
        CheckConstraint(
            "status IN ('draft', 'ai_draft', 'under_review', 'approved', "
            "'rejected', 'shared', 'archived')",
            name="report_version_status_check",
        ),
        CheckConstraint("version_number >= 1", name="report_version_number_check"),
    )

    report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Authorship ───────────────────────────────────────────────────────

    author_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    author_name: Mapped[str] = mapped_column(String(200), nullable=False)
    author_type: Mapped[str] = mapped_column(
        String(20), default="doctor", nullable=False, index=True
    )
    """doctor | ai | system — what distinguishes an AI draft from a doctor edit."""

    status: Mapped[str] = mapped_column(
        String(20), default="draft", nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(String(500), default="", nullable=False)

    # ── Frozen document snapshot ─────────────────────────────────────────
    # These are what the PDF is rendered from, so a version can always be
    # re-rendered exactly as it was without depending on the live Report row.

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    chief_complaint: Mapped[str] = mapped_column(Text, default="", nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    diagnosis: Mapped[str] = mapped_column(Text, default="", nullable=False)
    clinical_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    prescription: Mapped[str] = mapped_column(Text, default="", nullable=False)
    follow_up_instructions: Mapped[str] = mapped_column(Text, default="", nullable=False)
    ai_findings: Mapped[str] = mapped_column(Text, default="", nullable=False)
    """AI-produced content, kept separate so it is labelled as such in the PDF."""

    symptoms: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    recommended_tests: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    recommendations: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    ai_confidence_score: Mapped[float] = mapped_column(Float, nullable=True)

    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    """SHA-256 of the snapshot. Identical content never becomes a new version."""

    # ── Rendered artefact ────────────────────────────────────────────────

    file_url: Mapped[str] = mapped_column(String(255), nullable=True)
    file_size: Mapped[str] = mapped_column(String(50), nullable=True)

    # ── Approval trail ───────────────────────────────────────────────────

    approval_note: Mapped[str] = mapped_column(Text, nullable=True)
    rejection_reason: Mapped[str] = mapped_column(Text, nullable=True)
    approved_by_name: Mapped[str] = mapped_column(String(200), nullable=True)
    approved_at: Mapped[str] = mapped_column(String(50), nullable=True)

    restored_from_version: Mapped[int] = mapped_column(Integer, nullable=True)
    """Set when this version was created by restoring an earlier one."""

    report = relationship("Report", back_populates="versions")
