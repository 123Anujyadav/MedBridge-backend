"""
Clinical document lifecycle: versions, rendering, comparison and sharing.

Design rules this service exists to enforce:

**Nothing is overwritten.** A revision appends a version; it never edits one.
The database trigger backs this up, so even a bug cannot rewrite an approved
document.

**AI never silently replaces a clinician.** An AI regeneration always lands as
a new `ai_draft`. If the report's newest version is doctor-approved, that
version stays newest in status terms — the AI draft sits above it awaiting
review rather than superseding it.

**Identical content is not a new version.** Every snapshot is hashed; a save
that changes nothing returns the existing version and its already-rendered PDF
instead of writing a near-duplicate and re-running ReportLab.

**One generator.** Previews and downloads are the same bytes from
`report_generator`, so a preview cannot drift from what the patient receives.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AuthorizationException,
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.core.upload import UPLOADS_ROOT
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.report import Report
from app.models.report_version import ReportVersion
from app.models.user import User

logger = logging.getLogger(__name__)

# Fields that constitute the clinical document. The hash is taken over exactly
# these, so a change to any of them is a new version and a change to none is not.
SNAPSHOT_FIELDS = (
    "title", "chief_complaint", "summary", "content", "diagnosis",
    "clinical_notes", "prescription", "follow_up_instructions", "ai_findings",
    "symptoms", "recommended_tests", "recommendations",
)

# Text fields the comparison walks, in the order a reader expects them.
COMPARABLE_FIELDS = (
    ("title", "Title"),
    ("chief_complaint", "Chief Complaint"),
    ("summary", "Clinical Summary"),
    ("diagnosis", "Diagnosis"),
    ("clinical_notes", "Clinical Notes"),
    ("prescription", "Prescription"),
    ("follow_up_instructions", "Follow-up Instructions"),
    ("ai_findings", "AI Findings"),
    ("content", "Report Body"),
)

LIST_FIELDS = (
    ("symptoms", "Symptoms"),
    ("recommended_tests", "Recommended Tests"),
    ("recommendations", "Recommendations"),
)

# A report already released to the patient is not a draft any more.
NON_RESTORABLE_STATUSES = ("shared",)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _norm_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


class ReportVersionService:
    """Version history, rendering and comparison for one report."""

    # ── Access ───────────────────────────────────────────────────────────

    async def load_readable_report(
        self, db: AsyncSession, report_id: uuid.UUID, user: User
    ) -> Report:
        """
        Fetch a report the caller may read.

        A patient may only read their own report, and only once it has actually
        been released to them — an in-progress draft is not their document yet.
        """
        report = await db.get(Report, report_id)
        if report is None or report.deleted_at is not None:
            raise EntityNotFoundException("Report", str(report_id))

        if user.role == "patient":
            if report.patient_id != user.id:
                raise AuthorizationException("Access denied to this document.")
            if report.status not in ("shared", "ready", "approved", "reviewed"):
                raise AuthorizationException(
                    "This document has not been shared with you."
                )
        elif user.role == "doctor":
            linked = await db.scalar(
                select(Case.id)
                .where(Case.doctor_id == user.id)
                .where(Case.patient_id == report.patient_id)
                .limit(1)
            )
            if linked is None:
                raise AuthorizationException("Access denied to this document.")
        elif user.role != "admin":
            raise AuthorizationException("Access denied to this document.")
        return report

    async def load_editable_report(
        self, db: AsyncSession, report_id: uuid.UUID, user: User
    ) -> Report:
        """Write access is the treating doctor only — never the patient."""
        if user.role != "doctor":
            raise AuthorizationException("Only a treating clinician may revise a document.")
        return await self.load_readable_report(db, report_id, user)

    # ── Snapshot & hashing ───────────────────────────────────────────────

    @staticmethod
    def _snapshot_from_report(report: Report) -> dict[str, Any]:
        """
        The document as the live `Report` row currently holds it.

        Used to seed version 1 for a report created before version tracking, so
        history starts from what the record actually says rather than a guess.
        """
        return {
            "title": _norm(report.title),
            "chief_complaint": "",
            "summary": _norm(report.summary),
            "content": _norm(report.content),
            "diagnosis": "",
            "clinical_notes": "",
            "prescription": "",
            "follow_up_instructions": "",
            "ai_findings": "",
            "symptoms": [],
            "recommended_tests": [],
            "recommendations": [],
        }

    @staticmethod
    def compute_hash(snapshot: dict[str, Any]) -> str:
        """Stable digest over the snapshot fields, order-independent."""
        payload = {
            key: (
                _norm_list(snapshot.get(key))
                if key in {"symptoms", "recommended_tests", "recommendations"}
                else _norm(snapshot.get(key))
            )
            for key in SNAPSHOT_FIELDS
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # ── Version creation ─────────────────────────────────────────────────

    async def create_version(
        self,
        db: AsyncSession,
        *,
        report: Report,
        snapshot: dict[str, Any],
        author: Optional[User],
        author_type: str,
        author_name: str,
        status: str,
        description: str = "",
        approval_note: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        restored_from: Optional[int] = None,
        render: bool = True,
    ) -> tuple[ReportVersion, bool]:
        """
        Append a version. Returns `(version, created)`.

        `created` is False when the snapshot is byte-identical to the newest
        version — the caller gets that version back and no PDF is re-rendered.
        """
        content_hash = self.compute_hash(snapshot)
        latest = await self.latest_version(db, report.id)

        if latest is not None and latest.content_hash == content_hash:
            logger.info(
                "[REPORT_VERSION] report=%s unchanged (hash %s) — reusing v%d",
                report.id, content_hash[:8], latest.version_number,
            )
            return latest, False

        next_number = (latest.version_number if latest else 0) + 1

        version = ReportVersion(
            report_id=report.id,
            version_number=next_number,
            author_id=author.id if author else None,
            author_name=author_name[:200],
            author_type=author_type,
            status=status,
            description=description[:500],
            title=_norm(snapshot.get("title")) or report.title,
            chief_complaint=_norm(snapshot.get("chief_complaint")),
            summary=_norm(snapshot.get("summary")),
            content=_norm(snapshot.get("content")),
            diagnosis=_norm(snapshot.get("diagnosis")),
            clinical_notes=_norm(snapshot.get("clinical_notes")),
            prescription=_norm(snapshot.get("prescription")),
            follow_up_instructions=_norm(snapshot.get("follow_up_instructions")),
            ai_findings=_norm(snapshot.get("ai_findings")),
            symptoms=_norm_list(snapshot.get("symptoms")),
            recommended_tests=_norm_list(snapshot.get("recommended_tests")),
            recommendations=_norm_list(snapshot.get("recommendations")),
            ai_confidence_score=report.ai_confidence_score,
            content_hash=content_hash,
            approval_note=approval_note,
            rejection_reason=rejection_reason,
            approved_by_name=author_name if status == "approved" else None,
            approved_at=_now_iso() if status == "approved" else None,
            restored_from_version=restored_from,
        )
        db.add(version)
        await db.flush()

        report.current_version = next_number
        await db.flush()

        if render:
            await self.render_version(db, report, version)

        logger.info(
            "[REPORT_VERSION] report=%s v%d by %s (%s) status=%s",
            report.id, next_number, author_name, author_type, status,
        )
        return version, True

    async def ensure_initial_version(
        self, db: AsyncSession, report: Report, user: Optional[User] = None
    ) -> ReportVersion:
        """
        Materialise version 1 for a report that predates version tracking.

        Seeded from the report's own stored content, and attributed to the
        system rather than to whoever happened to open it.
        """
        latest = await self.latest_version(db, report.id)
        if latest is not None:
            return latest

        version, _ = await self.create_version(
            db,
            report=report,
            snapshot=self._snapshot_from_report(report),
            author=None,
            author_type="system",
            author_name="System",
            status=self._status_for_report(report),
            description="Initial version captured from the existing document.",
            render=False,
        )
        # Reuse the report's already-rendered file if it has one; there is no
        # reason to re-render a document that was generated at issue time.
        if report.file_url and not version.file_url:
            version.file_url = report.file_url
            version.file_size = report.file_size
            await db.flush()
        return version

    @staticmethod
    def _status_for_report(report: Report) -> str:
        return {
            "ready": "approved",
            "approved": "approved",
            "reviewed": "approved",
            "shared": "shared",
            "rejected": "rejected",
            "archived": "archived",
            "pending": "draft",
            "pending_review": "under_review",
            "needs_revision": "under_review",
        }.get(report.status, "draft")

    # ── Rendering ────────────────────────────────────────────────────────

    async def render_version(
        self, db: AsyncSession, report: Report, version: ReportVersion
    ) -> ReportVersion:
        """
        Render a version to PDF, once.

        Cached by construction: a version's content can never change, so a file
        that exists is permanently valid and is never regenerated.
        """
        if version.file_url and self._resolve_file(version.file_url):
            return version

        from app.services.report_generator import report_generator

        patient = await db.get(Patient, report.patient_id)
        patient_name = (
            f"{patient.first_name} {patient.last_name}".strip()
            if patient else report.patient_name
        )
        case = await db.get(Case, report.case_id) if report.case_id else None
        doctor = await db.get(Doctor, case.doctor_id) if case and case.doctor_id else None
        doctor_name = (
            f"Dr. {doctor.first_name} {doctor.last_name}".strip()
            if doctor else (report.doctor_name or "Attending Physician")
        )

        approval_info = {
            "status": version.status.replace("_", " ").title(),
            "approved_by": version.approved_by_name,
            "approved_at": version.approved_at,
            "approval_note": version.approval_note,
            "rejection_reason": version.rejection_reason,
        }

        meta = report_generator.generate_pdf(
            patient_name=patient_name,
            patient_id=str(report.patient_id),
            doctor_name=doctor_name,
            doctor_id=str(doctor.id) if doctor else "",
            symptoms=", ".join(version.symptoms) or (
                case.symptom_summary if case else "Not recorded"
            ),
            diagnosis=version.diagnosis or "Pending evaluation",
            clinical_notes=version.clinical_notes,
            medications=[],
            recommended_tests=version.recommended_tests or None,
            follow_up_date=None,
            doctor_remarks=None,
            hospital_name=report.hospital_name or "MedBridge Medical Center",
            chief_complaint=version.chief_complaint or None,
            clinical_summary=version.summary or None,
            ai_findings=version.ai_findings or None,
            prescription_text=version.prescription or None,
            follow_up_instructions=version.follow_up_instructions or None,
            recommendations=version.recommendations or None,
            approval_info=approval_info,
            version_label=f"v{version.version_number}",
            filename_stem=f"report_{str(report.id)[:8]}_v{version.version_number}",
        )

        version.file_url = meta["file_url"]
        version.file_size = meta["file_size"]
        await db.flush()
        return version

    async def render_current_document(
        self, db: AsyncSession, report: Report
    ) -> Optional[str]:
        """
        Render the report as it stands now, and return the file's path.

        Only some reports arrive with a rendered document. The consultation flow
        generates a PDF at issue time and stores it on `report.file_url`, but a
        report written by the AI intake pipeline, an uploaded record, or a lab
        result holds its text in `report.content` and has no file at all — so
        the download route, which served `file_url` and nothing else, answered
        404 for a report that plainly exists and is being displayed on screen.
        `render_version` already renders a missing document on demand; this is
        the same behaviour for the live one.

        Returns None when there is genuinely nothing to render, which is the
        caller's cue to say so rather than to serve an empty document.

        The result is deliberately **not** written back to `report.file_url`.
        Report content is editable (`PUT /doctor/reports/{id}/content`) and that
        path does not clear the stored file, so caching here would pin the
        download to superseded text. Rendering under a stable per-report name
        instead keeps one file per report, overwritten in place, always current.
        """
        if report.file_url:
            existing = self._resolve_file(report.file_url)
            if existing:
                return existing

        body = _norm(report.content) or _norm(report.summary)
        if not body:
            return None

        from app.services.report_generator import report_generator

        patient = await db.get(Patient, report.patient_id)
        patient_name = (
            f"{patient.first_name} {patient.last_name}".strip()
            if patient else report.patient_name
        )
        case = await db.get(Case, report.case_id) if report.case_id else None
        doctor = await db.get(Doctor, case.doctor_id) if case and case.doctor_id else None
        doctor_name = (
            f"Dr. {doctor.first_name} {doctor.last_name}".strip()
            if doctor else (report.doctor_name or "Attending Physician")
        )

        meta = report_generator.generate_pdf(
            patient_name=patient_name,
            patient_id=str(report.patient_id),
            doctor_name=doctor_name,
            doctor_id=str(doctor.id) if doctor else "",
            symptoms=(case.symptom_summary if case else "") or "Not recorded",
            # This document was never through a clinician's hands, so there is
            # no diagnosis to print. "Pending evaluation" is what the version
            # renderer prints in the same situation.
            diagnosis="Pending evaluation",
            clinical_notes=body,
            medications=[],
            hospital_name=report.hospital_name or "MedBridge Medical Center",
            clinical_summary=_norm(report.summary) or None,
            filename_stem=f"report_{report.id}_current",
        )
        return self._resolve_file(meta["file_url"])

    @staticmethod
    def _resolve_file(file_url: str | None) -> Optional[str]:
        """Absolute path for a stored document, or None if it is not servable."""
        stored = file_url or ""
        if not stored.startswith("/uploads/"):
            return None
        path = os.path.abspath(os.path.join(UPLOADS_ROOT, stored[len("/uploads/"):]))
        if not path.startswith(UPLOADS_ROOT + os.sep) or not os.path.isfile(path):
            return None
        return path

    # ── Queries ──────────────────────────────────────────────────────────

    async def latest_version(
        self, db: AsyncSession, report_id: uuid.UUID
    ) -> Optional[ReportVersion]:
        result = await db.execute(
            select(ReportVersion)
            .where(ReportVersion.report_id == report_id)
            .order_by(ReportVersion.version_number.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def get_version(
        self, db: AsyncSession, report_id: uuid.UUID, number: int
    ) -> ReportVersion:
        result = await db.execute(
            select(ReportVersion)
            .where(ReportVersion.report_id == report_id)
            .where(ReportVersion.version_number == number)
        )
        version = result.scalars().first()
        if version is None:
            raise EntityNotFoundException("Report version", f"v{number}")
        return version

    async def list_versions(
        self, db: AsyncSession, report: Report, *, skip: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Newest first, paginated so a long history loads lazily."""
        total = await db.scalar(
            select(func.count(ReportVersion.id))
            .where(ReportVersion.report_id == report.id)
        ) or 0

        result = await db.execute(
            select(ReportVersion)
            .where(ReportVersion.report_id == report.id)
            .order_by(ReportVersion.version_number.desc())
            .offset(skip).limit(limit)
        )
        rows = list(result.scalars().all())
        latest_number = rows[0].version_number if rows and skip == 0 else report.current_version

        return {
            "report_id": report.id,
            "report_status": report.status,
            "current_version": report.current_version,
            "total": total,
            "skip": skip,
            "limit": limit,
            "has_more": skip + len(rows) < total,
            "versions": [
                self._version_payload(v, is_latest=v.version_number == latest_number)
                for v in rows
            ],
        }

    @staticmethod
    def _version_payload(version: ReportVersion, *, is_latest: bool) -> dict[str, Any]:
        return {
            "version_number": version.version_number,
            "created_at": version.created_at.isoformat() if version.created_at else None,
            "author_name": version.author_name,
            "author_type": version.author_type,
            "status": version.status,
            "description": version.description,
            "file_url": version.file_url,
            "file_size": version.file_size,
            "content_hash": version.content_hash,
            "approval_note": version.approval_note,
            "rejection_reason": version.rejection_reason,
            "approved_by_name": version.approved_by_name,
            "approved_at": version.approved_at,
            "restored_from_version": version.restored_from_version,
            "is_latest": is_latest,
            # Historical versions are read-only in the database; saying so in
            # the payload keeps the UI from offering an action that would fail.
            "is_editable": is_latest,
        }

    # ── Comparison ───────────────────────────────────────────────────────

    async def compare(
        self, db: AsyncSession, report: Report, a: int, b: int
    ) -> dict[str, Any]:
        """
        Field-by-field diff between two versions.

        Attribution comes from the *newer* version's author: the change being
        described is the one that produced version B, so B's author is who made
        it. This is what separates an AI redraft from a clinician's edit.
        """
        if a == b:
            raise BusinessRuleValidationException(
                "Choose two different versions to compare."
            )
        left = await self.get_version(db, report.id, min(a, b))
        right = await self.get_version(db, report.id, max(a, b))

        fields: list[dict[str, Any]] = []
        for key, label in COMPARABLE_FIELDS:
            before, after = _norm(getattr(left, key)), _norm(getattr(right, key))
            if before == after:
                continue
            fields.append({
                "field": key,
                "label": label,
                "change": "added" if not before else "removed" if not after else "modified",
                "previous_value": before,
                "new_value": after,
                "segments": self._segment_diff(before, after),
            })

        for key, label in LIST_FIELDS:
            before_items = _norm_list(getattr(left, key))
            after_items = _norm_list(getattr(right, key))
            if before_items == after_items:
                continue
            before_set, after_set = set(before_items), set(after_items)
            fields.append({
                "field": key,
                "label": label,
                "change": "modified",
                "previous_value": ", ".join(before_items),
                "new_value": ", ".join(after_items),
                "added_items": sorted(after_set - before_set),
                "removed_items": sorted(before_set - after_set),
                "segments": [],
            })

        return {
            "report_id": report.id,
            "version_a": self._version_payload(left, is_latest=False),
            "version_b": self._version_payload(right, is_latest=False),
            # Who produced version B, i.e. who made the changes listed here.
            "changed_by_type": right.author_type,
            "changed_by_name": right.author_name,
            "identical": not fields,
            "fields": fields,
            "added_count": sum(1 for f in fields if f["change"] == "added"),
            "removed_count": sum(1 for f in fields if f["change"] == "removed"),
            "modified_count": sum(1 for f in fields if f["change"] == "modified"),
        }

    @staticmethod
    def _segment_diff(before: str, after: str, limit: int = 400) -> list[dict[str, str]]:
        """
        Word-level segments marking what stayed, went and arrived.

        Word granularity rather than character: a character diff of clinical
        prose renders as unreadable confetti.
        """
        if not before and not after:
            return []
        left, right = before.split(), after.split()
        segments: list[dict[str, str]] = []
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, left, right).get_opcodes():
            if tag == "equal":
                segments.append({"type": "equal", "text": " ".join(left[i1:i2])})
            elif tag == "delete":
                segments.append({"type": "removed", "text": " ".join(left[i1:i2])})
            elif tag == "insert":
                segments.append({"type": "added", "text": " ".join(right[j1:j2])})
            else:
                segments.append({"type": "removed", "text": " ".join(left[i1:i2])})
                segments.append({"type": "added", "text": " ".join(right[j1:j2])})
            if len(segments) >= limit:
                break
        return segments

    # ── Restore ──────────────────────────────────────────────────────────

    async def restore(
        self, db: AsyncSession, *, report: Report, user: User, source_number: int
    ) -> tuple[ReportVersion, bool]:
        """
        Bring an earlier version's content forward as a NEW version.

        The source version is not touched, and nothing is deleted. Blocked once
        the report has been shared: the patient already holds that document, so
        silently rewinding the clinician's copy would put the two out of sync.
        """
        if report.status in NON_RESTORABLE_STATUSES:
            raise BusinessRuleValidationException(
                "This report has already been shared with the patient and "
                "cannot be rolled back. Issue a new revision instead."
            )
        source = await self.get_version(db, report.id, source_number)
        snapshot = {key: getattr(source, key) for key in SNAPSHOT_FIELDS}

        return await self.create_version(
            db,
            report=report,
            snapshot=snapshot,
            author=user,
            author_type="doctor",
            author_name=user.email,
            status="draft",
            description=f"Restored from version {source_number}.",
            restored_from=source_number,
        )


report_version_service = ReportVersionService()
