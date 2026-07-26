"""
Bulk actions for the AI Reports list.

Design constraints that shaped this:

**Partial success is the normal case.** A batch of forty reports will routinely
contain some already in the target state, some without a linked case, and some
the caller cannot touch. None of those is a reason to abort the other
thirty-seven, so every item resolves to completed / skipped / failed
independently and the caller gets a per-item breakdown.

**Ownership is enforced by intersection, not iteration.** The requested ids are
filtered through the caller's own patients in a single query. An id that does
not come back is reported with a deliberately non-committal reason — confirming
"exists but not yours" would turn this endpoint into a report-enumeration
oracle.

**Constant query count.** Reports are loaded once, partitioned in Python, and
written with one bulk UPDATE per outcome group. A five-hundred-report batch
costs the same handful of queries as a five-report one.

**Only clinically safe operations.** Prescribing and diagnosis finalisation are
absent by design: both need per-patient judgement and must never be applied to
a selection.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.core.upload import UPLOADS_ROOT
from app.models.audit import AuditLog
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.report import Report
from app.models.user import User
from app.schemas.doctor_api import BulkReportAction

logger = logging.getLogger(__name__)

SYNC_THRESHOLD = 25
"""Batches at or below this run inline; larger ones move to the background so
the request returns immediately with a job to poll."""

JOB_TTL_SECONDS = 3600

# action -> (column updates, statuses that make the action a no-op)
_STATUS_ACTIONS: dict[BulkReportAction, tuple[dict[str, Any], tuple[str, ...]]] = {
    BulkReportAction.APPROVE: ({"status": "ready"}, ("ready", "approved")),
    BulkReportAction.REJECT: ({"status": "rejected"}, ("rejected",)),
    BulkReportAction.ARCHIVE: ({"status": "archived"}, ("archived",)),
    BulkReportAction.MARK_REVIEWED: ({"status": "reviewed"}, ("reviewed",)),
    BulkReportAction.REMOVE_REVIEW_FLAG: (
        {"status": "pending_review"},
        ("pending_review",),
    ),
    BulkReportAction.FLAG_FOLLOW_UP: ({"flagged_for_follow_up": True}, ()),
}

_INACCESSIBLE = "Not found, or not accessible to you."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReportBulkService:
    """Executes and tracks bulk operations over a doctor's own reports."""

    # ── Selection ────────────────────────────────────────────────────────

    async def owned_report_ids(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        *,
        status: str | None = None,
        urgency: str | None = None,
        flagged: bool | None = None,
    ) -> list[uuid.UUID]:
        """
        Every report id matching the caller's filters.

        Backs "Select all matching current filters": the client never has to
        page through the list to build a selection, and the set is resolved
        server-side so it cannot include anything the caller may not touch.
        """
        stmt = select(Report.id).where(
            Report.patient_id.in_(self._own_patients(doctor_id))
        )
        stmt = self._apply_filters(stmt, status=status, urgency=urgency, flagged=flagged)
        result = await db.execute(stmt.order_by(Report.created_at.desc()))
        return list(result.scalars().all())

    @staticmethod
    def _own_patients(doctor_id: uuid.UUID):
        return (
            select(Case.patient_id).where(Case.doctor_id == doctor_id).scalar_subquery()
        )

    @staticmethod
    def _apply_filters(stmt, *, status=None, urgency=None, flagged=None):
        if status:
            stmt = stmt.where(Report.status == status)
        if flagged is not None:
            stmt = stmt.where(Report.flagged_for_follow_up.is_(flagged))
        if urgency:
            # Urgency lives on the case, so this is an EXISTS against cases
            # rather than a join that would duplicate report rows.
            stmt = stmt.where(
                Report.case_id.in_(
                    select(Case.id).where(Case.urgency_level == urgency).scalar_subquery()
                )
            )
        return stmt

    # ── Execution ────────────────────────────────────────────────────────

    async def start(
        self,
        db: AsyncSession,
        *,
        user: User,
        action: BulkReportAction,
        report_ids: list[uuid.UUID],
        reason: str | None,
        target_doctor_id: uuid.UUID | None,
        ip_address: str,
        redis: Any,
        background: Any = None,
    ) -> dict[str, Any]:
        """
        Run a bulk action, inline for small batches or in the background for
        large ones. Returns the job state either way.

        `background` is a FastAPI `BackgroundTasks`. It matters that the work is
        registered there rather than with `asyncio.create_task`: the latter
        shares the event loop with response serialisation, and a large batch
        starved the acknowledgement for seconds. BackgroundTasks runs strictly
        after the response has been sent.
        """
        job_id = str(uuid.uuid4())

        if action is BulkReportAction.ASSIGN_SPECIALIST:
            # Validated before any work starts: a bad target is a request error,
            # not a per-item failure.
            await self._require_verified_doctor(db, target_doctor_id)

        job = {
            "job_id": job_id,
            "action": action.value,
            "status": "running",
            "total": len(report_ids),
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "items": [],
            "started_at": _now(),
            "finished_at": None,
            "message": "",
        }
        # Deliberately not persisted yet. Each store is a network round trip,
        # and when Redis is unreachable the client pays a connect timeout for
        # it — the initial write bought nothing that the write below does not.

        if len(report_ids) <= SYNC_THRESHOLD:
            job = await self._execute(
                db, user=user, action=action, report_ids=report_ids,
                reason=reason, target_doctor_id=target_doctor_id,
                ip_address=ip_address, job=job,
            )
            await self._save_job(redis, job)
            return job

        # Larger batches detach from the request. A fresh session is used
        # because the request-scoped one is closed as soon as we return.
        job["status"] = "queued"
        job["message"] = (
            f"{len(report_ids)} reports queued. Poll this job for progress."
        )
        await self._save_job(redis, job)

        runner = self._run_detached(
            job=dict(job), user_id=user.id, action=action,
            report_ids=report_ids, reason=reason,
            target_doctor_id=target_doctor_id, ip_address=ip_address, redis=redis,
        )
        if background is not None:
            background.add_task(runner)
        else:  # pragma: no cover - direct service use outside a request
            asyncio.get_running_loop().create_task(runner())
        return job

    def _run_detached(self, *, job, user_id, action, report_ids, reason,
                      target_doctor_id, ip_address, redis):
        """Build the deferred coroutine factory BackgroundTasks will invoke."""

        async def _run() -> None:
            await self._process_detached(
                job=job, user_id=user_id, action=action, report_ids=report_ids,
                reason=reason, target_doctor_id=target_doctor_id,
                ip_address=ip_address, redis=redis,
            )

        return _run

    async def _process_detached(self, *, job, user_id, action, report_ids,
                                reason, target_doctor_id, ip_address, redis) -> None:
        from app.core.database import AsyncSessionLocal

        job["status"] = "running"
        await self._save_job(redis, job)
        try:
            async with AsyncSessionLocal() as db:
                user = await db.get(User, user_id)
                if user is None:  # pragma: no cover - session outlived the user
                    raise EntityNotFoundException("User", str(user_id))
                job = await self._execute(
                    db, user=user, action=action, report_ids=report_ids,
                    reason=reason, target_doctor_id=target_doctor_id,
                    ip_address=ip_address, job=job,
                )
                await db.commit()
        except Exception as exc:
            logger.exception("[BULK_REPORTS] background job %s failed", job["job_id"])
            job["status"] = "completed"
            job["finished_at"] = _now()
            job["message"] = f"Batch aborted: {exc}"
        await self._save_job(redis, job)

    async def _execute(
        self,
        db: AsyncSession,
        *,
        user: User,
        action: BulkReportAction,
        report_ids: list[uuid.UUID],
        reason: str | None,
        target_doctor_id: uuid.UUID | None,
        ip_address: str,
        job: dict[str, Any],
    ) -> dict[str, Any]:
        items: list[dict[str, str]] = []

        # One query establishes both existence and ownership.
        result = await db.execute(
            select(Report)
            .where(Report.id.in_(report_ids))
            .where(Report.patient_id.in_(self._own_patients(user.id)))
        )
        owned = {r.id: r for r in result.scalars().all()}

        for rid in report_ids:
            if rid not in owned:
                items.append(
                    {"report_id": str(rid), "outcome": "skipped", "detail": _INACCESSIBLE}
                )

        try:
            if action is BulkReportAction.ASSIGN_SPECIALIST:
                items += await self._assign_specialist(
                    db, owned=owned, target_doctor_id=target_doctor_id
                )
            else:
                items += await self._apply_column_action(db, action=action, owned=owned)
        except Exception as exc:
            logger.exception("[BULK_REPORTS] action %s failed", action.value)
            for rid in owned:
                items.append(
                    {"report_id": str(rid), "outcome": "failed", "detail": str(exc)[:200]}
                )

        job["items"] = items
        job["completed"] = sum(1 for i in items if i["outcome"] == "completed")
        job["skipped"] = sum(1 for i in items if i["outcome"] == "skipped")
        job["failed"] = sum(1 for i in items if i["outcome"] == "failed")
        job["status"] = "completed"
        job["finished_at"] = _now()
        job["message"] = (
            f"{job['completed']} updated, {job['skipped']} skipped, "
            f"{job['failed']} failed."
        )

        await self._record_audit(
            db, user=user, action=action, job=job, reason=reason,
            ip_address=ip_address, target_doctor_id=target_doctor_id,
        )
        return job

    async def _apply_column_action(
        self,
        db: AsyncSession,
        *,
        action: BulkReportAction,
        owned: dict[uuid.UUID, Report],
    ) -> list[dict[str, str]]:
        """Status / flag changes, written as a single bulk UPDATE."""
        values, no_op_statuses = _STATUS_ACTIONS[action]
        items: list[dict[str, str]] = []
        to_update: list[uuid.UUID] = []

        for rid, report in owned.items():
            if action is BulkReportAction.FLAG_FOLLOW_UP:
                if report.flagged_for_follow_up:
                    items.append({"report_id": str(rid), "outcome": "skipped",
                                  "detail": "Already flagged for follow-up."})
                    continue
            elif report.status in no_op_statuses:
                items.append({"report_id": str(rid), "outcome": "skipped",
                              "detail": f"Already {report.status}."})
                continue
            to_update.append(rid)

        if to_update:
            await db.execute(
                update(Report).where(Report.id.in_(to_update)).values(**values)
            )
            await db.flush()
            items += [
                {"report_id": str(rid), "outcome": "completed",
                 "detail": ", ".join(f"{k}={v}" for k, v in values.items())}
                for rid in to_update
            ]
        return items

    async def _assign_specialist(
        self,
        db: AsyncSession,
        *,
        owned: dict[uuid.UUID, Report],
        target_doctor_id: uuid.UUID | None,
    ) -> list[dict[str, str]]:
        """
        Refer each report's case to another clinician.

        A report has no specialist of its own — the case carries the assignment,
        so this reassigns `cases.doctor_id`. Reports with no linked case are
        skipped rather than guessing which case they belong to.
        """
        doctor = await db.get(Doctor, target_doctor_id)
        items: list[dict[str, str]] = []

        case_ids = {r.case_id for r in owned.values() if r.case_id}
        by_report = {rid: r.case_id for rid, r in owned.items()}

        for rid, case_id in by_report.items():
            if case_id is None:
                items.append({"report_id": str(rid), "outcome": "skipped",
                              "detail": "No linked case to assign."})

        if case_ids:
            await db.execute(
                update(Case)
                .where(Case.id.in_(case_ids))
                .values(
                    doctor_id=doctor.id,
                    doctor_name=f"Dr. {doctor.first_name} {doctor.last_name}".strip(),
                    specialty=doctor.specialty,
                    status="routed",
                    assigned_at=datetime.now(timezone.utc),
                )
            )
            await db.flush()
            items += [
                {"report_id": str(rid), "outcome": "completed",
                 "detail": f"Case referred to Dr. {doctor.last_name} ({doctor.specialty})."}
                for rid, case_id in by_report.items()
                if case_id is not None
            ]
        return items

    @staticmethod
    async def _require_verified_doctor(
        db: AsyncSession, doctor_id: uuid.UUID | None
    ) -> Doctor:
        doctor = await db.get(Doctor, doctor_id) if doctor_id else None
        if doctor is None:
            raise EntityNotFoundException("Doctor", str(doctor_id))
        if doctor.verification_status != "verified":
            raise BusinessRuleValidationException(
                "Reports can only be assigned to a verified clinician."
            )
        return doctor

    # ── Audit ────────────────────────────────────────────────────────────

    async def _record_audit(
        self,
        db: AsyncSession,
        *,
        user: User,
        action: BulkReportAction,
        job: dict[str, Any],
        reason: str | None,
        ip_address: str,
        target_doctor_id: uuid.UUID | None,
    ) -> None:
        """
        One HIPAA audit row per batch, via the existing `audit_logs` table.

        The affected ids are stored in `details` rather than as one row per
        report: a batch is a single clinician decision, and exploding it into
        five hundred rows would bury every other event in the trail.
        """
        affected = [i["report_id"] for i in job["items"] if i["outcome"] == "completed"]
        details = {
            "batch_action": action.value,
            "requested": job["total"],
            "completed": job["completed"],
            "skipped": job["skipped"],
            "failed": job["failed"],
            "reason": (reason or "").strip() or None,
            "target_doctor_id": str(target_doctor_id) if target_doctor_id else None,
            "affected_report_ids": affected,
        }
        db.add(
            AuditLog(
                user_id=user.id,
                user_name=user.email,
                user_role=user.role,
                action=f"BULK_{action.value.upper()}_REPORTS",
                resource="ReportBatch",
                resource_id=job["job_id"],
                # Server-derived, never taken from the request body.
                ip_address=ip_address,
                status="success" if job["failed"] == 0 else "warning",
                details=json.dumps(details),
            )
        )
        await db.flush()
        logger.info(
            "[BULK_REPORTS] doctor=%s action=%s job=%s completed=%d skipped=%d failed=%d",
            user.id, action.value, job["job_id"],
            job["completed"], job["skipped"], job["failed"],
        )

    # ── Job storage ──────────────────────────────────────────────────────

    @staticmethod
    def _job_key(job_id: str) -> str:
        return f"bulk_report_job:{job_id}"

    async def _save_job(self, redis: Any, job: dict[str, Any]) -> None:
        """
        Persist progress.

        Redis is the natural home for ephemeral job state, and the platform's
        client already degrades to an in-memory store when Redis is down — so a
        Redis outage costs progress polling, not the batch itself.
        """
        try:
            await redis.set(
                self._job_key(job["job_id"]), json.dumps(job), ex=JOB_TTL_SECONDS
            )
        except Exception:
            logger.warning(
                "[BULK_REPORTS] could not persist job %s progress", job["job_id"]
            )

    async def get_job(self, redis: Any, job_id: str) -> dict[str, Any]:
        raw = None
        try:
            raw = await redis.get(self._job_key(job_id))
        except Exception:
            logger.warning("[BULK_REPORTS] could not read job %s", job_id)
        if not raw:
            raise EntityNotFoundException("Bulk job", job_id)
        return json.loads(raw)

    # ── Export ───────────────────────────────────────────────────────────

    async def _load_owned(
        self, db: AsyncSession, doctor_id: uuid.UUID, report_ids: Iterable[uuid.UUID]
    ) -> list[Report]:
        result = await db.execute(
            select(Report)
            .where(Report.id.in_(list(report_ids)))
            .where(Report.patient_id.in_(self._own_patients(doctor_id)))
            .order_by(Report.created_at.desc())
        )
        return list(result.scalars().all())

    async def export_csv(
        self, db: AsyncSession, doctor_id: uuid.UUID, report_ids: list[uuid.UUID]
    ) -> str:
        """Metadata export. Report bodies are deliberately excluded — a CSV of
        free-text clinical narrative is a PHI spill waiting to be emailed."""
        reports = await self._load_owned(db, doctor_id, report_ids)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "report_id", "case_id", "patient_name", "title", "type", "status",
            "date", "doctor_name", "hospital_name", "ai_generated",
            "ai_confidence_score", "flagged_for_follow_up", "summary",
        ])
        for r in reports:
            writer.writerow([
                str(r.id), str(r.case_id) if r.case_id else "", r.patient_name,
                r.title, r.type, r.status, r.date, r.doctor_name or "",
                r.hospital_name or "", r.ai_generated,
                r.ai_confidence_score if r.ai_confidence_score else "",
                r.flagged_for_follow_up, (r.summary or "").replace("\n", " "),
            ])
        return buffer.getvalue()

    async def export_pdf_bundle(
        self, db: AsyncSession, doctor_id: uuid.UUID, report_ids: list[uuid.UUID]
    ) -> tuple[bytes, list[str]]:
        """
        Zip the stored PDFs for the selection.

        Returns the archive plus the list of reports that had no retrievable
        file, so the caller can report skips instead of silently shipping a
        smaller bundle than the doctor selected.
        """
        reports = await self._load_owned(db, doctor_id, report_ids)
        skipped: list[str] = []
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for r in reports:
                path = self._pdf_path(r)
                if path is None:
                    skipped.append(r.title)
                    continue
                safe = "".join(
                    c for c in r.title if c.isalnum() or c in " -_"
                ).strip() or "report"
                archive.writestr(f"{safe}-{str(r.id)[:8]}.pdf", open(path, "rb").read())

        return buffer.getvalue(), skipped

    @staticmethod
    def _pdf_path(report: Report) -> Optional[str]:
        """Resolve a stored file, refusing anything outside the uploads tree."""
        stored = report.file_url or ""
        if not stored.startswith("/uploads/"):
            return None
        path = os.path.abspath(
            os.path.join(UPLOADS_ROOT, stored[len("/uploads/"):])
        )
        if not path.startswith(UPLOADS_ROOT + os.sep) or not os.path.isfile(path):
            return None
        return path


report_bulk_service = ReportBulkService()
