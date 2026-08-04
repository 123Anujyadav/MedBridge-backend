"""
Application service for prescription safety review and printable output.

Sits between the API layer and the `rxsafety` bounded context, and owns the two
things that context deliberately does not: loading the patient's clinical
context from the database, and deciding what a caller is allowed to see.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import EntityNotFoundException
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.rx_verification import PrescriptionVerification
from app.rxsafety.factory import get_prescription_verifier, is_enabled
from app.services.prescription_pdf import prescription_pdf_generator

logger = logging.getLogger(__name__)


def _age_from_dob(date_of_birth: Any) -> int | None:
    """
    Whole years from a stored date of birth, or None if it cannot be read.

    `patients.date_of_birth` is a free-text string column, so several formats
    are in circulation. An unparseable value yields None, which leaves the
    age-dependent checks switched off rather than guessing an age and either
    firing or suppressing an elderly warning on invented data.
    """
    if not date_of_birth:
        return None

    raw = str(date_of_birth).strip()
    parsed: date | None = None

    if isinstance(date_of_birth, (datetime, date)):
        parsed = (
            date_of_birth.date() if isinstance(date_of_birth, datetime) else date_of_birth
        )
    else:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                parsed = datetime.strptime(raw[:10], fmt).date()
                break
            except ValueError:
                continue

    if parsed is None:
        return None

    today = date.today()
    years = today.year - parsed.year - (
        (today.month, today.day) < (parsed.month, parsed.day)
    )
    return years if 0 <= years <= 130 else None


class PrescriptionSafetyService:
    async def _load(self, db: AsyncSession, prescription_id: uuid.UUID) -> Prescription:
        result = await db.execute(
            select(Prescription)
            .where(Prescription.id == prescription_id)
            .options(selectinload(Prescription.medications))
        )
        rx = result.scalar_one_or_none()
        if not rx:
            raise EntityNotFoundException("Prescription", str(prescription_id))
        return rx

    async def _patient_context(self, db: AsyncSession, patient_id: uuid.UUID) -> dict:
        """
        Clinical context the label checks are filtered against.

        A missing patient is not fatal: the review still runs, it simply cannot
        apply the patient-specific filters. Returning an empty context makes
        those checks no-ops rather than making them wrong.
        """
        patient = await db.get(Patient, patient_id)
        if not patient:
            return {}

        context: dict[str, Any] = {
            "allergies": list(getattr(patient, "allergies", None) or []),
            "conditions": list(getattr(patient, "chronic_conditions", None) or []),
        }

        age = _age_from_dob(getattr(patient, "date_of_birth", None))
        if age is not None:
            context["age"] = age

        # There is no pregnancy field on the patient record. Rather than default
        # it to False — which would silently suppress every pregnancy warning —
        # it is left unset, and the label check for it simply does not fire.
        # Wiring a real field in later turns those warnings on with no change
        # here.
        return context

    async def verify(
        self, db: AsyncSession, prescription_id: uuid.UUID
    ) -> PrescriptionVerification | None:
        """
        Run a fresh review. Returns None when the feature is switched off.

        Never raises on provider failure — a review that could not run must not
        block a prescription from being issued or read.
        """
        if not is_enabled():
            logger.info("[RXSAFETY_DISABLED] skipping review for rx=%s", prescription_id)
            return None

        rx = await self._load(db, prescription_id)
        context = await self._patient_context(db, rx.patient_id)

        try:
            return await get_prescription_verifier().verify(
                db, rx, rx.medications, context
            )
        except Exception as exc:
            # Record the failure rather than swallowing it, so the UI can say
            # "the check did not run" instead of showing nothing at all.
            logger.exception("[RXSAFETY_FAILED] rx=%s: %s", prescription_id, exc)
            record = PrescriptionVerification(
                prescription_id=rx.id,
                status="failed",
                verdict="unknown",
                confidence=0.0,
                summary=(
                    "The automated safety check could not be completed. This "
                    "prescription has not been reviewed and needs a manual check."
                ),
                unchecked_medications=[m.name for m in rx.medications],
                error=str(exc)[:2000],
            )
            db.add(record)
            return record

    async def latest_verification(
        self, db: AsyncSession, prescription_id: uuid.UUID
    ) -> PrescriptionVerification | None:
        result = await db.execute(
            select(PrescriptionVerification)
            .where(PrescriptionVerification.prescription_id == prescription_id)
            .options(selectinload(PrescriptionVerification.findings))
            .order_by(PrescriptionVerification.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_or_run_verification(
        self, db: AsyncSession, prescription_id: uuid.UUID
    ) -> PrescriptionVerification | None:
        """Return the stored review, running one if none exists yet."""
        existing = await self.latest_verification(db, prescription_id)
        if existing:
            return existing
        return await self.verify(db, prescription_id)

    async def generate_pdf(
        self, db: AsyncSession, prescription_id: uuid.UUID, *, force: bool = False
    ) -> dict:
        """
        Render the printable prescription, reusing the cached file unless forced.

        The cached path is only trusted when the file is actually on disk;
        `pdf_url` surviving a wiped uploads volume would otherwise produce a
        download link to nothing.
        """
        rx = await self._load(db, prescription_id)
        verification = await self.latest_verification(db, prescription_id)

        result = prescription_pdf_generator.generate(rx, rx.medications, verification)
        rx.pdf_url = result["file_url"]
        await db.flush()
        return result


prescription_safety_service = PrescriptionSafetyService()
