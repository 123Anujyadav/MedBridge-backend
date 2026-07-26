"""
Doctor directory implementing `DoctorDirectoryPort`.

Queries the real `doctors` table. Specialty-aware and availability-aware, so a
routed case reaches a clinician who can actually act on it — the existing
`ai_service.process_symptom_intake` path instead assigns
`select(Doctor).limit(1)`, which hands every AI-triaged case to whichever row
sorts first regardless of specialty.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.intake.application.dto import DoctorRef
from app.models.doctor import Doctor
from app.models.user import User

logger = logging.getLogger(__name__)

_ACCEPTING_AVAILABILITY = ("available", "busy")
"""
Statuses that can still receive an asynchronous case.

'busy' qualifies because intake routing queues work rather than demanding an
immediate response; 'offline' and 'on_leave' do not.
"""


class SqlDoctorDirectory:
    """Reads candidate clinicians from the primary database."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    @staticmethod
    def _base_query() -> Select:
        """Active, non-deleted doctors only."""
        return (
            select(Doctor)
            .join(User, User.id == Doctor.id)
            .where(User.is_active.is_(True))
            .where(Doctor.deleted_at.is_(None))
        )

    @staticmethod
    def _to_ref(doctor: Doctor) -> DoctorRef:
        first = (doctor.first_name or "").strip()
        last = (doctor.last_name or "").strip()
        return DoctorRef(
            doctor_id=str(doctor.id),
            full_name=f"Dr. {first} {last}".strip(),
            specialty=doctor.specialty or "General Medicine",
            hospital_name=doctor.hospital_name,
            rating=float(doctor.rating or 0.0),
            years_of_experience=int(doctor.years_of_experience or 0),
            is_available=(doctor.availability in _ACCEPTING_AVAILABILITY),
            is_verified=(doctor.verification_status == "verified"),
            avatar_url=doctor.avatar_url,
        )

    async def find_for_specialty(
        self, specialty: str, *, limit: int = 3
    ) -> list[DoctorRef]:
        """
        Best-matching clinicians for a specialty, most suitable first.

        Ranked by verification, then rating, then experience. Falls back to a
        substring match ("Cardiology" against "Interventional Cardiology") when
        no exact match exists, since specialty strings are free text on the
        doctor profile.
        """
        exact_stmt = (
            self._base_query()
            .where(Doctor.specialty.ilike(specialty))
            .where(Doctor.availability.in_(_ACCEPTING_AVAILABILITY))
            .order_by(
                Doctor.verification_status.desc(),
                Doctor.rating.desc(),
                Doctor.years_of_experience.desc(),
            )
            .limit(limit)
        )
        result = await self._db.execute(exact_stmt)
        doctors = list(result.scalars().all())

        if not doctors:
            loose_stmt = (
                self._base_query()
                .where(Doctor.specialty.ilike(f"%{specialty}%"))
                .where(Doctor.availability.in_(_ACCEPTING_AVAILABILITY))
                .order_by(Doctor.rating.desc(), Doctor.years_of_experience.desc())
                .limit(limit)
            )
            result = await self._db.execute(loose_stmt)
            doctors = list(result.scalars().all())

        logger.info(
            "[INTAKE_DIRECTORY] specialty=%s matched=%d", specialty, len(doctors)
        )
        return [self._to_ref(d) for d in doctors]

    async def get(self, doctor_id: str) -> DoctorRef | None:
        """Resolve one doctor by id, or None if absent/inactive."""
        try:
            parsed = uuid.UUID(str(doctor_id))
        except (ValueError, AttributeError, TypeError):
            logger.warning("[INTAKE_DIRECTORY_BAD_ID] doctor_id=%r", doctor_id)
            return None

        result = await self._db.execute(self._base_query().where(Doctor.id == parsed))
        doctor = result.scalars().first()
        return self._to_ref(doctor) if doctor else None
