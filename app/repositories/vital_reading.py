"""
Repository for time-series patient vital readings.

`vital_readings.timestamp` is an ISO-8601 string rather than a timestamptz
column, so ordering and range filters are lexicographic. That is correct for
ISO-8601 (it sorts chronologically) and lets the existing schema be queried
without a migration, but it means callers must pass ISO strings, not datetimes.
"""

from __future__ import annotations

import uuid
from typing import List, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.models.vital_reading import VitalReading
from app.repositories.base import BaseRepository


class VitalReadingRepository(BaseRepository[VitalReading, BaseModel, BaseModel]):
    async def get_by_patient(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        *,
        types: Sequence[str] | None = None,
        since_iso: str | None = None,
        skip: int = 0,
        limit: int = 500,
    ) -> List[VitalReading]:
        """
        Readings for one patient, oldest first so charts plot left-to-right.

        Filtering by `types` and `since_iso` is pushed into SQL rather than done
        in Python: a patient's vitals history is unbounded, and pulling it all
        back to filter in the application would not scale.
        """
        stmt = (
            select(VitalReading)
            .where(VitalReading.patient_id == patient_id)
            .where(VitalReading.deleted_at.is_(None))
        )
        if types:
            stmt = stmt.where(VitalReading.type.in_(list(types)))
        if since_iso:
            stmt = stmt.where(VitalReading.timestamp >= since_iso)

        stmt = stmt.order_by(VitalReading.timestamp.asc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_by_type(
        self, db: AsyncSession, patient_id: uuid.UUID, reading_type: str
    ) -> VitalReading | None:
        """Most recent reading of one type, used for current-value tiles."""
        result = await db.execute(
            select(VitalReading)
            .where(VitalReading.patient_id == patient_id)
            .where(VitalReading.type == reading_type)
            .where(VitalReading.deleted_at.is_(None))
            .order_by(VitalReading.timestamp.desc())
            .limit(1)
        )
        return result.scalars().first()


vital_reading_repository = VitalReadingRepository(VitalReading)
