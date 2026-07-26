import uuid
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.report import Report
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class ReportRepository(BaseRepository[Report, BaseModel, BaseModel]):
    async def get_by_patient(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Report]:
        """
        Lists a patient's clinical reports, newest first. Bounded by default.
        """
        result = await db.execute(
            select(Report)
            .where(Report.patient_id == patient_id)
            .order_by(Report.date.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

report_repository = ReportRepository(Report)
