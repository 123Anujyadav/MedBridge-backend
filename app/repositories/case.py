import uuid
from typing import List
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.case import Case
from app.models.patient import Patient
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class CaseRepository(BaseRepository[Case, BaseModel, BaseModel]):
    async def get_by_doctor(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Case]:
        """
        Cases assigned to a doctor, most recently updated first. Bounded.
        """
        result = await db.execute(
            select(Case)
            .where(Case.doctor_id == doctor_id)
            .order_by(Case.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_patients_by_doctor(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Patient]:
        """
        Unique patients who have a consultation case with this doctor. Bounded.
        """
        subquery = select(Case.patient_id).where(Case.doctor_id == doctor_id).scalar_subquery()
        result = await db.execute(
            select(Patient)
            .where(Patient.id.in_(subquery))
            .order_by(Patient.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

case_repository = CaseRepository(Case)
