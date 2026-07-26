import uuid
from typing import Any, List, Optional
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.prescription import Prescription, Medication
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class PrescriptionRepository(BaseRepository[Prescription, BaseModel, BaseModel]):
    async def get(self, db: AsyncSession, id: Any) -> Optional[Prescription]:
        """
        Retrieves a prescription and eagerly loads its medications.
        """
        from sqlalchemy.orm import selectinload
        import uuid
        if isinstance(id, str):
            try:
                id = uuid.UUID(id)
            except ValueError:
                pass
        result = await db.execute(
            select(Prescription)
            .options(selectinload(Prescription.medications))
            .where(Prescription.id == id)
        )
        return result.scalars().first()

    async def get_by_patient(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Prescription]:
        """
        Prescriptions issued to a patient, newest first. Bounded by default.
        """
        from sqlalchemy.orm import selectinload
        result = await db.execute(
            select(Prescription)
            .options(selectinload(Prescription.medications))
            .where(Prescription.patient_id == patient_id)
            .order_by(Prescription.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_doctor(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Prescription]:
        """
        Prescriptions issued by a doctor, newest first. Bounded by default.
        """
        from sqlalchemy.orm import selectinload
        result = await db.execute(
            select(Prescription)
            .options(selectinload(Prescription.medications))
            .where(Prescription.doctor_id == doctor_id)
            .order_by(Prescription.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())



class MedicationRepository(BaseRepository[Medication, BaseModel, BaseModel]):
    async def get_patient_meds_for_today(
        self, db: AsyncSession, patient_id: uuid.UUID, today_str: str
    ) -> List[Medication]:
        """
        Queries all medications for a patient scheduled for today.
        Joins Medication with Prescription to filter by patient.
        """
        result = await db.execute(
            select(Medication)
            .join(Prescription)
            .where(
                and_(
                    Prescription.patient_id == patient_id,
                    Medication.start_date <= today_str,
                    Medication.end_date >= today_str,
                    Medication.status == "active",
                    Prescription.status == "active"
                )
            )
        )
        return list(result.scalars().all())

prescription_repository = PrescriptionRepository(Prescription)
medication_repository = MedicationRepository(Medication)
