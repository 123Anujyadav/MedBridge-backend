import uuid
from typing import List
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.appointment import Appointment
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class AppointmentRepository(BaseRepository[Appointment, BaseModel, BaseModel]):
    async def get_by_patient(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Appointment]:
        """
        Lists a patient's appointments, most recent first.

        Bounded by default: an unpaginated query grows linearly with a patient's
        history and eventually times out the dashboard.
        """
        result = await db.execute(
            select(Appointment)
            .where(Appointment.patient_id == patient_id)
            .order_by(Appointment.date.desc(), Appointment.time.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def check_conflict(
        self, db: AsyncSession, doctor_id: uuid.UUID, date: str, time: str
    ) -> bool:
        """
        Checks if a doctor is already booked for a specific date and time slot.
        Returns True if a conflict exists.
        """
        result = await db.execute(
            select(Appointment).where(
                and_(
                    Appointment.doctor_id == doctor_id,
                    Appointment.date == date,
                    Appointment.time == time,
                    Appointment.status.in_(["scheduled", "confirmed", "in_progress"])
                )
            )
        )
        return result.scalars().first() is not None

appointment_repository = AppointmentRepository(Appointment)
