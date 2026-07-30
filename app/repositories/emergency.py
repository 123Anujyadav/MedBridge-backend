import uuid
from typing import Optional
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.emergency import ACTIVE_SOS_STATUSES, EmergencyRequest
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class EmergencyRequestRepository(BaseRepository[EmergencyRequest, BaseModel, BaseModel]):
    async def get_active_by_patient(
        self, db: AsyncSession, patient_id: uuid.UUID
    ) -> Optional[EmergencyRequest]:
        """
        Retrieves the current active emergency request for a patient, if any.
        """
        result = await db.execute(
            select(EmergencyRequest)
            .where(
                and_(
                    EmergencyRequest.patient_id == patient_id,
                    EmergencyRequest.status.in_(ACTIVE_SOS_STATUSES)
                )
            )
            .order_by(EmergencyRequest.created_at.desc())
        )
        return result.scalars().first()

emergency_repository = EmergencyRequestRepository(EmergencyRequest)
