import uuid
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.hospital import Hospital
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class HospitalRepository(BaseRepository[Hospital, BaseModel, BaseModel]):
    async def get_available_emergency(self, db: AsyncSession) -> List[Hospital]:
        """
        Lists all hospitals with active emergency capacity.
        """
        result = await db.execute(
            select(Hospital).where(Hospital.emergency_capacity == "available")
        )
        return list(result.scalars().all())

hospital_repository = HospitalRepository(Hospital)
