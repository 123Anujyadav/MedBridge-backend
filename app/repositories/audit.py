import uuid
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.audit import AuditLog
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class AuditLogRepository(BaseRepository[AuditLog, BaseModel, BaseModel]):
    async def get_logs(self, db: AsyncSession, limit: int = 100) -> List[AuditLog]:
        """
        Retrieves recent audit log events sorted chronologically by timestamp.
        """
        result = await db.execute(
            select(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

audit_log_repository = AuditLogRepository(AuditLog)
