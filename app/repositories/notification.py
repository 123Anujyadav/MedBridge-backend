import uuid
from typing import List
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.notification import NotificationItem
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class NotificationRepository(BaseRepository[NotificationItem, BaseModel, BaseModel]):
    async def get_by_user(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[NotificationItem]:
        """
        Lists a user's notifications, newest first. Bounded by default —
        notification volume grows without limit over an account's lifetime.
        """
        result = await db.execute(
            select(NotificationItem)
            .where(NotificationItem.user_id == user_id)
            .order_by(NotificationItem.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_unread_count(self, db: AsyncSession, user_id: uuid.UUID) -> int:
        """
        Aggregates unread notification count for a user.
        """
        result = await db.execute(
            select(func.count(NotificationItem.id))
            .where(
                and_(
                    NotificationItem.user_id == user_id,
                    NotificationItem.read == False
                )
            )
        )
        return result.scalar() or 0

notification_repository = NotificationRepository(NotificationItem)
