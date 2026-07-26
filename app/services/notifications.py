"""
Real-time notification delivery and the notification centre.

Delivery reuses the platform's existing WebSocket `ConnectionManager` —
specifically `send_personal_message`, which targets one user rather than
broadcasting. No second real-time framework is introduced, and a notification
is never broadcast: pushing a clinical alert to every open socket would leak
one doctor's patients to another.

Three behaviours are worth knowing about:

**Deduplication.** Every notification carries a `dedupe_key` naming the
underlying event. A second attempt with the same key for the same user is
suppressed, so a retried request, a double-submitted form, or two code paths
observing one change cannot produce two identical cards.

**Preferences are honoured at creation.** A doctor who has switched a category
off does not get the row written at all, rather than getting it written and
hidden — a hidden row still drives the unread counter.

**Failures never propagate.** A notification is a side effect of clinical work.
If delivery fails, the consultation, report or prescription that triggered it
must still succeed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import String, cast, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EntityNotFoundException
from app.models.case import Case
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.user import User

logger = logging.getLogger(__name__)

CATEGORIES = (
    "case", "ai", "appointment", "report", "prescription",
    "patient", "system", "security", "general",
)

# Critical first, then descending. `urgent` is the legacy spelling of critical
# and sorts with it so old rows do not fall to the bottom of the list.
PRIORITY_RANK = {"critical": 0, "urgent": 0, "high": 1, "medium": 2, "low": 3}

# Categories a user may not switch off. A security alert the recipient has
# muted is worse than no alerting at all.
NON_SUPPRESSIBLE = frozenset({"security", "system"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NotificationService:
    """Creates, delivers and queries notifications."""

    # ── Creation & delivery ──────────────────────────────────────────────

    async def notify(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        category: str,
        type: str,
        title: str,
        message: str,
        priority: str = "medium",
        case_id: Optional[uuid.UUID] = None,
        patient_id: Optional[uuid.UUID] = None,
        patient_name: Optional[str] = None,
        action_url: Optional[str] = None,
        action_label: Optional[str] = None,
        group_key: Optional[str] = None,
        dedupe_key: Optional[str] = None,
        redis: Any = None,
    ) -> Optional[NotificationItem]:
        """
        Create and push one notification. Returns None when suppressed.

        Suppression happens for a duplicate `dedupe_key` or a category the
        recipient has turned off; both are normal outcomes, not errors.
        """
        if category not in CATEGORIES:
            category = "general"

        if dedupe_key:
            existing = await db.scalar(
                select(NotificationItem.id)
                .where(NotificationItem.user_id == user_id)
                .where(NotificationItem.dedupe_key == dedupe_key)
                .limit(1)
            )
            if existing is not None:
                logger.debug(
                    "[NOTIFY] suppressed duplicate key=%s user=%s", dedupe_key, user_id
                )
                return None

        if not await self._category_enabled(redis, user_id, category):
            logger.debug(
                "[NOTIFY] suppressed by preference category=%s user=%s",
                category, user_id,
            )
            return None

        item = NotificationItem(
            user_id=user_id,
            type=type[:50],
            title=title[:200],
            message=message[:500],
            timestamp=_now(),
            read=False,
            priority=priority,
            action_url=action_url[:255] if action_url else None,
            action_label=action_label[:100] if action_label else None,
            category=category,
            case_id=case_id,
            patient_id=patient_id,
            patient_name=patient_name[:200] if patient_name else None,
            group_key=(group_key or type)[:80],
            dedupe_key=dedupe_key[:160] if dedupe_key else None,
        )
        db.add(item)
        await db.flush()

        await self._push(item)
        await self._audit(db, item, "notification.created",
                          f"Notification created: {title}")
        return item

    async def notify_case_doctor(
        self, db: AsyncSession, *, case: Case, redis: Any = None, **kwargs: Any
    ) -> Optional[NotificationItem]:
        """
        Notify the clinician a case is assigned to.

        Returns None when the case has no doctor — an unassigned case is not
        silently routed to some other clinician's inbox.
        """
        if case.doctor_id is None:
            logger.info(
                "[NOTIFY] case %s has no assigned doctor; no notification sent", case.id
            )
            return None
        return await self.notify(
            db,
            user_id=case.doctor_id,
            case_id=case.id,
            patient_id=case.patient_id,
            patient_name=case.patient_name,
            redis=redis,
            **kwargs,
        )

    async def broadcast_system_alert(
        self,
        db: AsyncSession,
        *,
        title: str,
        message: str,
        priority: str = "high",
        category: str = "system",
        roles: tuple[str, ...] = ("doctor",),
        dedupe_key: Optional[str] = None,
        action_url: Optional[str] = None,
        action_label: Optional[str] = None,
        redis: Any = None,
    ) -> int:
        """
        Fan a system or security alert out to every active user in a role.

        Delivered as individual notifications rather than a socket broadcast, so
        each recipient gets a row they can read, dismiss and audit — an alert
        that vanishes on refresh is not an operational record.

        `dedupe_key` is shared across recipients; because deduplication is
        per-user, the same key correctly yields one notification each rather
        than suppressing everyone after the first.
        """
        result = await db.execute(
            select(User.id)
            .where(User.role.in_(roles))
            .where(User.is_active.is_(True))
        )
        recipients = list(result.scalars().all())

        delivered = 0
        for user_id in recipients:
            created = await self.notify(
                db,
                user_id=user_id,
                category=category,
                type=category + "_alert",
                title=title,
                message=message,
                priority=priority,
                action_url=action_url,
                action_label=action_label,
                group_key=f"{category}_alert",
                dedupe_key=dedupe_key,
                redis=redis,
            )
            if created is not None:
                delivered += 1

        logger.info(
            "[NOTIFY] system alert '%s' delivered to %d/%d %s account(s)",
            title, delivered, len(recipients), "/".join(roles),
        )
        return delivered

    async def safe_notify(self, db: AsyncSession, **kwargs: Any) -> None:
        """`notify` that never propagates — for use inside clinical writes."""
        try:
            await self.notify(db, **kwargs)
        except Exception:
            logger.exception("[NOTIFY] failed to create notification")

    async def safe_notify_case_doctor(
        self, db: AsyncSession, *, case: Case, **kwargs: Any
    ) -> None:
        try:
            await self.notify_case_doctor(db, case=case, **kwargs)
        except Exception:
            logger.exception("[NOTIFY] failed to notify case doctor")

    @staticmethod
    async def _push(item: NotificationItem) -> None:
        """
        Deliver over the existing WebSocket manager, to one user only.

        `send_personal_message`, never `broadcast`: a clinical alert on every
        open socket would hand one doctor's patient data to another.
        """
        from app.core.websocket import websocket_manager

        try:
            await websocket_manager.send_personal_message(
                {
                    "type": "NOTIFICATION_CREATED",
                    "notification_id": str(item.id),
                    "category": item.category,
                    "priority": item.priority,
                    "title": item.title,
                    "message": item.message,
                    "action_url": item.action_url,
                    "case_id": str(item.case_id) if item.case_id else None,
                },
                str(item.user_id),
            )
            item.delivered_at = _now()
        except Exception:
            # An undeliverable push is not a failure: the row exists and the
            # client will see it on its next fetch.
            logger.warning("[NOTIFY] live push failed for %s", item.id)

    @staticmethod
    async def _audit(
        db: AsyncSession, item: NotificationItem, event_type: str, description: str
    ) -> None:
        """Record against the existing audit trail, not a separate log."""
        from app.services.case_timeline import case_timeline_service

        await case_timeline_service.safe_record(
            db,
            event_type=event_type,
            description=description,
            actor_type="system",
            actor_name="Notification Service",
            case_id=item.case_id,
            patient_id=item.patient_id,
            resource="Notification",
            resource_id=str(item.id),
        )

    # ── Preferences ──────────────────────────────────────────────────────

    @staticmethod
    async def _category_enabled(redis: Any, user_id: uuid.UUID, category: str) -> bool:
        """
        Consult the existing Redis-backed settings store.

        No new preference system: `shared_service.get_user_settings` already
        owns this. Absent settings mean everything is on, and security/system
        alerts cannot be switched off at all.
        """
        if category in NON_SUPPRESSIBLE or redis is None:
            return True
        try:
            from app.services.shared import shared_service

            settings = await shared_service.get_user_settings(redis, user_id)
        except Exception:
            return True

        if settings.get("notifications_enabled") is False:
            return False
        muted = settings.get("muted_notification_categories") or []
        return category not in muted

    # ── Queries ──────────────────────────────────────────────────────────

    async def list_for_user(
        self,
        db: AsyncSession,
        user: User,
        *,
        category: Optional[str] = None,
        unread_only: bool = False,
        critical_only: bool = False,
        include_archived: bool = False,
        search: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        skip: int = 0,
        limit: int = 30,
    ) -> dict[str, Any]:
        """
        The notification centre feed for one user.

        Ownership is the `user_id` column: a user can only ever be handed their
        own rows, so there is no cross-doctor leak to guard against separately.
        Critical items are lifted to the top regardless of age — an urgent alert
        that has scrolled out of view has failed at its job.
        """
        stmt = select(NotificationItem).where(NotificationItem.user_id == user.id)

        if not include_archived:
            stmt = stmt.where(NotificationItem.archived.is_(False))
        if unread_only:
            stmt = stmt.where(NotificationItem.read.is_(False))
        if critical_only:
            stmt = stmt.where(NotificationItem.priority.in_(("critical", "urgent")))
        if category and category in CATEGORIES:
            stmt = stmt.where(NotificationItem.category == category)
        if date_from:
            stmt = stmt.where(NotificationItem.timestamp >= date_from)
        if date_to:
            upper = date_to if len(date_to) > 10 else f"{date_to}T23:59:59.999999+00:00"
            stmt = stmt.where(NotificationItem.timestamp <= upper)
        if search:
            needle = f"%{search.strip()}%"
            stmt = stmt.where(or_(
                NotificationItem.title.ilike(needle),
                NotificationItem.message.ilike(needle),
                NotificationItem.patient_name.ilike(needle),
                NotificationItem.type.ilike(needle),
                # Case id as text, so the short id shown on a card is findable.
                cast(NotificationItem.case_id, String).ilike(needle),
            ))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await db.scalar(count_stmt) or 0

        rows = list((await db.execute(
            stmt.order_by(NotificationItem.created_at.desc())
            .offset(skip).limit(limit)
        )).scalars().all())

        # Critical first, then newest. Sorted in Python because the priority
        # vocabulary is not lexically ordered ('critical' < 'low' < 'medium').
        rows.sort(key=lambda n: (
            PRIORITY_RANK.get(n.priority, 9),
            -(n.created_at.timestamp() if n.created_at else 0),
        ))

        unread = await db.scalar(
            select(func.count(NotificationItem.id))
            .where(NotificationItem.user_id == user.id)
            .where(NotificationItem.read.is_(False))
            .where(NotificationItem.archived.is_(False))
        ) or 0
        critical = await db.scalar(
            select(func.count(NotificationItem.id))
            .where(NotificationItem.user_id == user.id)
            .where(NotificationItem.read.is_(False))
            .where(NotificationItem.archived.is_(False))
            .where(NotificationItem.priority.in_(("critical", "urgent")))
        ) or 0

        return {
            "total": total,
            "returned": len(rows),
            "skip": skip,
            "limit": limit,
            "has_more": skip + len(rows) < total,
            "unread_count": unread,
            "critical_count": critical,
            "groups": self._group(rows),
            "notifications": [self._payload(n) for n in rows],
        }

    @staticmethod
    def _payload(item: NotificationItem) -> dict[str, Any]:
        return {
            "id": str(item.id),
            "type": item.type,
            "category": item.category,
            "title": item.title,
            "message": item.message,
            "priority": item.priority,
            "timestamp": item.timestamp,
            "read": item.read,
            "archived": item.archived,
            "action_url": item.action_url,
            "action_label": item.action_label,
            "case_id": str(item.case_id) if item.case_id else None,
            # Short form for the card. Absent when the notification has no case.
            "case_short_id": str(item.case_id).split("-")[0] if item.case_id else None,
            "patient_id": str(item.patient_id) if item.patient_id else None,
            "patient_name": item.patient_name,
            "group_key": item.group_key,
            "read_at": item.read_at,
            "delivered_at": item.delivered_at,
        }

    @staticmethod
    def _group(rows: list[NotificationItem]) -> list[dict[str, Any]]:
        """
        Collapse similar unread notifications into counted summaries.

        Only groups of two or more are reported; a "group" of one is just the
        notification, and presenting it as a group would overstate the volume.
        """
        buckets: dict[str, dict[str, Any]] = {}
        for item in rows:
            if item.read:
                continue
            key = item.group_key or item.type
            bucket = buckets.setdefault(key, {
                "group_key": key,
                "category": item.category,
                "label": item.title,
                "count": 0,
                "highest_priority": item.priority,
            })
            bucket["count"] += 1
            if PRIORITY_RANK.get(item.priority, 9) < PRIORITY_RANK.get(
                bucket["highest_priority"], 9
            ):
                bucket["highest_priority"] = item.priority

        return sorted(
            (b for b in buckets.values() if b["count"] > 1),
            key=lambda b: (PRIORITY_RANK.get(b["highest_priority"], 9), -b["count"]),
        )

    # ── Mutations ────────────────────────────────────────────────────────

    async def mark_read(
        self, db: AsyncSession, user: User, notification_id: uuid.UUID
    ) -> NotificationItem:
        item = await db.get(NotificationItem, notification_id)
        if item is None or item.user_id != user.id:
            # Same response for "not yours" and "does not exist", so the
            # endpoint cannot be used to probe for other users' notifications.
            raise EntityNotFoundException("Notification", str(notification_id))

        if not item.read:
            item.read = True
            item.read_at = _now()
            await db.flush()
            await self._audit(db, item, "notification.read",
                              f"Notification read: {item.title}")
        return item

    async def mark_all_read(self, db: AsyncSession, user: User) -> int:
        result = await db.execute(
            update(NotificationItem)
            .where(NotificationItem.user_id == user.id)
            .where(NotificationItem.read.is_(False))
            .where(NotificationItem.archived.is_(False))
            .values(read=True, read_at=_now())
        )
        await db.flush()
        return result.rowcount or 0

    async def mark_selected_read(
        self, db: AsyncSession, user: User, ids: Iterable[uuid.UUID]
    ) -> int:
        ids = list(ids)
        if not ids:
            return 0
        result = await db.execute(
            update(NotificationItem)
            .where(NotificationItem.user_id == user.id)
            .where(NotificationItem.id.in_(ids))
            .where(NotificationItem.read.is_(False))
            .values(read=True, read_at=_now())
        )
        await db.flush()
        return result.rowcount or 0

    async def archive(
        self, db: AsyncSession, user: User, notification_id: uuid.UUID
    ) -> NotificationItem:
        item = await db.get(NotificationItem, notification_id)
        if item is None or item.user_id != user.id:
            raise EntityNotFoundException("Notification", str(notification_id))

        item.archived = True
        if not item.read:
            item.read = True
            item.read_at = _now()
        await db.flush()
        await self._audit(db, item, "notification.dismissed",
                          f"Notification dismissed: {item.title}")
        return item

    async def record_opened(
        self, db: AsyncSession, user: User, notification_id: uuid.UUID
    ) -> NotificationItem:
        """Mark read and log that the doctor followed the action link."""
        item = await self.mark_read(db, user, notification_id)
        await self._audit(db, item, "notification.opened",
                          f"Notification action opened: {item.title}")
        return item


notification_service = NotificationService()
