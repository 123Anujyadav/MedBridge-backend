import logging
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.worker.celery_app import celery_app
from app.worker.tasks.jobs import send_notification_task, run_sync

logger = logging.getLogger(__name__)

async def _find_and_notify_medications() -> bool:
    from app.core.database import AsyncSessionLocal
    from app.models.prescription import Prescription, Medication
    from sqlalchemy.orm import selectinload
    from datetime import datetime

    today_str = datetime.now().strftime("%Y-%m-%d")
    
    async with AsyncSessionLocal() as db:
        # Select active medications and eagerly load prescription to get patient_id
        stmt = (
            select(Medication)
            .options(selectinload(Medication.prescription))
            .where(
                and_(
                    Medication.start_date <= today_str,
                    Medication.end_date >= today_str,
                    Medication.status == "active"
                )
            )
        )
        result = await db.execute(stmt)
        meds = result.scalars().all()

        for med in meds:
            patient_id = med.prescription.patient_id
            logger.info(f"Triggering medication reminder for Patient {patient_id}: {med.name}")
            
            # Queue the notification task (execute synchronously if in testing mode)
            from app.core.config import settings
            if settings.ENVIRONMENT.lower() in ("testing", "test"):
                send_notification_task(
                    user_id=str(patient_id),
                    title="Medicine Reminder",
                    message=f"It is time to take your dose of {med.name} ({med.dosage}). Details: {med.special_instructions}",
                    priority="high"
                )
            else:
                send_notification_task.delay(
                    user_id=str(patient_id),
                    title="Medicine Reminder",
                    message=f"It is time to take your dose of {med.name} ({med.dosage}). Details: {med.special_instructions}",
                    priority="high"
                )
        return True


@celery_app.task(name="app.worker.tasks.reminder.send_medicine_reminders")
def send_medicine_reminders() -> bool:
    """
    Periodic background task checking active daily medications and issuing notifications.
    """
    logger.info("Starting scheduled medicine reminder checks.")
    return run_sync(_find_and_notify_medications())


async def _find_and_notify_follow_ups(db: AsyncSession | None = None) -> int:
    """
    Notify treating clinicians about prescriptions whose follow-up is due.

    Sweeps for follow-up dates on or before today rather than exactly today, so
    a day the worker was down does not silently drop every follow-up scheduled
    for it. Re-running is safe: the dedupe key is per prescription per due date,
    so a prescription can only ever produce one notification for its follow-up.

    Routes through `notification_service` rather than writing rows directly, so
    these get the same deduplication, preference handling, live delivery and
    audit trail as every other notification.

    `db` is optional so the sweep can be driven with a caller-supplied session.
    The Celery task passes nothing and gets its own, which is what a worker
    needs; tests pass theirs so the sweep reads the same database they wrote to.
    """
    from datetime import datetime, timezone

    from app.core.database import AsyncSessionLocal

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if db is not None:
        return await _sweep_follow_ups(db, today_str)
    async with AsyncSessionLocal() as owned:
        sent = await _sweep_follow_ups(owned, today_str)
        await owned.commit()
        return sent


async def _sweep_follow_ups(db, today_str: str) -> int:
    """The sweep itself, independent of who owns the session."""
    from app.models.prescription import Prescription
    from app.services.notifications import notification_service

    sent = 0
    result = await db.execute(
        select(Prescription).where(
            and_(
                Prescription.follow_up_date.isnot(None),
                Prescription.follow_up_date != "",
                Prescription.follow_up_date <= today_str,
                Prescription.status.in_(("active", "verified", "parsed")),
                Prescription.deleted_at.is_(None),
            )
        )
    )

    for rx in result.scalars().all():
        created = await notification_service.notify(
            db,
            user_id=rx.doctor_id,
            category="appointment",
            type="follow_up_due",
            title="Follow-up Due",
            message=(
                f"Follow-up for {rx.patient_name} is due "
                f"({rx.follow_up_date}) — {rx.diagnosis}."
            ),
            priority="high",
            case_id=rx.case_id,
            patient_id=rx.patient_id,
            patient_name=rx.patient_name,
            action_url=f"/doctor/prescriptions?prescription={rx.id}",
            action_label="Open Patient History",
            group_key="follow_up_due",
            # Per prescription, per due date: a sweep that runs hourly
            # still produces exactly one notification per follow-up.
            dedupe_key=f"follow_up_due:{rx.id}:{rx.follow_up_date}",
        )
        if created is not None:
            sent += 1

    logger.info("[FOLLOW_UP_SWEEP] %d follow-up notification(s) issued.", sent)
    return sent


async def _check_system_health(db: AsyncSession | None = None) -> int:
    """
    Alert administrators when a core dependency is unreachable.

    Only the database and Redis are checked, because those are the two the
    platform actually pings. The monitor endpoint also reports Celery status and
    CPU/memory, but those are hardcoded constants rather than measurements —
    raising an operational alert from a fabricated number would be worse than
    not alerting at all.

    Deduped per service per hour: a dependency that stays down produces one
    alert an hour, not one per sweep.

    `db` is optional for the same reason as the follow-up sweep: the worker
    owns its session, a caller may supply one.
    """
    from datetime import datetime, timezone

    from app.core.database import AsyncSessionLocal

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H")

    if db is not None:
        return await _sweep_health(db, stamp)
    async with AsyncSessionLocal() as owned:
        sent = await _sweep_health(owned, stamp)
        await owned.commit()
        return sent


async def _sweep_health(db: AsyncSession, stamp: str) -> int:
    """The dependency check itself, independent of who owns the session."""
    from sqlalchemy import text

    from app.core.redis import redis_manager
    from app.services.notifications import notification_service

    degraded: list[str] = []

    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("[HEALTH_SWEEP] database unreachable: %s", exc)
        degraded.append("PostgreSQL database")

    if not await redis_manager.ping():
        degraded.append("Redis cache")

    if not degraded:
        logger.info("[HEALTH_SWEEP] all monitored dependencies responding.")
        return 0

    sent = 0
    for service in degraded:
        sent += await notification_service.broadcast_system_alert(
            db,
            title="Degraded Service Detected",
            message=(
                f"{service} is not responding. Background processing and "
                "session handling may be affected."
            ),
            priority="critical",
            category="system",
            roles=("admin",),
            dedupe_key=f"degraded:{service}:{stamp}",
            action_url="/admin/system",
            action_label="Open System Health",
        )
    return sent


@celery_app.task(name="app.worker.tasks.reminder.check_system_health")
def check_system_health() -> int:
    """Periodic dependency check. Scheduled by Celery beat."""
    return run_sync(_check_system_health())


@celery_app.task(name="app.worker.tasks.reminder.send_follow_up_reminders")
def send_follow_up_reminders() -> int:
    """
    Periodic sweep for due follow-ups. Scheduled by Celery beat.

    Returns the number of notifications actually issued, so a run that finds
    nothing new is distinguishable in the logs from one that failed.
    """
    logger.info("Starting scheduled follow-up due checks.")
    return run_sync(_find_and_notify_follow_ups())
