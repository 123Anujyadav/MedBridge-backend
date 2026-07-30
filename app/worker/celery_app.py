import logging
import sys
from celery import Celery, Task
from celery.schedules import crontab
from app.core.config import settings
from app.middleware.logging import correlation_id_ctx, register_tracing_filter

logger = logging.getLogger(__name__)

class ContextTask(Task):
    """
    Custom Celery Task that propagates Correlation IDs & Tracing headers
    from API requests into background worker execution contexts.
    """
    abstract = True

    def __call__(self, *args, **kwargs):
        register_tracing_filter()
        
        corr_id = None
        if self.request and self.request.headers:
            corr_id = self.request.headers.get("correlation_id") or self.request.headers.get("X-Correlation-ID")
            
        token = None
        if corr_id:
            token = correlation_id_ctx.set(corr_id)
            
        try:
            logger.info(f"Executing Celery task {self.name} [{self.request.id}]")
            return super().__call__(*args, **kwargs)
        except Exception as exc:
            logger.error(f"Celery task {self.name} [{self.request.id}] failed: {str(exc)}", exc_info=True)
            raise exc
        finally:
            if token:
                correlation_id_ctx.reset(token)

# Instantiate Celery Application
celery_app = Celery(
    "aronofy_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Set custom Task class as default
celery_app.Task = ContextTask

# Configure task execution policies, retry backoffs, dead letter handling, and timeouts
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_time_limit=300,        # Hard timeout (5 mins)
    task_soft_time_limit=240,   # Soft timeout (4 mins)
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="default",
    task_queues={
        "default": {"exchange": "default", "routing_key": "default"},
        "dead_letter": {"exchange": "dead_letter", "routing_key": "dead_letter"},
    },
    imports=[
        "app.worker.tasks.triage",
        "app.worker.tasks.prescription",
        "app.worker.tasks.dispatch",
        "app.worker.tasks.jobs",
        "app.worker.tasks.reminder",
        "app.worker.tasks.emergency_comms",
    ],
    # Periodic schedule. There was none before, which is why the reminder task
    # that already existed had never actually run.
    #
    # The follow-up sweep is idempotent — its dedupe key is per prescription per
    # due date — so an hourly cadence costs nothing when there is no new work
    # and recovers automatically from a window where the worker was down.
    beat_schedule={
        "follow-up-due-sweep": {
            "task": "app.worker.tasks.reminder.send_follow_up_reminders",
            "schedule": crontab(minute=0),  # hourly, on the hour
        },
        "medicine-reminders": {
            "task": "app.worker.tasks.reminder.send_medicine_reminders",
            "schedule": crontab(hour=8, minute=0),  # daily, 08:00 UTC
        },
        # Emergency retries are swept far more often than anything else here:
        # a queued alert that waits an hour is an alert nobody received. The
        # sweep is a no-op when there is nothing due.
        "emergency-communication-retries": {
            "task": "app.worker.tasks.emergency_comms.sweep_emergency_communications",
            "schedule": 30.0,
        },
        "system-health-sweep": {
            "task": "app.worker.tasks.reminder.check_system_health",
            "schedule": crontab(minute="*/15"),
        },
    },
    timezone="UTC",
)

if "pytest" in sys.modules:
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True
    )
