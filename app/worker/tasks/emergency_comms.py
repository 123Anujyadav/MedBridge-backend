"""
The retry sweep, as a Celery task.

The same sweep also runs in-process inside the API (see
`app.services.emergency_comms.retry_sweep_loop`). Both are safe together: rows
are claimed with an atomic conditional update, so whichever worker reaches a
row first owns the attempt and a family cannot be telephoned twice.

Two runners because they fail differently. Celery needs a reachable broker; the
in-process loop needs the API to be up. An emergency notification should not
depend on either one in particular.
"""

import logging

from app.worker.celery_app import celery_app
from app.worker.tasks.jobs import run_sync

logger = logging.getLogger(__name__)


@celery_app.task(name="app.worker.tasks.emergency_comms.sweep_emergency_communications")
def sweep_emergency_communications() -> dict:
    """
    Send every emergency call or message whose retry is due.

    Idempotent: it only picks up rows still in `queued` whose `next_attempt_at`
    has passed, so running it more often than necessary costs a query and
    nothing else.
    """
    from app.services.emergency_comms import emergency_comms_service

    counts = run_sync(emergency_comms_service.sweep_retries())
    if any(counts.values()):
        logger.info("[COMMS_SWEEP] %s", counts)
    return counts
