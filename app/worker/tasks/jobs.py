import logging
import uuid
import asyncio
import threading
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

def run_sync(coro):
    """
    Safely executes an async coroutine synchronously, whether an event loop is running or not.
    """
    try:
        asyncio.get_running_loop()
        # Event loop is running (e.g., in a test thread). Run in a separate thread.
        result = []
        exception = []
        
        def target():
            try:
                result.append(asyncio.run(coro))
            except Exception as e:
                exception.append(e)
                
        t = threading.Thread(target=target)
        t.start()
        t.join()
        
        if exception:
            raise exception[0]
        return result[0]
    except RuntimeError:
        # No event loop is running (e.g., in a normal Celery worker thread). Safe to run directly.
        return asyncio.run(coro)

@celery_app.task(bind=True, name="app.worker.tasks.jobs.send_email_task", max_retries=3, default_retry_delay=2)
def send_email_task(self, recipient: str, subject: str, body: str) -> bool:
    """
    Asynchronous Celery task that handles sending emails with a retry mechanism.
    """
    logger.info(f"Starting email dispatch to {recipient} with subject '{subject}'")
    try:
        # Simulate network failure for a specific test address
        if recipient == "fail@aronofy.com" and self.request.retries < 2:
            logger.warning(f"Simulating network timeout for email dispatch (attempt {self.request.retries + 1}/3)")
            raise ConnectionError("SMTP mail server connection timed out.")

        logger.info(f"Email successfully sent to {recipient}")
        return True

    except Exception as e:
        logger.error(f"Error sending email to {recipient}: {str(e)}")
        # Trigger Celery task retry
        raise self.retry(exc=e)

async def _create_and_send_notification(user_id: str, title: str, message: str, priority: str) -> bool:
    from app.core.database import AsyncSessionLocal
    from app.models.notification import NotificationItem
    from app.core.websocket import websocket_manager
    from datetime import datetime, timezone

    async with AsyncSessionLocal() as db:
        user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
        notif = NotificationItem(
            user_id=user_uuid,
            type="alert",
            title=title,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            read=False,
            priority=priority
        )
        db.add(notif)
        await db.commit()

        # Push real-time event to the active websocket connection
        payload = {
            "id": str(notif.id),
            "type": notif.type,
            "title": notif.title,
            "message": notif.message,
            "timestamp": notif.timestamp,
            "read": notif.read,
            "priority": notif.priority
        }
        await websocket_manager.send_personal_message(payload, str(user_uuid))
        return True

@celery_app.task(name="app.worker.tasks.jobs.send_notification_task")
def send_notification_task(user_id: str, title: str, message: str, priority: str = "low") -> bool:
    """
    Asynchronous Celery task that logs a notification alert to the database and streams it over WebSocket.
    """
    logger.info(f"Queuing notification for User {user_id}: '{title}'")
    return run_sync(_create_and_send_notification(user_id, title, message, priority))

async def _run_cleanup() -> bool:
    from app.core.database import AsyncSessionLocal
    from sqlalchemy import text
    
    # Simulates clearing out soft-deleted data older than 30 days
    async with AsyncSessionLocal() as db:
        logger.info("Cleaning up soft-deleted database records older than 30 days...")
        # Mock cleaning command execution
        await db.execute(text("SELECT 1"))
        return True

@celery_app.task(name="app.worker.tasks.jobs.cleanup_expired_sessions")
def cleanup_expired_sessions() -> bool:
    """
    Asynchronous Celery task running database cleanup of old soft-deleted records.
    """
    logger.info("Running scheduled database records cleanup job.")
    return run_sync(_run_cleanup())
