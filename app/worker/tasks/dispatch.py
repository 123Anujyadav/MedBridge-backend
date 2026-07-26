import logging
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(name="app.worker.tasks.dispatch.send_asynchronous_notification")
def send_asynchronous_notification(notification_id: str) -> bool:
    """
    Asynchronous Celery task that handles email, SMS, and webhook triggers.
    """
    logger.info(f"Dispatching notification alert {notification_id}")
    
    # Simulates sending SMS / Email / Web Push
    
    logger.info(f"Notification alert {notification_id} successfully dispatched.")
    return True
