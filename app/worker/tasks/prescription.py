import logging
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(name="app.worker.tasks.prescription.parse_prescription_ocr")
def parse_prescription_ocr(prescription_id: str) -> dict:
    """
    Asynchronous Celery task simulating OCR text extraction and 
    NLP ingestion of doctor prescriptions.
    """
    logger.info(f"Starting OCR/NLP parsing for Prescription {prescription_id}")
    
    result = {
        "prescription_id": prescription_id,
        "parsed_medications": [
            {
                "name": "Amoxicillin",
                "dosage": "500mg",
                "frequency": "three times daily",
                "duration": "7 days"
            }
        ],
        "confidence": 0.95
    }
    
    logger.info(f"Successfully parsed Prescription {prescription_id}")
    return result
