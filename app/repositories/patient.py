import uuid
from app.models.patient import Patient
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class PatientRepository(BaseRepository[Patient, BaseModel, BaseModel]):
    """
    Repository for Patient clinical profiles.
    """
    pass

patient_repository = PatientRepository(Patient)
