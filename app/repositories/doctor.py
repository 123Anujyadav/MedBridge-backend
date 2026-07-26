import uuid
from app.models.doctor import Doctor
from app.repositories.base import BaseRepository
from pydantic import BaseModel

class DoctorRepository(BaseRepository[Doctor, BaseModel, BaseModel]):
    """
    Repository for Doctor clinical profiles.
    """
    pass

doctor_repository = DoctorRepository(Doctor)
