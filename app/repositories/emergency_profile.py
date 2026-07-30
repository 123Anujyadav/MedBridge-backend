from app.models.emergency_profile import EmergencyProfile
from app.repositories.base import BaseRepository
from pydantic import BaseModel


class EmergencyProfileRepository(BaseRepository[EmergencyProfile, BaseModel, BaseModel]):
    """
    Repository for patient Emergency Profiles.

    The base class is enough: the profile's primary key is the patient's id, so
    `get(db, patient_id)` is already the only lookup this module needs and there
    is no query here that could accidentally be written without a patient scope.
    """

    pass


emergency_profile_repository = EmergencyProfileRepository(EmergencyProfile)
