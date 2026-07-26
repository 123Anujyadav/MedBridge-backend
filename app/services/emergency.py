import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import EntityNotFoundException, AuthorizationException
from app.models.emergency import EmergencyRequest
from app.repositories.emergency import emergency_repository
from app.repositories.hospital import hospital_repository
from app.repositories.patient import patient_repository
from app.schemas.patient_api import EmergencyLocation

logger = logging.getLogger(__name__)

class EmergencyService: 
    async def trigger_panic(
        self, db: AsyncSession, patient_id: uuid.UUID, location_in: EmergencyLocation
    ) -> EmergencyRequest:
        """
        Registers an active emergency request, maps coordinates to an available hospital, 
        and mocks dispatching an ambulance unit.
        """
        patient = await patient_repository.get(db, patient_id)
        if not patient:
            raise EntityNotFoundException("Patient", str(patient_id))

        # Query available emergency-capable hospitals
        hospitals = await hospital_repository.get_available_emergency(db)
        
        hospital_id = None
        hospital_name = "MedBridge General Hospital"  # Mock default fallback
        
        if hospitals:
            # Route to the first available hospital 
            chosen_hospital = hospitals[0]
            hospital_id = chosen_hospital.id
            hospital_name = chosen_hospital.name
            logger.info(f"Routed emergency trigger to hospital: {hospital_name}")
        else:
            logger.warning("No hospital with available capacity was found. Using default mock hospital routing.")

        # Instantiate EmergencyRequest
        loc_dict = location_in.model_dump() if hasattr(location_in, "model_dump") else location_in
        req = EmergencyRequest(
            patient_id=patient_id,
            patient_name=f"{patient.first_name} {patient.last_name}",
            patient_phone=patient.phone,
            location=loc_dict,
            hospital_id=hospital_id,
            hospital_name=hospital_name,
            ambulance_dispatched=True,
            ambulance_id="AMB-101",
            status="dispatched",
            eta=12  # Mock ETA minutes
        )
        db.add(req)
        await db.flush()
        
        logger.info(f"Ambulance AMB-101 dispatched to {loc_dict.get('address', 'Current Location')} for Patient {patient_id}. ETA: 12 minutes.")
        return req

    async def track_emergency(
        self, db: AsyncSession, patient_id: uuid.UUID, emergency_id: uuid.UUID
    ) -> EmergencyRequest:
        """
        Tracks dispatch progress and ETA for an active emergency request.
        """
        req = await emergency_repository.get(db, emergency_id)
        if not req:
            raise EntityNotFoundException("EmergencyRequest", str(emergency_id))

        # Verify tracking permissions
        if req.patient_id != patient_id:
            raise AuthorizationException("You are not authorized to track this emergency request.")

        return req

emergency_service = EmergencyService()
