import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import EntityNotFoundException
from app.models.consent import ConsentRecord
from app.models.patient import Patient
from app.repositories.patient import patient_repository
from app.repositories.appointment import appointment_repository
from app.repositories.prescription import medication_repository
from app.repositories.report import report_repository
from app.repositories.notification import notification_repository
from app.schemas.patient import ConsentFlagsSchema, PatientUpdate

logger = logging.getLogger(__name__)

class PatientService:
    async def get_profile(self, db: AsyncSession, patient_id: uuid.UUID) -> Patient:
        """
        Retrieves the Patient record for a given ID.
        """
        patient = await patient_repository.get(db, patient_id)
        if not patient:
            raise EntityNotFoundException("Patient", str(patient_id))
        return patient

    async def update_profile(
        self, db: AsyncSession, patient_id: uuid.UUID, profile_in: PatientUpdate
    ) -> Patient:
        """
        Updates fields on a patient profile.
        """
        patient = await self.get_profile(db, patient_id)
        # Update using base repository update routine
        return await patient_repository.update(db, db_obj=patient, obj_in=profile_in)


    async def update_consent(
        self, db: AsyncSession, patient_id: uuid.UUID, consent_in: ConsentFlagsSchema
    ) -> Patient:
        """
        Updates consent flags and logs historical changes in the consent registry table.
        """
        patient = await self.get_profile(db, patient_id)
        
        # Save consent flags on profile
        consent_flags_dict = consent_in.model_dump()
        await patient_repository.update(db, db_obj=patient, obj_in={"consent_flags": consent_flags_dict})

        # Add tracking log for auditing purposes and if not shown then it terminates 
        log_detail = f"Consent updated: dataSharing={consent_in.dataSharing}, research={consent_in.researchParticipation}, emergency={consent_in.emergencyAccess}, ai={consent_in.aiProcessing}"
        audit_consent = ConsentRecord(
            patient_id=patient_id,
            patient_name=f"{patient.first_name} {patient.last_name}",
            consent_type="PRIVACY_SETTINGS",
            granted=True,
            granted_at=datetime.now(timezone.utc).isoformat(),
            version="2.0",
            details=log_detail
        )
        db.add(audit_consent)
        await db.flush()

        logger.info(f"Consent audit log recorded for patient {patient_id}")
        return patient

      # it should be async then it is async , else it will not work 
    async def get_dashboard(self, db: AsyncSession, patient_id: uuid.UUID) -> dict:
        """
        Assembles complete patient dashboard state.
        """
        patient = await self.get_profile(db, patient_id)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 1. Fetch upcoming appointments 
        all_appts = await appointment_repository.get_by_patient(db, patient_id)
        upcoming_appts = [
            a for a in all_appts 
            if a.status in ["scheduled", "confirmed", "in_progress"]
        ][:3]  # Limit to 3 upcoming 

        # 2. Fetch today's medications
        today_meds = await medication_repository.get_patient_meds_for_today(db, patient_id, today_str)

        # 3. Fetch recent reports
        all_reports = await report_repository.get_by_patient(db, patient_id)
        recent_reports = all_reports[:5]  # Limit to 5 recent

        # 4. Fetch unread notification counts
        unread_notifs = await notification_repository.get_unread_count(db, patient_id)

        return {
            "patient_id": patient_id,
            "health_score": patient.health_score,
            "upcoming_appointments": upcoming_appts,
            "today_medications": today_meds,
            "recent_reports": recent_reports,
            "unread_notifications_count": unread_notifs
        }

patient_service = PatientService()
