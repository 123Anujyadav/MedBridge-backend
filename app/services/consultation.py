import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import EntityNotFoundException, AuthorizationException
from app.models.case import Case
from app.models.prescription import Prescription, Medication
from app.models.report import Report
from app.repositories.case import case_repository
from app.repositories.doctor import doctor_repository
from app.repositories.patient import patient_repository
from app.repositories.prescription import prescription_repository
from app.repositories.report import report_repository
from app.schemas.doctor_api import (
    CreatePrescriptionRequest,
    CreateReportRequest,
    DiagnoseCaseRequest,
    UpdateCaseNotesRequest,
)

logger = logging.getLogger(__name__)

class ConsultationService:
    async def get_case(self, db: AsyncSession, case_id: uuid.UUID) -> Case:
        """
        Retrieves a consultation Case.
        """
        case = await case_repository.get(db, case_id)
        if not case:
            raise EntityNotFoundException("Case", str(case_id))
        return case

    async def update_case_notes(
        self, db: AsyncSession, doctor_id: uuid.UUID, case_id: uuid.UUID, notes_in: UpdateCaseNotesRequest
    ) -> Case:
        """
        Appends or updates progressive clinical notes on an assigned case.
        """
        case = await self.get_case(db, case_id)
        
        # Verify ownership
        if case.doctor_id != doctor_id:
            raise AuthorizationException("You are not authorized to edit notes on this case.")

        case.notes = notes_in.notes
        await db.flush()
        return case

    async def diagnose_case(
        self, db: AsyncSession, doctor_id: uuid.UUID, case_id: uuid.UUID, diagnose_in: DiagnoseCaseRequest
    ) -> Case:
        """
        Records the final consultation diagnosis and marks the case as completed.
        """
        case = await self.get_case(db, case_id)

        # Verify ownership
        if case.doctor_id != doctor_id:
            raise AuthorizationException("You are not authorized to diagnose this case.")

        case.notes = diagnose_in.notes
        case.status = "completed"
        await db.flush()
        
        logger.info(f"Case {case_id} successfully diagnosed and completed by doctor {doctor_id}.")
        return case

    async def write_prescription(
        self, db: AsyncSession, doctor_id: uuid.UUID, req: CreatePrescriptionRequest
    ) -> Prescription:
        """
        Registers a new Prescription with itemized medications.
        Transitions the parent consultation case status to 'prescribed'.
        """
        doctor = await doctor_repository.get(db, doctor_id)
        if not doctor:
            raise EntityNotFoundException("Doctor", str(doctor_id))

        case = await self.get_case(db, req.case_id)
        if case.doctor_id != doctor_id:
            raise AuthorizationException("You are not authorized to write prescriptions for this case.")

        # The case is the authority on who the patient is. Taking `patient_id`
        # from the request while taking `patient_name` from the case allowed a
        # prescription to be filed under one patient's id carrying another
        # patient's name — and to surface on the wrong patient's record.
        if req.patient_id != case.patient_id:
            raise AuthorizationException(
                "The prescription patient does not match the patient on this case."
            )

        # Create Prescription record.
        #
        # The prescriber's details are snapshotted rather than read live through
        # the `doctor` relationship. A prescription is the legal record of who
        # ordered what on whose authority; if these were resolved at read time,
        # a clinician moving hospital or renewing a licence would silently
        # rewrite every prescription they had ever signed.
        qualification = ", ".join(
            str(entry) for entry in (doctor.education or []) if entry
        )
        rx = Prescription(
            case_id=case.id,
            patient_id=case.patient_id,
            patient_name=case.patient_name,
            doctor_id=doctor_id,
            doctor_name=f"{doctor.first_name} {doctor.last_name}",
            diagnosis=req.diagnosis,
            notes=req.notes,
            status="active",
            follow_up_date=req.follow_up_date,
            attachment_url=req.attachment_url,
            doctor_specialty=doctor.specialty,
            doctor_qualification=qualification[:255] or None,
            doctor_hospital=doctor.hospital_name,
            doctor_registration_number=doctor.license_number,
            doctor_experience_years=doctor.years_of_experience,
            doctor_avatar_url=doctor.avatar_url,
            consultation_date=datetime.now(timezone.utc),
            signed_at=datetime.now(timezone.utc),
        )
        db.add(rx)
        await db.flush()  # Populate rx.id

        # Add medication lines
        for item in req.medications:
            # Calculate total doses based on scheduled times count * duration estimation
            # Let's count times per day * days
            times_per_day = len(item.scheduled_times) if item.scheduled_times else 1
            # duration parse: e.g. "7 days" -> parse number 7, fallback to 7
            days = 7
            try:
                days = int(item.duration.split()[0])
            except Exception:
                pass
            total_doses = times_per_day * days

            med = Medication(
                prescription_id=rx.id,
                name=item.name,
                generic_name=item.generic_name,
                brand_name=item.brand_name,
                strength=item.strength,
                dosage=item.dosage,
                frequency=item.frequency,
                duration=item.duration,
                food_instruction=item.food_instruction,
                route=item.route,
                # Falls back to the computed dose count so a pharmacy always has
                # a number to dispense against, even when the clinician did not
                # state one explicitly.
                quantity=item.quantity if item.quantity is not None else total_doses,
                special_instructions=item.special_instructions,
                status="active",
                scheduled_times=item.scheduled_times,
                taken_doses=0,
                total_doses=total_doses,
                start_date=item.start_date,
                end_date=item.end_date,
                side_effects=item.side_effects,
                interactions=item.interactions
            )
            db.add(med)

        # Update case status
        previous_status = case.status
        case.status = "prescribed"
        await db.flush()

        from app.models.user import User
        from app.services.case_timeline import case_timeline_service

        await case_timeline_service.safe_record(
            db, event_type="prescription.created",
            description=f"Prescription issued for {req.diagnosis}.",
            actor=await db.get(User, doctor_id), actor_type="doctor",
            case_id=case.id, patient_id=case.patient_id,
            resource="Prescription", resource_id=str(rx.id),
            field="status", previous=previous_status, new=case.status,
        )

        logger.info(f"Prescription written successfully for Case {req.case_id} by Doctor {doctor_id}.")
        return await prescription_repository.get(db, rx.id)


    async def write_report(
        self, db: AsyncSession, doctor_id: uuid.UUID, req: CreateReportRequest
    ) -> Report:
        """
        Authors a new clinical Report (e.g. imaging result, lab report) for a patient.

        The caller must already have a case with the named patient. Without this
        check any authenticated doctor could file a report into any patient's
        permanent record simply by supplying their id.
        """
        from sqlalchemy import select

        linked = await db.scalar(
            select(Case.id)
            .where(Case.doctor_id == doctor_id)
            .where(Case.patient_id == req.patient_id)
            .limit(1)
        )
        if linked is None:
            raise AuthorizationException(
                "You are not authorised to author a report for this patient."
            )

        # A supplied case must belong to the named patient. Attaching a result
        # to someone else's case would place it on the wrong timeline.
        linked_case = None
        if req.case_id is not None:
            linked_case = await case_repository.get(db, req.case_id)
            if linked_case is None or linked_case.patient_id != req.patient_id:
                raise AuthorizationException(
                    "The supplied case does not belong to this patient."
                )

        doctor = await doctor_repository.get(db, doctor_id)
        doctor_name = f"{doctor.first_name} {doctor.last_name}" if doctor else "Unknown Doctor"

        report = Report(
            patient_id=req.patient_id,
            case_id=req.case_id,
            patient_name=req.patient_name,
            type=req.type,
            title=req.title,
            summary=req.summary,
            content=req.content,
            doctor_name=doctor_name,
            hospital_name=req.hospital_name,
            date=req.date,
            status="ready",
            file_url=req.file_url,
            file_size=req.file_size,
            ai_generated=req.ai_generated,
            ai_confidence_score=req.ai_confidence_score,
            tags=req.tags,
            vitals=req.vitals
        )
        db.add(report)
        await db.flush()

        await self._notify_lab_result(db, report=report, case=linked_case,
                                      author_id=doctor_id)

        logger.info(f"Medical report '{req.title}' successfully authored for patient {req.patient_id}.")
        return report

    # Report types that represent a diagnostic result arriving on a case.
    DIAGNOSTIC_TYPES = frozenset({
        "lab_result", "lab_report", "blood_test", "imaging", "scan",
        "x_ray", "mri", "ct_scan", "ultrasound", "pathology",
    })

    async def _notify_lab_result(
        self, db: AsyncSession, *, report: Report, case, author_id: uuid.UUID
    ) -> None:
        """
        Tell the treating clinician a diagnostic result has landed on their case.

        Three conditions, all deliberate:

        * Only diagnostic report types. A discharge summary is not a lab result.
        * Only when the report is linked to a case — an unlinked document has no
          treating clinician to notify, and guessing one would misroute PHI.
        * Never back to the author. A clinician who just filed a result does not
          need to be told it exists; self-notification is the fastest way to
          make people stop reading notifications.
        """
        if report.type not in self.DIAGNOSTIC_TYPES or case is None:
            return
        if case.doctor_id is None or case.doctor_id == author_id:
            return

        from app.services.notifications import notification_service

        await notification_service.safe_notify(
            db,
            user_id=case.doctor_id,
            category="report",
            type="lab_result_uploaded",
            title="Lab Results Uploaded",
            message=(
                f"{report.title} attached to {case.patient_name}'s case. "
                f"{(report.summary or '')[:160]}"
            ).strip(),
            priority="high" if case.urgency_level in ("high", "critical") else "medium",
            case_id=case.id,
            patient_id=report.patient_id,
            patient_name=case.patient_name,
            action_url=f"/doctor/ai-reports?report={report.id}",
            action_label="View Report",
            group_key="lab_result_uploaded",
            dedupe_key=f"lab_result_uploaded:{report.id}",
        )

    async def complete_consultation(
        self,
        db: AsyncSession,
        doctor_id: uuid.UUID,
        case_id: uuid.UUID,
        diagnosis: str,
        clinical_notes: str,
        medications: Optional[List[dict]] = None,
        recommended_tests: Optional[List[str]] = None,
        follow_up_date: Optional[str] = None,
        doctor_remarks: Optional[str] = None
    ) -> Report:
        """
        Completes a consultation case, generates an official PDF Medical Report using ReportLab,
        saves the report in the database, and broadcasts real-time WebSocket updates across role dashboards.
        """
        from app.services.report_generator import report_generator
        from app.core.websocket import websocket_manager
        from sqlalchemy import select
        from app.models.appointment import Appointment

        case = await self.get_case(db, case_id)
        if case.doctor_id != doctor_id:
            raise AuthorizationException("You are not authorized to complete this consultation.")

        doctor = await doctor_repository.get(db, doctor_id)
        doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}" if doctor else "Doctor"

        patient = await patient_repository.get(db, case.patient_id)
        patient_name = f"{patient.first_name} {patient.last_name}" if patient else case.patient_name

        # Update case
        case.status = "completed"
        case.diagnosis = diagnosis
        case.notes = clinical_notes
        await db.flush()

        # Update related appointments to completed
        appts = await db.execute(select(Appointment).where(Appointment.case_id == case_id))
        for appt in appts.scalars().all():
            appt.status = "completed"
        await db.flush()

        # Generate structured PDF Medical Report
        pdf_meta = report_generator.generate_pdf(
            patient_name=patient_name,
            patient_id=str(case.patient_id),
            doctor_name=doctor_name,
            doctor_id=str(doctor_id),
            symptoms=case.symptom_summary or "General Medical Consultation",
            diagnosis=diagnosis,
            clinical_notes=clinical_notes,
            medications=medications or [],
            recommended_tests=recommended_tests,
            follow_up_date=follow_up_date,
            doctor_remarks=doctor_remarks,
            hospital_name=doctor.hospital_name if doctor else "MedBridge Medical Center"
        )

        # Save Report in Database
        import datetime
        now_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        
        report = Report(
            patient_id=case.patient_id,
            # Without this the report carried a NULL case_id, and every consumer
            # that needs case context had to guess which case it belonged to.
            case_id=case.id,
            patient_name=patient_name,
            type="medical_report",
            title=f"Medical Consultation Report - {diagnosis[:40]}",
            summary=f"Primary Diagnosis: {diagnosis}. Prescribed {len(medications or [])} medication(s). Notes: {clinical_notes[:100]}...",
            content=f"Consultation completed by {doctor_name}. Diagnosis: {diagnosis}. Notes: {clinical_notes}",
            doctor_name=doctor_name,
            hospital_name=doctor.hospital_name if doctor else "MedBridge Medical Center",
            date=now_date,
            status="ready",
            file_url=pdf_meta["file_url"],
            file_size=pdf_meta["file_size"],
            ai_generated=False,
            tags=["consultation", "official_report", "pdf"],
            vitals={}
        )
        db.add(report)
        await db.flush()

        # Broadcast real-time WebSocket event to all role dashboards
        ws_msg = {
            "type": "CONSULTATION_COMPLETED",
            "event": "consultation_completed",
            "case_id": str(case_id),
            "patient_id": str(case.patient_id),
            "doctor_id": str(doctor_id),
            "report_id": str(report.id),
            "file_url": pdf_meta["file_url"]
        }
        await websocket_manager.broadcast(ws_msg)

        logger.info(f"Consultation {case_id} completed and official PDF report created: {pdf_meta['file_url']}")
        return report

consultation_service = ConsultationService()

