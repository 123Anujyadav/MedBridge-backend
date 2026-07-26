import uuid
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import (
    BusinessRuleValidationException,
    EntityNotFoundException,
    AuthorizationException,
)
from app.models.appointment import Appointment
from app.repositories.appointment import appointment_repository
from app.repositories.doctor import doctor_repository
from app.repositories.patient import patient_repository
from app.schemas.patient_api import AppointmentCreateRequest

class AppointmentService:
    async def book_appointment(
        self, db: AsyncSession, patient_id: uuid.UUID, appt_in: AppointmentCreateRequest
    ) -> Appointment:
        """
        Validates slots and schedules an appointment for a patient.
        """
        # Verify doctor profile exists
        doctor = await doctor_repository.get(db, appt_in.doctor_id)
        if not doctor:
            raise EntityNotFoundException("Doctor", str(appt_in.doctor_id))

        # Verify patient profile exists
        patient = await patient_repository.get(db, patient_id)
        if not patient:
            raise EntityNotFoundException("Patient", str(patient_id))

        # Verify double-booking conflict
        has_conflict = await appointment_repository.check_conflict(
            db, appt_in.doctor_id, appt_in.date, appt_in.time
        )
        if has_conflict:
            raise BusinessRuleValidationException("The requested appointment slot is already booked.")

        # Create Appointment object
        appt = Appointment(
            patient_id=patient_id,
            doctor_id=appt_in.doctor_id,
            patient_name=f"{patient.first_name} {patient.last_name}",
            doctor_name=f"{doctor.first_name} {doctor.last_name}",
            specialty=appt_in.specialty,
            hospital_name=appt_in.hospital_name,
            date=appt_in.date,
            time=appt_in.time,
            duration=30,  # Default duration
            type=appt_in.type,
            status="scheduled",
            reason=appt_in.reason,
            notes=""
        )
        db.add(appt)

        # The check above can be overtaken by a concurrent booking, so the slot
        # is claimed under `uq_appointment_active_slot`. Writing inside a
        # savepoint keeps a rejected claim from poisoning the request's
        # transaction, and the caller still sees the same message the
        # check-then-insert path produces.
        try:
            async with db.begin_nested():
                await db.flush()
        except IntegrityError:
            raise BusinessRuleValidationException(
                "The requested appointment slot is already booked."
            )

        # Broadcast real-time event to role dashboards
        from app.core.websocket import websocket_manager
        await websocket_manager.broadcast({
            "type": "APPOINTMENT_CREATED",
            "appointment_id": str(appt.id),
            "patient_id": str(patient_id),
            "doctor_id": str(appt_in.doctor_id),
            "status": appt.status
        })

        # Notify the treating clinician. Targeted at the appointment's own
        # doctor_id — the broadcast above only refreshes caches and carries no
        # clinical detail, so it is not a substitute for this.
        from app.services.notifications import notification_service

        await notification_service.safe_notify(
            db,
            user_id=appt.doctor_id,
            category="appointment",
            type="appointment_scheduled",
            title="Appointment Scheduled",
            message=f"{appt.patient_name} booked {appt.date} at {appt.time} — {appt.reason}",
            priority="medium",
            patient_id=appt.patient_id,
            patient_name=appt.patient_name,
            case_id=appt.case_id,
            action_url=f"/doctor/schedule?appointment={appt.id}",
            action_label="Open Appointment",
            group_key="appointment_scheduled",
            dedupe_key=f"appointment_scheduled:{appt.id}",
        )

        return appt

    async def cancel_appointment(
        self, db: AsyncSession, patient_id: uuid.UUID, appt_id: uuid.UUID
    ) -> Appointment:
        """
        Cancels an upcoming appointment.
        """
        appt = await appointment_repository.get(db, appt_id)
        if not appt:
            raise EntityNotFoundException("Appointment", str(appt_id))

        # Ensure patient ownership of the appointment record
        if appt.patient_id != patient_id:
            raise AuthorizationException("You are not authorized to cancel this appointment.")

        appt.status = "cancelled"
        await db.flush()

        from app.services.notifications import notification_service

        await notification_service.safe_notify(
            db,
            user_id=appt.doctor_id,
            category="appointment",
            type="appointment_cancelled",
            title="Appointment Cancelled",
            message=f"{appt.patient_name} cancelled {appt.date} at {appt.time}.",
            priority="high",
            patient_id=appt.patient_id,
            patient_name=appt.patient_name,
            case_id=appt.case_id,
            action_url=f"/doctor/schedule?appointment={appt.id}",
            action_label="Open Appointment",
            group_key="appointment_cancelled",
            dedupe_key=f"appointment_cancelled:{appt.id}",
        )

        from app.core.websocket import websocket_manager
        await websocket_manager.broadcast({
            "type": "APPOINTMENT_STATUS_UPDATED",
            "appointment_id": str(appt.id),
            "patient_id": str(patient_id),
            "doctor_id": str(appt.doctor_id),
            "status": "cancelled"
        })

        return appt

appointment_service = AppointmentService()

