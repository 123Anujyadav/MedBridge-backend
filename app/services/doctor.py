import logging
import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import EntityNotFoundException
from app.models.doctor import Doctor
from app.models.appointment import Appointment
from app.models.prescription import Prescription, Medication
from app.repositories.doctor import doctor_repository
from app.repositories.case import case_repository
from app.schemas.doctor_api import UpdateAvailabilityRequest

logger = logging.getLogger(__name__)

class DoctorService:
    async def get_profile(self, db: AsyncSession, doctor_id: uuid.UUID) -> Doctor:
        """
        Retrieves the Doctor profile record.
        """
        doctor = await doctor_repository.get(db, doctor_id)
        if not doctor:
            raise EntityNotFoundException("Doctor", str(doctor_id))
        return doctor

    async def update_availability(
        self, db: AsyncSession, doctor_id: uuid.UUID, update_in: UpdateAvailabilityRequest
    ) -> Doctor:
        """
        Updates scheduling availability settings for a doctor.
        """
        doctor = await self.get_profile(db, doctor_id)
        return await doctor_repository.update(db, db_obj=doctor, obj_in=update_in)

    async def get_dashboard(self, db: AsyncSession, doctor_id: uuid.UUID) -> dict:
        """
        Assembles complete doctor portal dashboard state.
        """
        doctor = await self.get_profile(db, doctor_id)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 1. Fetch unique patients assigned
        patients = await case_repository.get_patients_by_doctor(db, doctor_id)

        # 2. Fetch today's appointments
        appt_stmt = select(Appointment).where(
            and_(
                Appointment.doctor_id == doctor_id,
                Appointment.date == today_str
            )
        ).order_by(Appointment.time.asc())
        appt_result = await db.execute(appt_stmt)
        today_appts = list(appt_result.scalars().all())

        # 3. Calculate weekly consultations count
        one_week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        week_appt_stmt = select(Appointment).where(
            and_(
                Appointment.doctor_id == doctor_id,
                Appointment.date >= one_week_ago,
                Appointment.status == "completed"
            )
        )
        week_result = await db.execute(week_appt_stmt)
        completed_week_appts = list(week_result.scalars().all())

        # 4. Fetch pending cases assigned
        all_cases = await case_repository.get_by_doctor(db, doctor_id)
        pending_cases = [c for c in all_cases if c.status in ["intake", "ai_processing", "routed", "in_consultation"]]

        return {
            "doctor_id": doctor_id,
            "total_patients": len(patients),
            "total_consultations_week": len(completed_week_appts),
            "rating": doctor.rating,
            "today_appointments": today_appts,
            "pending_cases": pending_cases
        }

    async def get_analytics(self, db: AsyncSession, doctor_id: uuid.UUID) -> dict:
        """
        Calculates analytical distributions for the physician's patient base and prescriptions.
        """
        patients = await case_repository.get_patients_by_doctor(db, doctor_id)
        all_cases = await case_repository.get_by_doctor(db, doctor_id)

        # 1. Age distribution calculations
        under_18 = 0
        r_18_35 = 0
        r_36_50 = 0
        over_50 = 0
        
        current_year = datetime.now(timezone.utc).year
        for p in patients:
            age = 0
            if p.date_of_birth:
                try:
                    birth_year = int(p.date_of_birth.split("-")[0])
                    age = current_year - birth_year
                except Exception:
                    age = 30  # Default fallback if parsing fails
            else:
                age = 30

            if age < 18:
                under_18 += 1
            elif age <= 35:
                r_18_35 += 1
            elif age <= 50:
                r_36_50 += 1
            else:
                over_50 += 1

        age_dist = {
            "under_18": under_18,
            "18_35": r_18_35,
            "36_50": r_36_50,
            "over_50": over_50
        }

        # 2. Case status distribution calculations
        status_dist = {}
        for c in all_cases:
            status_dist[c.status] = status_dist.get(c.status, 0) + 1

        # 3. Adherence rate calculation
        meds_stmt = select(Medication).join(Prescription).where(
            and_(
                Prescription.doctor_id == doctor_id,
                Medication.total_doses > 0
            )
        )
        meds_result = await db.execute(meds_stmt)
        meds = list(meds_result.scalars().all())

        adherence_sum = 0.0
        adherence_count = 0
        for m in meds:
            adherence_sum += (m.taken_doses / m.total_doses) * 100.0
            adherence_count += 1

        adherence_rate = 100.0
        if adherence_count > 0:
            adherence_rate = round(adherence_sum / adherence_count, 1)

        # 4. Monthly case volume: intake vs resolved, from real case rows.
        #    Months with no activity are omitted rather than zero-filled, so a
        #    quiet month is not misread as a collapse in volume.
        trend_buckets: dict[str, dict[str, int]] = {}
        for case in all_cases:
            if not case.created_at:
                continue
            key = case.created_at.strftime("%Y-%m")
            bucket = trend_buckets.setdefault(key, {"cases": 0, "resolved": 0})
            bucket["cases"] += 1
            if case.status in ("completed", "report_generated", "archived"):
                bucket["resolved"] += 1

        case_trend = [
            {
                "month": datetime.strptime(key, "%Y-%m").strftime("%b"),
                "period": key,
                "cases": vals["cases"],
                "resolved": vals["resolved"],
            }
            for key, vals in sorted(trend_buckets.items())
        ]

        # 5. Case mix by specialty, for the distribution chart.
        specialty_counts: dict[str, int] = {}
        for case in all_cases:
            if case.specialty:
                specialty_counts[case.specialty] = (
                    specialty_counts.get(case.specialty, 0) + 1
                )
        specialty_distribution = [
            {"name": name, "value": count}
            for name, count in sorted(
                specialty_counts.items(), key=lambda kv: kv[1], reverse=True
            )
        ]

        return {
            "age_distribution": age_dist,
            "status_distribution": status_dist,
            "adherence_rate": adherence_rate,
            "case_trend": case_trend,
            "specialty_distribution": specialty_distribution,
        }

doctor_service = DoctorService()
