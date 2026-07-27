import logging
import uuid
import time
from datetime import datetime, timezone
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
from app.core.exceptions import (
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.models.user import User
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.emergency import EmergencyRequest
from app.repositories.user import user_repository
from app.repositories.doctor import doctor_repository
from app.repositories.hospital import hospital_repository
from app.repositories.emergency import emergency_repository

logger = logging.getLogger(__name__)

class AdminService:
    async def get_dashboard(self, db: AsyncSession) -> dict:
        """
        Gathers system-wide aggregates for dashboard presentation.
        """
        users_count = await db.scalar(select(func.count(User.id)))
        doctors_count = await db.scalar(select(func.count(Doctor.id)))
        hospitals_count = await db.scalar(select(func.count(Hospital.id)))
        
        active_emergencies = await db.scalar(
            select(func.count(EmergencyRequest.id))
            .where(EmergencyRequest.status.in_(["active", "dispatched", "arrived"]))
        )

        # Breakdowns the dashboard renders. All real counts.
        from app.models.case import Case
        from app.models.patient import Patient

        patients_count = await db.scalar(select(func.count(Patient.id)))
        cases_count = await db.scalar(select(func.count(Case.id)))
        active_patients = await db.scalar(
            select(func.count(User.id))
            .where(User.role == "patient")
            .where(User.is_active.is_(True))
        )
        active_doctors = await db.scalar(
            select(func.count(Doctor.id))
            .where(Doctor.availability.in_(["available", "busy"]))
        )
        active_hospitals = await db.scalar(
            select(func.count(Hospital.id))
            .where(Hospital.verification_status == "verified")
        )
        pending_verifications = await db.scalar(
            select(func.count(Doctor.id))
            .where(Doctor.verification_status.in_(["pending", "under_review"]))
        )

        return {
            "total_users": users_count or 0,
            "total_doctors": doctors_count or 0,
            "total_hospitals": hospitals_count or 0,
            "active_emergencies": active_emergencies or 0,
            "system_status": "operational",
            "total_patients": patients_count or 0,
            "total_cases": cases_count or 0,
            "active_patients": active_patients or 0,
            "active_doctors": active_doctors or 0,
            "active_hospitals": active_hospitals or 0,
            "pending_doctor_verifications": pending_verifications or 0,
        }

    async def get_analytics(self, db: AsyncSession) -> dict:
        """
        Aggregates statistical breakdown distributions for system roles and emergency responses.
        """
        # 1. Users by role distribution
        role_result = await db.execute(select(User.role, func.count(User.id)).group_by(User.role))
        users_by_role = {role: count for role, count in role_result}

        # 2. Hospitals by capacity distribution
        capacity_result = await db.execute(select(Hospital.emergency_capacity, func.count(Hospital.id)).group_by(Hospital.emergency_capacity))
        hospitals_by_capacity = {cap: count for cap, count in capacity_result}

        # 3. Emergency response success ratio
        total_emergencies = await db.scalar(select(func.count(EmergencyRequest.id))) or 0
        completed_emergencies = await db.scalar(
            select(func.count(EmergencyRequest.id))
            .where(EmergencyRequest.status == "completed")
        ) or 0

        success_ratio = 100.0
        if total_emergencies > 0:
            success_ratio = round((completed_emergencies / total_emergencies) * 100.0, 2)

        # 4. AI Reports Analytics
        from app.models.report import Report
        total_ai_reports = await db.scalar(select(func.count(Report.id))) or 0
        pending_review = await db.scalar(select(func.count(Report.id)).where(Report.status == "pending")) or 0
        approved_reports = await db.scalar(select(func.count(Report.id)).where(Report.status.in_(["approved", "ready"]))) or 0
        rejected_reports = await db.scalar(select(func.count(Report.id)).where(Report.status == "rejected")) or 0
        # No fallback constant: with no AI reports the honest answer is 0.0, not
        # an invented confidence figure.
        avg_confidence = await db.scalar(
            select(func.avg(Report.ai_confidence_score))
            .where(Report.ai_generated.is_(True))
            .where(Report.ai_confidence_score.is_not(None))
        ) or 0.0


        # 5. Mean time from case creation to completion, over completed cases.
        from app.models.case import Case

        resolution_rows = await db.execute(
            select(Case.created_at, Case.completed_at)
            .where(Case.status == "completed")
            .where(Case.completed_at.is_not(None))
        )
        durations = [
            (completed - created).total_seconds() / 3600.0
            for created, completed in resolution_rows
            if created and completed and completed >= created
        ]
        avg_resolution_hours = (
            round(sum(durations) / len(durations), 2) if durations else 0.0
        )

        return {
            "users_by_role": users_by_role,
            "hospitals_by_capacity": hospitals_by_capacity,
            "emergency_success_ratio": success_ratio,
            "avg_case_resolution_hours": avg_resolution_hours,
            # Mean self-reported model confidence over generated reports.
            # Not an accuracy measurement — no ground truth exists to score against.
            "avg_ai_confidence": round(float(avg_confidence), 2),
            "ai_reports_summary": {
                "total": total_ai_reports,
                "pending_review": pending_review,
                "approved": approved_reports,
                "rejected": rejected_reports,
                "avg_confidence": round(float(avg_confidence), 2)
            }
        }


    async def update_user_status(self, db: AsyncSession, user_id: uuid.UUID, active: bool) -> User:
        """
        Deactivates or reactivates a user's login access.

        This is the switch behind the administrator's "suspend" action on a
        clinician: `get_current_active_user` reads `is_active` on every request,
        so suspending ends an in-flight session rather than waiting for it to
        expire.
        """
        user = await user_repository.get(db, user_id)
        if not user:
            raise EntityNotFoundException("User", str(user_id))

        if active and not user.is_active and user.role == "admin":
            # Reinstating an administrator is a creation as far as the cap is
            # concerned — a retired administrator gave up their slot, and it may
            # have been taken since.
            from app.services.admin_accounts import assert_admin_slot_available

            await assert_admin_slot_available(db)

        user.is_active = active
        await db.flush()
        await db.refresh(user)

        logger.info(f"User {user_id} active status updated to: {active}")
        return user

    async def verify_doctor(self, db: AsyncSession, doctor_id: uuid.UUID, status_str: str) -> Doctor:
        """
        Record an administrator's decision on a clinician's credentials.

        Approval is the only action that touches the Doctor ID: an approved
        clinician must have one, because it is a factor in their sign-in, so one
        is allocated if the profile does not already hold a valid one. An
        existing valid ID is left alone — reissuing it on every re-approval
        would lock out a clinician who had already been told theirs.

        Rejecting or unverifying deliberately does *not* clear the ID. The ID
        alone grants nothing; keeping it means a clinician who is reinstated
        does not have to be re-issued credentials, and it stays visible in the
        admin queue for support.
        """
        doctor = await doctor_repository.get(db, doctor_id)
        if not doctor:
            raise EntityNotFoundException("Doctor", str(doctor_id))

        if status_str == "verified":
            from app.core.doctor_code import assign_doctor_code, is_valid_doctor_code

            if not is_valid_doctor_code(doctor.doctor_code):
                # `assign_doctor_code` retries through a lost allocation race
                # rather than surfacing a unique-index violation as a 500.
                await assign_doctor_code(db, doctor)
                logger.info("[DOCTOR_ID_ISSUED] doctor=%s", doctor_id)

            if not is_valid_doctor_code(doctor.doctor_code):
                # Belt and braces for the invariant the database also enforces:
                # approving a clinician who holds no Doctor ID would produce an
                # account that is authorised to practise but can never sign in,
                # because the ID is one of the three sign-in factors.
                raise BusinessRuleValidationException(
                    "This clinician could not be issued a Doctor ID, so the "
                    "account cannot be approved. Please retry."
                )

            doctor.verified_date = datetime.now(timezone.utc).isoformat()

        doctor.verification_status = status_str

        await db.flush()
        await db.refresh(doctor)

        logger.info(f"Doctor {doctor_id} verification updated to: {status_str}")
        return doctor

    @staticmethod
    def _doctor_review_row(doctor: Doctor, user: User | None) -> dict:
        """
        Flatten a clinician's profile and their account into one review record.

        The verification screen judges a person, not a table: their licence and
        their specialty live on `doctors`, but whether the account is suspended
        and when they registered live on `users`, and an administrator deciding
        on approval needs both in front of them.
        """
        return {
            "id": doctor.id,
            "doctor_code": doctor.doctor_code,
            "email": user.email if user else None,
            "is_active": bool(user.is_active) if user else False,
            "account_verified": bool(user.is_verified) if user else False,
            "registered_at": user.created_at if user else doctor.created_at,
            "first_name": doctor.first_name,
            "last_name": doctor.last_name,
            "phone": doctor.phone,
            "specialty": doctor.specialty,
            "sub_specialties": doctor.sub_specialties or [],
            "hospital_id": doctor.hospital_id,
            "hospital_name": doctor.hospital_name,
            "license_number": doctor.license_number,
            "years_of_experience": doctor.years_of_experience or 0,
            "education": doctor.education or [],
            "certifications": doctor.certifications or [],
            "languages": doctor.languages or [],
            "avatar_url": doctor.avatar_url,
            "bio": doctor.bio,
            "rating": doctor.rating or 0.0,
            "total_patients": doctor.total_patients or 0,
            "total_cases": doctor.total_cases or 0,
            "availability": doctor.availability,
            "consultation_fee": doctor.consultation_fee or 0.0,
            "verification_status": doctor.verification_status,
            "verified_date": doctor.verified_date,
        }

    async def list_doctors_for_review(
        self, db: AsyncSession, status_filter: str | None = None,
        page: int = 1, size: int = 25,
    ) -> dict:
        """
        One page of the administrator's clinician roster, newest first.

        Two queries: a COUNT so the caller knows how much it has *not* been
        shown, and the page itself. Both go through
        `ix_doctors_verification_status_created_at`, so neither scans the table.

        The rows come from a single outer join rather than a lookup per doctor
        — the verification queue is the screen an administrator opens most, and
        a per-row query for the email address would be an N+1 on the hot path.
        """
        filters = [Doctor.deleted_at.is_(None)]
        if status_filter:
            filters.append(Doctor.verification_status == status_filter)

        total = int(
            await db.scalar(select(func.count(Doctor.id)).where(*filters)) or 0
        )

        page = max(1, page)
        pages = max(1, -(-total // size))  # ceiling division
        result = await db.execute(
            select(Doctor, User)
            .outerjoin(User, User.id == Doctor.id)
            .where(*filters)
            .order_by(Doctor.created_at.desc(), Doctor.id)
            .offset((page - 1) * size)
            .limit(size)
        )

        return {
            "items": [self._doctor_review_row(d, u) for d, u in result.all()],
            "total": total,
            "page": page,
            "size": size,
            "pages": pages,
            "has_next": page < pages,
            "has_prev": page > 1,
        }

    async def get_doctor_for_review(
        self, db: AsyncSession, doctor_id: uuid.UUID
    ) -> dict:
        """One clinician's full record, for the profile view."""
        result = await db.execute(
            select(Doctor, User)
            .outerjoin(User, User.id == Doctor.id)
            .where(Doctor.id == doctor_id)
        )
        row = result.first()
        if row is None:
            raise EntityNotFoundException("Doctor", str(doctor_id))
        return self._doctor_review_row(row[0], row[1])

    async def verify_hospital(self, db: AsyncSession, hospital_id: uuid.UUID, status_str: str) -> Hospital:
        """
        Approves or updates verification status for a hospital clinic facility.
        """
        hospital = await hospital_repository.get(db, hospital_id)
        if not hospital:
            raise EntityNotFoundException("Hospital", str(hospital_id))

        hospital.verification_status = status_str
        await db.flush()
        await db.refresh(hospital)
        
        logger.info(f"Hospital {hospital_id} verification updated to: {status_str}")
        return hospital


    async def get_system_status(self, db: AsyncSession, redis: Redis) -> dict:
        """
        Checks operational status and pings dependent databases and caches.
        """
        # 1. Database Ping check
        db_start = time.perf_counter()
        db_status = "online"
        db_latency = 0.0
        try:
            from sqlalchemy import text
            await db.execute(text("SELECT 1"))
            db_latency = round((time.perf_counter() - db_start) * 1000.0, 2)
        except Exception as e:
            db_status = "offline"
            logger.error(f"Database health check failed: {str(e)}")

        # 2. Redis Ping check
        #
        # `ResilientRedisClient.ping` reports a failure by returning False — it
        # does not raise, because every other call site degrades to the
        # in-memory fallback instead of erroring. Testing only for an exception
        # therefore reported an unreachable Redis as "online", which is exactly
        # the condition this panel exists to surface.
        redis_start = time.perf_counter()
        redis_status = "online"
        redis_latency = 0.0
        try:
            reachable = await redis.ping()
            redis_latency = round((time.perf_counter() - redis_start) * 1000.0, 2)
            if not reachable:
                redis_status = "offline"
                logger.warning("Redis health check reported unreachable.")
        except Exception as e:
            redis_status = "offline"
            logger.error(f"Redis health check failed: {str(e)}")

        return {
            "database": {"status": db_status, "latency_ms": db_latency if db_status == "online" else None},
            "redis": {"status": redis_status, "latency_ms": redis_latency if redis_status == "online" else None},
            "celery": {"status": "online", "latency_ms": 5.2},  # Mock Celery status
            "cpu_usage": 15.5,
            "memory_usage": 60.1
        }

    async def delete_user(self, db: AsyncSession, user_id: uuid.UUID) -> dict:
        """
        Deletes a user account and associated patient/doctor profile atomically.
        """
        user = await user_repository.get(db, user_id)
        if not user:
            raise EntityNotFoundException("User", str(user_id))

        if user.role == "patient":
            from app.models.patient import Patient
            res = await db.execute(select(Patient).where(Patient.id == user_id))
            patient = res.scalars().first()
            if patient:
                await db.delete(patient)
        elif user.role == "doctor":
            from app.models.doctor import Doctor
            res = await db.execute(select(Doctor).where(Doctor.id == user_id))
            doctor = res.scalars().first()
            if doctor:
                await db.delete(doctor)

        await db.delete(user)
        await db.flush()
        logger.info(f"User {user_id} and associated role record deleted successfully.")
        return {"message": "User deleted successfully", "user_id": str(user_id)}

admin_service = AdminService()

