import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select, or_, and_

from app.core.upload import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
    sanitize_filename,
    scan_file_virus_hook,
)
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.user import User
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.appointment import Appointment
from app.models.prescription import Prescription
from app.models.case import Case
from app.schemas.shared_api import CalendarEventResponse, TimelineEventResponse, SearchResult

logger = logging.getLogger(__name__)

class SharedService:

    async def save_uploaded_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Validates and stores an uploaded clinical file, returning its metadata.

        Validation reuses the helpers in `app.core.upload` (MIME allowlist,
        extension allowlist, size ceiling, filename sanitisation, anti-virus
        hook). Those helpers already existed but nothing called them, so this
        endpoint previously accepted any file of any type or size — including
        executables. Enforcing here rather than in the controller means every
        caller of this service is covered.
        """
        # 1. MIME type allowlist
        if file.content_type not in ALLOWED_MIME_TYPES:
            logger.warning(
                f"Upload rejected: disallowed MIME type '{file.content_type}'"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Disallowed file type: {file.content_type}",
            )

        # 2. Extension allowlist (defends against a spoofed Content-Type)
        original_name = file.filename or "file"
        file_ext = os.path.splitext(original_name)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Upload rejected: disallowed extension '{file_ext}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Disallowed file extension: {file_ext}",
            )

        # 3. Size ceiling
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            logger.warning(
                f"Upload rejected: {len(content)} bytes exceeds "
                f"{MAX_FILE_SIZE_BYTES}"
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "File exceeds the maximum allowed size of "
                    f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                ),
            )

        # 4. Anti-virus hook
        if not scan_file_virus_hook(content):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File failed anti-virus inspection.",
            )

        upload_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
        )
        os.makedirs(upload_dir, exist_ok=True)

        # 5. Store under a generated name; the original is kept only as metadata,
        #    so a hostile filename can never influence the path on disk.
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        dest_path = os.path.join(upload_dir, unique_filename)

        # Defence in depth: confirm the resolved path stays inside uploads/.
        if not os.path.abspath(dest_path).startswith(upload_dir + os.sep):
            logger.error(f"Path traversal attempt blocked for: {original_name}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid destination path.",
            )

        with open(dest_path, "wb") as f:
            f.write(content)

        file_url = f"/uploads/{unique_filename}"
        logger.info(
            f"File stored: {unique_filename} "
            f"({len(content)} bytes, {file.content_type})"
        )

        return {
            "filename": sanitize_filename(original_name),
            "file_url": file_url,
            "content_type": file.content_type,
            "size_bytes": len(content),
        }

    async def search_entities(self, db: AsyncSession, q: str) -> List[SearchResult]:
        """
        Runs unified search queries across Doctor specialties and Hospital locations.
        """
        q_term = f"%{q}%"
        results = []

        # 1. Search Doctors 
        doctors_stmt = select(Doctor).where(
            or_(
                Doctor.first_name.ilike(q_term),
                Doctor.last_name.ilike(q_term),
                Doctor.specialty.ilike(q_term)
            )
        )
        doc_result = await db.execute(doctors_stmt)
        for doc in doc_result.scalars():
            results.append(
                SearchResult(
                    id=doc.id,
                    type="doctor",
                    name=f"Dr. {doc.first_name} {doc.last_name}",
                    details=doc.specialty,
                    city="Boston",  # Mock location
                    state="MA"
                )
            )

        # 2. Search Hospitals and append to results
        hospitals_stmt = select(Hospital).where(
            or_(
                Hospital.name.ilike(q_term),
                Hospital.city.ilike(q_term),
                Hospital.state.ilike(q_term)
            )
        )
        hosp_result = await db.execute(hospitals_stmt)
        for hosp in hosp_result.scalars():
            results.append(
                SearchResult(
                    id=hosp.id,
                    type="hospital",
                    name=hosp.name,
                    details=hosp.address,
                    city=hosp.city,
                    state=hosp.state
                )
            )

        return results


    async def get_calendar_events(self, db: AsyncSession, user: User) -> List[CalendarEventResponse]:
        """
        Aggregates scheduled appointments into a unified calendar schedule.
        """
        events = []
        stmt = select(Appointment)
        if user.role == "patient":
            stmt = stmt.where(Appointment.patient_id == user.id)
        elif user.role == "doctor":
            stmt = stmt.where(Appointment.doctor_id == user.id)
        else:
            # Admins view all appointments
            pass

        stmt = stmt.order_by(Appointment.date.desc(), Appointment.time.desc())
        result = await db.execute(stmt)

        for appt in result.scalars():
            events.append(
                CalendarEventResponse(
                    id=appt.id,
                    title=f"{appt.type.capitalize()} Appointment: {appt.reason}",
                    date=appt.date,
                    time=appt.time,
                    duration=appt.duration,
                    type="appointment",
                    status=appt.status,
                    description=f"Consultation scheduled at {appt.hospital_name}"
                )
            )

        return events

    # The case timeline now lives in `app.services.case_timeline`, which
    # merges recorded audit events with derived milestones. The thin
    # builder that used to sit here was replaced rather than kept beside
    # it, so there is one implementation of a case's history, not two.

    async def get_user_settings(self, redis: Redis, user_id: uuid.UUID) -> Dict[str, Any]:
        """
        Fetches a user's interface preferences from Redis, returning default preferences if none exist.
        """
        import json
        key = f"settings:{user_id}"
        val = await redis.get(key)
        if val:
            return json.loads(val)

        # Defaults
        return {
            "theme": "dark",
            "notifications_enabled": True,
            "email_notifications": True,
            "marketing_emails": False
        }

    async def update_user_settings(self, redis: Redis, user_id: uuid.UUID, settings: dict) -> Dict[str, Any]:
        """
        Saves updated notification and interface preferences for a user in Redis.
        """
        import json
        key = f"settings:{user_id}"
        current = await self.get_user_settings(redis, user_id)
        
        # Merge changes
        for k, v in settings.items():
            if v is not None:
                current[k] = v

        await redis.set(key, json.dumps(current))
        return current

shared_service = SharedService()
