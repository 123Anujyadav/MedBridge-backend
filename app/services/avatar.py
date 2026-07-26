"""
Profile photo management for patients and doctors.

One service for both roles: the storage rules, validation and propagation are
identical, only the profile table differs. The row written is always the
authenticated caller's own — no endpoint accepts a target user id — so a user
cannot replace or delete someone else's photo.
"""

from __future__ import annotations

import logging
import uuid
from typing import Union

from fastapi import UploadFile
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.avatar import (
    delete_avatar_files,
    process_avatar,
    read_and_validate_avatar,
    store_avatar,
)
from app.core.exceptions import EntityNotFoundException
from app.models.appointment import Appointment
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.user import User

logger = logging.getLogger(__name__)

Profile = Union[Patient, Doctor]


class AvatarService:
    """Upload, replace and removal of profile photos."""

    async def _load_profile(self, db: AsyncSession, user: User) -> Profile:
        """
        Fetch the caller's own profile row.

        Keyed on the authenticated user id, never on a request parameter — this
        is what makes cross-user overwriting impossible rather than merely
        checked.
        """
        if user.role == "patient":
            profile = await db.get(Patient, user.id)
            if profile is None:
                raise EntityNotFoundException("Patient", str(user.id))
            return profile

        if user.role == "doctor":
            profile = await db.get(Doctor, user.id)
            if profile is None:
                raise EntityNotFoundException("Doctor", str(user.id))
            return profile

        # Admins have no clinical profile row to attach a photo to.
        raise EntityNotFoundException("Profile", str(user.id))

    async def set_avatar(
        self, db: AsyncSession, user: User, file: UploadFile
    ) -> Profile:
        """
        Validate, process and store a new profile photo, replacing any existing one.

        The previous file is removed only after the new URL is recorded, so a
        failure part-way through leaves the old photo intact rather than a
        profile pointing at nothing.
        """
        profile = await self._load_profile(db, user)

        content = await read_and_validate_avatar(file)
        avatar_bytes, thumbnail_bytes = process_avatar(content)
        new_url = store_avatar(avatar_bytes, thumbnail_bytes)

        previous_url = profile.avatar_url
        profile.avatar_url = new_url
        await db.flush()

        await self._propagate(db, user, new_url)
        await db.flush()
        await db.refresh(profile)

        if previous_url and previous_url != new_url:
            delete_avatar_files(previous_url)

        logger.info(
            "[AVATAR_UPDATED] user=%s role=%s bytes=%d",
            user.id, user.role, len(avatar_bytes),
        )
        return profile

    async def remove_avatar(self, db: AsyncSession, user: User) -> Profile:
        """
        Clear the caller's profile photo and delete the stored files.

        Removing a photo that was never set is a no-op rather than an error, so
        the control behaves the same however many times it is pressed.
        """
        profile = await self._load_profile(db, user)
        previous_url = profile.avatar_url

        profile.avatar_url = None
        await db.flush()

        await self._propagate(db, user, None)
        await db.flush()
        await db.refresh(profile)

        if previous_url:
            delete_avatar_files(previous_url)

        logger.info("[AVATAR_REMOVED] user=%s role=%s", user.id, user.role)
        return profile

    async def _propagate(
        self, db: AsyncSession, user: User, avatar_url: str | None
    ) -> None:
        """
        Update the copies of this avatar held on other rows.

        `cases` and `appointments` each carry a denormalised avatar column that
        is written when the row is created. Without this step a changed photo
        would appear on the profile page while the doctor's case queue and the
        appointment lists kept showing the old one indefinitely.
        """
        user_id: uuid.UUID = user.id

        if user.role == "patient":
            await db.execute(
                update(Case)
                .where(Case.patient_id == user_id)
                .values(patient_avatar_url=avatar_url)
            )
            await db.execute(
                update(Appointment)
                .where(Appointment.patient_id == user_id)
                .values(patient_avatar_url=avatar_url)
            )
        elif user.role == "doctor":
            await db.execute(
                update(Appointment)
                .where(Appointment.doctor_id == user_id)
                .values(doctor_avatar_url=avatar_url)
            )


avatar_service = AvatarService()
