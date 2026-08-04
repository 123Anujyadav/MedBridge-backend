"""
The patient Emergency Profile: read, save, locate, delete.

Every method takes the patient's id from the authenticated caller and uses it as
the primary key. There is no method that accepts a target patient, so there is
no route by which one patient's request can reach another patient's profile —
the isolation is structural rather than a check that could be forgotten.

This module deliberately notifies nobody and dispatches nothing. It is the
record the SOS system will later read; the SOS system is not here.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EntityNotFoundException
from app.models.emergency_profile import EmergencyProfile
from app.repositories.emergency_profile import emergency_profile_repository
from app.repositories.patient import patient_repository
from app.schemas.emergency_profile import (
    EmergencyLocationUpdate,
    EmergencyProfileUpsert,
)

logger = logging.getLogger(__name__)

OSM_MAPS_TEMPLATE = (
    "https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map=17/{lat}/{lng}"
)
"""
The documented Maps URL form, not a hand-rolled one.

Built server-side from the stored coordinates so the link can only ever point
at those coordinates. Accepting a URL from the client would make the
destination client-controlled, and this link is the one somebody follows while
trying to reach a patient.
"""


def build_maps_url(latitude: float, longitude: float) -> str:
    """The Google Maps link for a coordinate pair."""
    return OSM_MAPS_TEMPLATE.format(lat=latitude, lng=longitude)


def format_address(profile: EmergencyProfile) -> str:
    """
    The address on one line, assembled in one place.

    Every screen that shows an address shows the same string, and a part that
    was never filled in simply does not appear rather than leaving a stray
    comma.
    """
    parts = [
        profile.house_number,
        profile.street,
        profile.landmark,
        profile.locality,
        profile.city,
        profile.district,
        profile.state,
        profile.country,
        profile.pincode,
    ]
    return ", ".join(p.strip() for p in parts if p and p.strip())


def to_response(profile: EmergencyProfile) -> dict:
    """Flatten a profile row into the response shape."""
    return {
        "id": str(profile.id),
        "contact_name": profile.contact_name,
        "contact_phone": profile.contact_phone,
        "contact_relationship": profile.contact_relationship,
        "alternate_phone": profile.alternate_phone,
        "house_number": profile.house_number,
        "street": profile.street,
        "landmark": profile.landmark,
        "locality": profile.locality,
        "city": profile.city,
        "district": profile.district,
        "state": profile.state,
        "country": profile.country,
        "pincode": profile.pincode,
        "latitude": profile.latitude,
        "longitude": profile.longitude,
        "maps_url": profile.maps_url,
        "location_updated_at": profile.location_updated_at,
        "formatted_address": format_address(profile),
    }


class EmergencyProfileService:
    async def _load(
        self, db: AsyncSession, patient_id: uuid.UUID
    ) -> EmergencyProfile | None:
        return await emergency_profile_repository.get(db, patient_id)

    async def get_profile(self, db: AsyncSession, patient_id: uuid.UUID) -> dict | None:
        """
        The caller's profile, or None if they have not created one yet.

        Absence is not an error. A patient who has never filled the form in is
        the normal starting state, and the page needs to render its empty state
        rather than an error state.
        """
        profile = await self._load(db, patient_id)
        return to_response(profile) if profile else None

    async def upsert_profile(
        self, db: AsyncSession, patient_id: uuid.UUID, payload: EmergencyProfileUpsert
    ) -> dict:
        """
        Create the profile, or update it in place if it already exists.

        One entry point for both because the client has one Save button, and
        because a create/update split would make the caller ask "does it exist
        yet?" — a question it would have to answer with a second request that
        could be stale by the time it acts on it.

        Coordinates are untouched: they are captured by their own endpoint, and
        saving an edited address must not silently discard a position that was
        already recorded.
        """
        fields = {
            **payload.contact.model_dump(),
            **payload.address.model_dump(),
        }

        # Loaded including soft-deleted rows. The primary key is the patient's
        # id, so a row that had been soft-deleted would still occupy that key:
        # the default lookup would report "no profile", the insert would then
        # collide, and the patient would be locked out of ever saving again.
        # Reviving is both the correct outcome and the only one that works.
        profile = await emergency_profile_repository.get(
            db, patient_id, include_deleted=True
        )

        if profile is None:
            # Only checked on the create path. An existing profile is proof the
            # patient row exists — the foreign key guarantees it — so verifying
            # it on every update would be a query per save that can never fail.
            patient = await patient_repository.get(db, patient_id)
            if not patient:
                # The router already guarantees the caller is a patient; this
                # catches the case where the clinical profile row is missing.
                raise EntityNotFoundException("Patient", str(patient_id))

            profile = await emergency_profile_repository.create(
                db, obj_in={"id": patient_id, **fields}
            )
            logger.info("[EMERGENCY_PROFILE_CREATED] patient=%s", patient_id)
        else:
            if profile.deleted_at is not None:
                profile.deleted_at = None
                logger.info("[EMERGENCY_PROFILE_REVIVED] patient=%s", patient_id)
            profile = await emergency_profile_repository.update(
                db, db_obj=profile, obj_in=fields
            )
            logger.info("[EMERGENCY_PROFILE_UPDATED] patient=%s", patient_id)

        return to_response(profile)

    async def update_location(
        self, db: AsyncSession, patient_id: uuid.UUID, payload: EmergencyLocationUpdate
    ) -> dict:
        """
        Record the coordinates the browser reported, and derive the Maps link.

        Requires the profile to exist. A position with no contact and no address
        attached to it is not something the emergency system can act on, and
        creating a half-profile here would let the patient believe they had
        completed a step they had not.
        """
        profile = await self._load(db, patient_id)
        if profile is None:
            raise EntityNotFoundException("Emergency profile", str(patient_id))

        profile.latitude = payload.latitude
        profile.longitude = payload.longitude
        profile.maps_url = build_maps_url(payload.latitude, payload.longitude)
        profile.location_updated_at = datetime.now(timezone.utc)

        # `flush` writes the assignments; no `refresh` follows it because the
        # values just set are the ones being returned, and re-reading the row
        # would be a second query for data already in hand.
        await db.flush()
        logger.info("[EMERGENCY_LOCATION_UPDATED] patient=%s", patient_id)
        return to_response(profile)

    async def clear_location(self, db: AsyncSession, patient_id: uuid.UUID) -> dict:
        """
        Forget the stored coordinates, keeping the contact and address.

        Its own operation because location is the most sensitive part of this
        record: a patient may reasonably want to withdraw it without deleting
        the emergency contact that makes the rest of the profile useful.

        All three location columns are cleared together — a Maps link outliving
        the coordinates it was built from would point somewhere the patient had
        asked to erase.
        """
        profile = await self._load(db, patient_id)
        if profile is None:
            raise EntityNotFoundException("Emergency profile", str(patient_id))

        profile.latitude = None
        profile.longitude = None
        profile.maps_url = None
        profile.location_updated_at = None

        await db.flush()
        logger.info("[EMERGENCY_LOCATION_CLEARED] patient=%s", patient_id)
        return to_response(profile)

    async def delete_profile(self, db: AsyncSession, patient_id: uuid.UUID) -> dict:
        """
        Remove the profile.

        A hard delete, deliberately, where most of this system soft-deletes.
        Two reasons, and the second is a bug the soft path would have:

        The row holds a *third party's* name and telephone number — the
        emergency contact never used this application and never agreed to
        anything. When the patient withdraws it, keeping a shadow copy
        indefinitely is not something either of them asked for.

        And because the primary key is the patient's id, a soft-deleted row
        still occupies that key. The next save would look for a live profile,
        find none, insert, and collide on the primary key — a patient who
        deleted their profile could never create another one.
        """
        profile = await self._load(db, patient_id)
        if profile is None:
            raise EntityNotFoundException("Emergency profile", str(patient_id))

        await emergency_profile_repository.remove(db, id=patient_id)
        logger.info("[EMERGENCY_PROFILE_DELETED] patient=%s", patient_id)
        return {"message": "Emergency profile deleted.", "patient_id": str(patient_id)}


emergency_profile_service = EmergencyProfileService()
