"""
Turning a prescription into ranked pharmacy offers.

Owns the translation from clinical record to stock question: a `Medication` row
becomes a `MedicineRequirement`, carrying the RxCUI that safety verification
already resolved so inventory joins on a real key instead of a drug name.
"""

from __future__ import annotations

import logging
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.pharmacy import Pharmacy
from app.models.prescription import Medication, Prescription
from app.pharmacy.domain.entities import PharmacyOffer
from app.pharmacy.domain.ports import MedicineRequirement

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_KM = 10.0
MAX_RESULTS = 5
"""Step 5 of the workflow specifies at most five pharmacies."""


def _quantity_for(medication: Medication) -> int:
    """
    How many units to price.

    Prefers the clinician's explicit `quantity`, falls back to the computed
    dose count, and finally to one. Ordering zero of something because a field
    was blank would silently drop a medicine from the basket.
    """
    if medication.quantity and medication.quantity > 0:
        return medication.quantity
    if medication.total_doses and medication.total_doses > 0:
        return medication.total_doses
    return 1


def requirements_from_medications(
    medications: Sequence[Medication],
) -> list[MedicineRequirement]:
    return [
        MedicineRequirement(
            name=m.name,
            generic_name=m.generic_name,
            rxcui=m.rxcui,
            strength=m.strength,
            quantity=_quantity_for(m),
            medication_id=str(m.id),
        )
        for m in medications
        if (m.name or "").strip()
    ]


class PharmacySearchService:
    def __init__(self, provider, discovery=None) -> None:
        self._provider = provider
        self._discovery = discovery

    async def offers_for_prescription(
        self,
        db: AsyncSession,
        *,
        prescription_id: uuid.UUID,
        patient_id: uuid.UUID,
        latitude: float,
        longitude: float,
        radius_km: float = DEFAULT_RADIUS_KM,
        limit: int = MAX_RESULTS,
    ) -> list[PharmacyOffer]:
        result = await db.execute(
            select(Prescription)
            .where(Prescription.id == prescription_id)
            .options(selectinload(Prescription.medications))
        )
        prescription = result.scalar_one_or_none()
        if not prescription:
            raise EntityNotFoundException("Prescription", str(prescription_id))
        if prescription.patient_id != patient_id:
            raise AuthorizationException(
                "You cannot search pharmacies for another patient's prescription."
            )

        requirements = requirements_from_medications(prescription.medications)
        if not requirements:
            return []

        return await self._provider.find_offers(
            db=db,
            latitude=latitude,
            longitude=longitude,
            requirements=requirements,
            radius_km=radius_km,
            limit=limit,
        )

    async def discover_and_register(
        self, db: AsyncSession, *, latitude: float, longitude: float, radius_km: float = 5.0
    ) -> int:
        """
        Record nearby chemists found through Places.

        They land as `is_partner=False`, so they are visible and routable but
        cannot be ordered from — Places knows where a shop is, never what is on
        its shelves. Returns how many rows were created.

        Upserted on `google_place_id` so repeated sweeps refresh the existing
        row instead of stacking duplicates of the same shop.
        """
        if not self._discovery:
            return 0

        places = await self._discovery.discover(
            latitude=latitude, longitude=longitude, radius_km=radius_km, limit=20
        )
        if not places:
            return 0

        created = 0
        for place in places:
            place_id = place.get("google_place_id")
            if not place_id:
                continue

            existing = (
                await db.execute(
                    select(Pharmacy).where(Pharmacy.google_place_id == place_id)
                )
            ).scalar_one_or_none()

            if existing:
                existing.rating = place.get("rating") or existing.rating
                existing.total_ratings = place.get("total_ratings") or existing.total_ratings
                continue

            db.add(
                Pharmacy(
                    name=place.get("name") or "Pharmacy",
                    address=place.get("address") or "",
                    latitude=place["latitude"],
                    longitude=place["longitude"],
                    google_place_id=place_id,
                    rating=place.get("rating") or 0.0,
                    total_ratings=place.get("total_ratings") or 0,
                    is_partner=False,
                    is_active=True,
                    delivers=False,
                )
            )
            created += 1

        logger.info("[PHARMACY_DISCOVERY] registered=%d near %.4f,%.4f",
                    created, latitude, longitude)
        return created
