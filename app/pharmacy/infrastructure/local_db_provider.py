"""
The MedBridge-owned pharmacy network.

Inventory, pricing and discounts come from our own tables, which is what makes
stock answers truthful rather than guessed. Geography is handled locally with a
haversine filter so ranking never depends on an external key; a configured Maps
key upgrades the distances to real road figures but is never required.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.pharmacy import Pharmacy, PharmacyInventory
from app.pharmacy.domain.entities import (
    MedicineAlternative,
    MedicineAvailability,
    PharmacyOffer,
    assign_badges,
    estimate_travel_minutes,
    haversine_km,
    score_offer,
)
from app.pharmacy.domain.ports import DistanceSource, MedicineRequirement

logger = logging.getLogger(__name__)

# Fetched before ranking so distance, price and availability can be compared
# across a real field of candidates. Ranking the first five rows the database
# happened to return would make "nearest" and "cheapest" meaningless.
CANDIDATE_FETCH_LIMIT = 60


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _is_open_now(pharmacy: Pharmacy, now: datetime | None = None) -> bool:
    """
    Opening state from the stored window.

    Unknown hours count as open rather than closed: hiding a pharmacy because
    nobody filled in its timings loses a real dispensing option, while showing
    one that turns out to be shut costs a phone call.
    """
    if pharmacy.is_24x7:
        return True
    if not pharmacy.opens_at or not pharmacy.closes_at:
        return True

    current = (now or datetime.now(timezone.utc)).strftime("%H:%M")
    opens, closes = pharmacy.opens_at, pharmacy.closes_at

    if opens <= closes:
        return opens <= current <= closes
    # Window crossing midnight, e.g. 22:00–06:00.
    return current >= opens or current <= closes


class LocalDbPharmacyProvider:
    """Implements `PharmacyProvider` against the MedBridge database."""

    name = "local_db"

    def __init__(self, distance_source: DistanceSource | None = None) -> None:
        self._distance = distance_source

    @property
    def supports_ordering(self) -> bool:
        return True

    # ── candidate selection ──────────────────────────────────────────────

    async def _candidates(
        self, db: AsyncSession, latitude: float, longitude: float, radius_km: float
    ) -> list[tuple[Pharmacy, float]]:
        """
        Partner pharmacies within radius, nearest first.

        A bounding box narrows the scan in SQL before the exact haversine runs
        in Python, so a growing network does not mean loading every pharmacy in
        the country to answer one search.
        """
        # One degree of latitude is ~111km everywhere; longitude shrinks with
        # latitude, and the cosine term is clamped so a search near the poles
        # cannot produce a division blow-up.
        import math

        lat_delta = radius_km / 111.0
        cos_lat = max(math.cos(math.radians(latitude)), 0.01)
        lon_delta = radius_km / (111.0 * cos_lat)

        result = await db.execute(
            select(Pharmacy)
            .where(
                Pharmacy.is_active.is_(True),
                Pharmacy.is_partner.is_(True),
                Pharmacy.deleted_at.is_(None),
                Pharmacy.latitude.between(latitude - lat_delta, latitude + lat_delta),
                Pharmacy.longitude.between(longitude - lon_delta, longitude + lon_delta),
            )
            .limit(CANDIDATE_FETCH_LIMIT)
        )

        within: list[tuple[Pharmacy, float]] = []
        for pharmacy in result.scalars().all():
            distance = haversine_km(
                latitude, longitude, pharmacy.latitude, pharmacy.longitude
            )
            # The box is a superset of the circle; this trims the corners, and
            # also respects each pharmacy's own delivery radius.
            if distance <= min(radius_km, pharmacy.delivery_radius_km or radius_km):
                within.append((pharmacy, distance))

        within.sort(key=lambda pair: pair[1])
        return within

    # ── inventory matching ───────────────────────────────────────────────

    async def _inventory_for(
        self,
        db: AsyncSession,
        pharmacy_ids: Sequence[uuid.UUID],
        requirements: Sequence[MedicineRequirement],
    ) -> dict:
        """
        Every inventory row at these pharmacies that could satisfy any line.

        One query for the whole search rather than one per pharmacy per drug —
        five pharmacies and five medicines would otherwise be twenty-five round
        trips to answer a single screen.
        """
        if not pharmacy_ids or not requirements:
            return {}

        rxcuis = [r.rxcui for r in requirements if r.rxcui]
        names: list[str] = []
        for requirement in requirements:
            for candidate in (requirement.name, requirement.generic_name):
                if candidate:
                    names.append(candidate.strip().lower())

        conditions = []
        if rxcuis:
            conditions.append(PharmacyInventory.rxcui.in_(rxcuis))
        for name in set(names):
            # Name matching is the fallback for lines that never resolved to an
            # RxCUI. Deliberately matched against both the product and generic
            # columns, since a brand may be stocked under either.
            conditions.append(PharmacyInventory.medicine_name.ilike(f"%{name}%"))
            conditions.append(PharmacyInventory.generic_name.ilike(f"%{name}%"))

        if not conditions:
            return {}

        result = await db.execute(
            select(PharmacyInventory).where(
                PharmacyInventory.pharmacy_id.in_(list(pharmacy_ids)),
                PharmacyInventory.deleted_at.is_(None),
                or_(*conditions),
            )
        )

        grouped: dict = {}
        for row in result.scalars().all():
            grouped.setdefault(str(row.pharmacy_id), []).append(row)
        return grouped

    @staticmethod
    def _matches(row: PharmacyInventory, requirement: MedicineRequirement) -> bool:
        if requirement.rxcui and row.rxcui:
            return row.rxcui == requirement.rxcui
        wanted = {
            (requirement.name or "").strip().lower(),
            (requirement.generic_name or "").strip().lower(),
        }
        wanted.discard("")
        haystack = " ".join(
            filter(None, [row.medicine_name, row.generic_name, row.brand_name])
        ).lower()
        return any(term in haystack for term in wanted)

    def _availability(
        self, rows: Sequence[PharmacyInventory], requirement: MedicineRequirement
    ) -> MedicineAvailability:
        matches = [row for row in rows if self._matches(row, requirement)]

        result = MedicineAvailability(
            requested_name=requirement.name,
            rxcui=requirement.rxcui,
            requested_quantity=requirement.quantity,
        )

        if not matches:
            result.status = "out_of_stock"
            return result

        # Prefer something that can actually supply the full quantity, cheapest
        # first. Falling straight to the cheapest row would happily pick one
        # with a single unit left.
        suppliable = [row for row in matches if row.can_supply(requirement.quantity)]
        chosen = min(
            suppliable or matches, key=lambda row: (row.selling_price or row.mrp)
        )

        result.inventory_id = str(chosen.id)
        result.matched_name = chosen.medicine_name
        result.generic_name = chosen.generic_name
        result.brand_name = chosen.brand_name
        result.strength = chosen.strength
        result.is_generic = chosen.is_generic
        result.mrp = round(chosen.mrp, 2)
        result.unit_price = round(chosen.selling_price or chosen.mrp, 2)
        result.discount_percent = round(chosen.discount_percent, 2)
        result.stock_quantity = chosen.stock_quantity
        result.restock_expected_at = _iso(chosen.restock_expected_at)
        result.stock_synced_at = _iso(chosen.stock_synced_at)
        result.status = (
            chosen.availability
            if chosen.can_supply(requirement.quantity)
            else ("limited" if chosen.stock_quantity > 0 else "out_of_stock")
        )

        # Cheaper equivalents, offered for the patient to choose. Never applied.
        for row in matches:
            if row.id == chosen.id or row.stock_quantity <= 0:
                continue
            price = row.selling_price or row.mrp
            if price >= result.unit_price:
                continue
            result.alternatives.append(
                MedicineAlternative(
                    inventory_id=str(row.id),
                    name=row.medicine_name,
                    generic_name=row.generic_name,
                    brand_name=row.brand_name,
                    strength=row.strength,
                    is_generic=row.is_generic,
                    unit_price=round(price, 2),
                    mrp=round(row.mrp, 2),
                    discount_percent=round(row.discount_percent, 2),
                    stock_quantity=row.stock_quantity,
                    availability=row.availability,
                    saving_per_unit=round(result.unit_price - price, 2),
                )
            )

        result.alternatives.sort(key=lambda alt: alt.unit_price)
        return result

    # ── the port ─────────────────────────────────────────────────────────

    async def find_offers(
        self,
        *,
        db: AsyncSession,
        latitude: float,
        longitude: float,
        requirements: Sequence[MedicineRequirement],
        radius_km: float = 10.0,
        limit: int = 5,
    ) -> list[PharmacyOffer]:
        candidates = await self._candidates(db, latitude, longitude, radius_km)
        if not candidates:
            logger.info(
                "[PHARMACY_NONE_NEARBY] lat=%.4f lng=%.4f radius=%.1fkm",
                latitude, longitude, radius_km,
            )
            return []

        # UUID objects for the query — the column is UUID-typed and a string
        # here fails at bind time. The grouping below keys by str for lookup.
        inventory = await self._inventory_for(
            db, [p.id for p, _ in candidates], requirements
        )

        # Real road distances when a Maps key is configured; the haversine
        # estimate stands in otherwise. Never fatal either way.
        road: list[tuple[float, int] | None] | None = None
        if self._distance:
            try:
                road = await self._distance.travel_times(
                    origin=(latitude, longitude),
                    destinations=[(p.latitude, p.longitude) for p, _ in candidates],
                )
            except Exception as exc:
                logger.warning("[PHARMACY_DISTANCE_FAILED] %s", exc)

        offers: list[PharmacyOffer] = []
        for index, (pharmacy, straight_km) in enumerate(candidates):
            distance_km, travel_minutes, source = straight_km, estimate_travel_minutes(straight_km), "haversine"
            if road and index < len(road) and road[index]:
                distance_km, travel_minutes = road[index]
                source = self._distance.name if self._distance else "haversine"

            rows = inventory.get(str(pharmacy.id), [])
            items = [self._availability(rows, requirement) for requirement in requirements]

            subtotal = round(sum(i.line_total for i in items), 2)
            savings = round(sum(i.savings for i in items), 2)
            unavailable = [i.requested_name for i in items if i.status == "out_of_stock"]

            fee = pharmacy.delivery_fee or 0.0
            if pharmacy.free_delivery_above and subtotal >= pharmacy.free_delivery_above:
                fee = 0.0

            offer = PharmacyOffer(
                pharmacy_id=str(pharmacy.id),
                name=pharmacy.name,
                address=pharmacy.address,
                phone=pharmacy.phone,
                latitude=pharmacy.latitude,
                longitude=pharmacy.longitude,
                rating=pharmacy.rating,
                total_ratings=pharmacy.total_ratings,
                is_partner=pharmacy.is_partner,
                is_24x7=pharmacy.is_24x7,
                is_open_now=_is_open_now(pharmacy),
                delivers=pharmacy.delivers,
                distance_km=round(distance_km, 2),
                travel_minutes=travel_minutes,
                eta_minutes=travel_minutes + (pharmacy.avg_prep_minutes or 0),
                distance_source=source,
                delivery_fee=round(fee, 2),
                min_order_value=pharmacy.min_order_value or 0.0,
                items=items,
                subtotal=subtotal,
                total_savings=savings,
                grand_total=round(subtotal + fee, 2),
                unavailable_items=unavailable,
                map_url=(
                    f"https://www.openstreetmap.org/?mlat={pharmacy.latitude}"
                    f"&mlon={pharmacy.longitude}"
                    f"#map=17/{pharmacy.latitude}/{pharmacy.longitude}"
                ),
                directions_url=(
                    "https://www.openstreetmap.org/directions"
                    "?engine=fossgis_osrm_car"
                    f"&route={latitude}%2C{longitude}"
                    f"%3B{pharmacy.latitude}%2C{pharmacy.longitude}"
                ),
            )

            # Orderable only if the shop can transact, is open, and holds at
            # least something — plus clears its own minimum basket value.
            offer.can_order = bool(
                pharmacy.can_fulfil
                and offer.is_open_now
                and offer.fulfilment_ratio > 0
                and subtotal >= (pharmacy.min_order_value or 0.0)
            )
            offers.append(offer)

        max_distance = max((o.distance_km for o in offers), default=1.0) or 1.0
        max_eta = max((o.eta_minutes for o in offers), default=1) or 1
        totals = [o.grand_total for o in offers if o.grand_total > 0]
        cheapest = min(totals) if totals else 0.0

        for offer in offers:
            offer.score = score_offer(
                offer,
                max_distance_km=max_distance,
                max_eta=max_eta,
                cheapest_total=cheapest,
            )

        offers.sort(key=lambda o: o.score, reverse=True)
        top = offers[:limit]
        assign_badges(top)

        logger.info(
            "[PHARMACY_OFFERS] lat=%.4f lng=%.4f candidates=%d returned=%d orderable=%d",
            latitude, longitude, len(candidates), len(top),
            sum(1 for o in top if o.can_order),
        )
        return top
