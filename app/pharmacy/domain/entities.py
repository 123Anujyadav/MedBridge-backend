"""Value objects for pharmacy discovery, availability and ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

Availability = Literal["available", "limited", "out_of_stock", "unknown"]

EARTH_RADIUS_KM = 6371.0

# Straight-line distance understates road distance. Multiplying by a detour
# factor keeps the local estimate honest rather than optimistic; Distance Matrix
# replaces it with a real road figure whenever a Maps key is configured.
ROAD_DETOUR_FACTOR = 1.3
URBAN_AVERAGE_SPEED_KMH = 22.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in kilometres.

    Computed locally so ranking works with no Maps key and no network call.
    Distance is needed for every pharmacy on every search; paying an API round
    trip per candidate would make the feature both slow and expensive.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def estimate_travel_minutes(distance_km: float) -> int:
    """Road-time estimate from straight-line distance. At least one minute."""
    road_km = distance_km * ROAD_DETOUR_FACTOR
    return max(1, round((road_km / URBAN_AVERAGE_SPEED_KMH) * 60))


@dataclass
class MedicineAvailability:
    """One prescription line as offered by one pharmacy."""

    requested_name: str
    rxcui: str | None
    requested_quantity: int
    status: Availability = "unknown"

    inventory_id: str | None = None
    matched_name: str | None = None
    generic_name: str | None = None
    brand_name: str | None = None
    strength: str | None = None
    is_generic: bool = False

    mrp: float = 0.0
    unit_price: float = 0.0
    discount_percent: float = 0.0
    stock_quantity: int = 0
    restock_expected_at: str | None = None
    stock_synced_at: str | None = None

    alternatives: list["MedicineAlternative"] = field(default_factory=list)

    @property
    def line_total(self) -> float:
        if self.status == "out_of_stock":
            return 0.0
        return round(self.unit_price * self.requested_quantity, 2)

    @property
    def savings(self) -> float:
        if self.status == "out_of_stock" or self.mrp <= self.unit_price:
            return 0.0
        return round((self.mrp - self.unit_price) * self.requested_quantity, 2)


@dataclass
class MedicineAlternative:
    """
    A substitutable product — usually the generic of a prescribed brand.

    Offered, never applied. Substituting silently would change what the patient
    receives from what the doctor wrote, which is the one thing this whole
    workflow must not do.
    """

    inventory_id: str
    name: str
    generic_name: str | None
    brand_name: str | None
    strength: str | None
    is_generic: bool
    unit_price: float
    mrp: float
    discount_percent: float
    stock_quantity: int
    availability: Availability
    saving_per_unit: float = 0.0


@dataclass
class PharmacyOffer:
    """
    A ranked pharmacy with its answer for the whole prescription.

    `fulfilment_ratio` rather than a boolean, because partial fulfilment is the
    common case and is often still useful — a patient may well take four of five
    lines now and source the fifth elsewhere.
    """

    pharmacy_id: str
    name: str
    address: str
    phone: str | None
    latitude: float
    longitude: float
    rating: float
    total_ratings: int
    is_partner: bool
    is_24x7: bool
    is_open_now: bool
    delivers: bool

    distance_km: float
    travel_minutes: int
    eta_minutes: int
    distance_source: str = "haversine"

    delivery_fee: float = 0.0
    min_order_value: float = 0.0

    items: list[MedicineAvailability] = field(default_factory=list)
    subtotal: float = 0.0
    total_savings: float = 0.0
    grand_total: float = 0.0

    can_order: bool = False
    unavailable_items: list[str] = field(default_factory=list)
    map_url: str = ""
    directions_url: str = ""

    badges: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def fulfilment_ratio(self) -> float:
        if not self.items:
            return 0.0
        supplied = sum(1 for i in self.items if i.status != "out_of_stock")
        return supplied / len(self.items)

    @property
    def fully_available(self) -> bool:
        return bool(self.items) and all(i.status != "out_of_stock" for i in self.items)


@dataclass(frozen=True)
class RankingWeights:
    """
    How the five ranking signals trade off.

    Availability dominates deliberately: the nearest pharmacy that cannot
    dispense the prescription is useless, and a ranking led by distance would
    put it first.
    """

    availability: float = 0.45
    distance: float = 0.25
    eta: float = 0.15
    rating: float = 0.10
    price: float = 0.05


def score_offer(
    offer: PharmacyOffer,
    *,
    max_distance_km: float,
    max_eta: int,
    cheapest_total: float,
    weights: RankingWeights | None = None,
) -> float:
    """
    Normalise the five signals to 0–1 and combine them.

    Each signal is scaled against the best candidate in the current result set
    rather than an absolute constant, so ranking stays meaningful in a dense
    city and in a rural area where the nearest chemist is 30km away.
    """
    w = weights or RankingWeights()

    availability = offer.fulfilment_ratio
    distance = 1.0 - min(offer.distance_km / max_distance_km, 1.0) if max_distance_km else 1.0
    eta = 1.0 - min(offer.eta_minutes / max_eta, 1.0) if max_eta else 1.0
    rating = (offer.rating / 5.0) if offer.rating else 0.0
    price = (
        min(cheapest_total / offer.grand_total, 1.0)
        if cheapest_total > 0 and offer.grand_total > 0
        else 0.0
    )

    return round(
        w.availability * availability
        + w.distance * distance
        + w.eta * eta
        + w.rating * rating
        + w.price * price,
        4,
    )


def assign_badges(offers: Sequence[PharmacyOffer]) -> None:
    """
    Tag the standouts, in place.

    Only awarded to pharmacies that can actually dispense everything — labelling
    a shop "Nearest" when it holds none of the prescription is worse than not
    labelling anything.
    """
    orderable = [o for o in offers if o.can_order and o.fully_available]
    if not orderable:
        return

    nearest = min(orderable, key=lambda o: o.distance_km)
    nearest.badges.append("Nearest")

    fastest = min(orderable, key=lambda o: o.eta_minutes)
    if "Fastest delivery" not in fastest.badges:
        fastest.badges.append("Fastest delivery")

    cheapest = min(orderable, key=lambda o: o.grand_total)
    if "Lowest price" not in cheapest.badges:
        cheapest.badges.append("Lowest price")

    for offer in orderable:
        if offer.is_24x7:
            offer.badges.append("Open 24×7")
