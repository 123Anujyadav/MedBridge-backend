"""
Provider ports for the pharmacy network.

The application layer depends only on these. Plugging in Apollo, Tata 1mg,
PharmEasy, Netmeds or an ONDC gateway means writing one adapter and registering
it in `factory.py` — no service, endpoint or model changes, because none of them
name a vendor.

`LocalDbPharmacyProvider` is the active implementation: MedBridge owns the
inventory, so it is the only source that can answer stock and price questions
truthfully today.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from app.pharmacy.domain.entities import PharmacyOffer


@runtime_checkable
class PharmacyProvider(Protocol):
    """
    Finds pharmacies and answers what they can dispense.

    Implementations degrade rather than raise: an unreachable upstream returns
    an empty list, and the caller reports "no pharmacies found nearby" instead
    of failing the whole prescription screen.
    """

    name: str

    @property
    def supports_ordering(self) -> bool:
        """
        Whether orders may be placed through this provider.

        A discovery-only source (Places, say) can locate a chemist but cannot
        transact with it, and the UI must not offer a button that cannot work.
        """
        ...

    async def find_offers(
        self,
        *,
        db,
        latitude: float,
        longitude: float,
        requirements: Sequence["MedicineRequirement"],
        radius_km: float,
        limit: int,
    ) -> list[PharmacyOffer]:
        ...


@runtime_checkable
class PharmacyDiscoverySource(Protocol):
    """
    Locates pharmacies geographically without knowing their stock.

    Google Places fills this: it returns real chemists, coordinates, ratings and
    opening state, and nothing whatsoever about inventory.
    """

    name: str

    async def discover(
        self, *, latitude: float, longitude: float, radius_km: float, limit: int
    ) -> list[dict]:
        ...


@runtime_checkable
class DistanceSource(Protocol):
    """
    Road distance and travel time.

    Optional throughout. When absent the caller falls back to a haversine
    estimate, which is why every consumer must tolerate None.
    """

    name: str

    async def travel_times(
        self,
        *,
        origin: tuple[float, float],
        destinations: Sequence[tuple[float, float]],
    ) -> list[tuple[float, int] | None] | None:
        """Per destination: (distance_km, minutes), or None where unknown."""
        ...


class MedicineRequirement:
    """
    One prescription line, as a stock question.

    Carries `rxcui` because that is what inventory joins on; `name` is the
    fallback for lines that never normalised, and the display label throughout.
    """

    __slots__ = ("name", "generic_name", "rxcui", "quantity", "strength", "medication_id")

    def __init__(
        self,
        *,
        name: str,
        quantity: int = 1,
        rxcui: str | None = None,
        generic_name: str | None = None,
        strength: str | None = None,
        medication_id: str | None = None,
    ) -> None:
        self.name = name
        self.generic_name = generic_name
        self.rxcui = rxcui
        self.quantity = max(1, quantity)
        self.strength = strength
        self.medication_id = medication_id

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MedicineRequirement {self.name} x{self.quantity} rxcui={self.rxcui}>"
