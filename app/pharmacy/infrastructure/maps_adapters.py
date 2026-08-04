"""
Mapping adapters for the pharmacy context.

Both wrap the shared `MapsService`, which owns key handling, timeouts, the
Nominatim rate limiter and one-warning-per-failure-class logging. Nothing here
re-reads the key or opens its own HTTP client, so switching routing on remains
a configuration change in exactly one place.

Both are optional. Without an ORS key, distance falls back to the local
haversine estimate; discovery keeps working either way because OpenStreetMap
needs no key at all — which is a strict improvement on the previous Places
implementation, where an unset key meant no discovery.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

import httpx

from app.services.maps import (
    NOMINATIM_USER_AGENT,
    OVERPASS_URL,
    REQUEST_TIMEOUT_SECONDS,
    get_maps_service,
)

logger = logging.getLogger(__name__)

# The ORS matrix is quota-limited and adds latency to a screen the patient is
# waiting on, so only the closest few candidates are upgraded to real road
# figures. The rest keep their haversine estimate, which is good enough to rank.
MAX_ROAD_LOOKUPS = 8


class ORSDistanceSource:
    """Implements `DistanceSource` using the OpenRouteService matrix."""

    name = "ors_matrix"

    async def travel_times(
        self,
        *,
        origin: tuple[float, float],
        destinations: Sequence[tuple[float, float]],
    ) -> list[tuple[float, int] | None] | None:
        maps = get_maps_service()
        if not maps.is_enabled() or not destinations:
            return None

        async def one(destination: tuple[float, float]) -> tuple[float, int] | None:
            try:
                body = await maps.distance_matrix(origin, destination)
            except Exception as exc:
                logger.warning("[PHARMACY_DISTANCE_ELEMENT_FAILED] %s", exc)
                return None
            if not body:
                return None
            km = body.get("distance_km")
            minutes = body.get("duration_minutes")
            if km is None or minutes is None:
                return None
            return float(km), int(minutes)

        head = list(destinations)[:MAX_ROAD_LOOKUPS]
        results = await asyncio.gather(*(one(d) for d in head), return_exceptions=True)

        resolved: list[tuple[float, int] | None] = []
        for item in results:
            resolved.append(None if isinstance(item, BaseException) else item)

        # Pad so the caller can index by candidate position without bounds
        # checks; the tail simply keeps its local estimate.
        resolved.extend([None] * (len(destinations) - len(resolved)))
        return resolved


class OSMPharmacyDiscovery:
    """
    Implements `PharmacyDiscoverySource` using OpenStreetMap via Overpass.

    Finds real chemists near a point. Deliberately returns *no* stock, price or
    availability — OSM does not know any of that, and a discovered shop is
    stored with `is_partner=False` so it can never be ordered from.

    Needs no API key, so unlike the Places implementation it keeps working in
    environments where routing is unconfigured.
    """

    name = "openstreetmap"

    async def discover(
        self, *, latitude: float, longitude: float, radius_km: float = 5.0, limit: int = 10
    ) -> list[dict]:
        radius_m = int(radius_km * 1000)
        query = f"""
        [out:json][timeout:10];
        (
          node["amenity"="pharmacy"](around:{radius_m},{latitude},{longitude});
          way["amenity"="pharmacy"](around:{radius_m},{latitude},{longitude});
        );
        out center {max(limit * 2, 20)};
        """

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    OVERPASS_URL,
                    data={"data": query},
                    headers={"User-Agent": NOMINATIM_USER_AGENT},
                )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            # Not an error the caller must handle: the partner network is found
            # from our own tables, and discovery only adds context.
            logger.warning("[PHARMACY_DISCOVERY_FAILED] %s", exc)
            return []

        found: list[dict] = []
        for element in (body.get("elements") or [])[: limit * 2]:
            centre = element.get("center") or element
            lat, lon = centre.get("lat"), centre.get("lon")
            if lat is None or lon is None:
                continue

            tags = element.get("tags") or {}
            found.append(
                {
                    # Keyed the same way the column is named, so the upsert in
                    # `discover_and_register` is untouched by this migration.
                    "google_place_id": f"osm:{element.get('type')}:{element.get('id')}",
                    "name": tags.get("name") or "Pharmacy",
                    "address": tags.get("addr:full")
                    or ", ".join(
                        filter(
                            None,
                            [
                                tags.get("addr:housenumber"),
                                tags.get("addr:street"),
                                tags.get("addr:city"),
                            ],
                        )
                    )
                    or "",
                    "latitude": lat,
                    "longitude": lon,
                    # OSM carries no rating or live opening state. Reported as
                    # zero and None rather than invented; the caller already
                    # treats both as unknown.
                    "rating": 0.0,
                    "total_ratings": 0,
                    "open_now": None,
                }
            )
            if len(found) >= limit:
                break

        logger.info(
            "[PHARMACY_DISCOVERED] %d place(s) near %.4f,%.4f", len(found), latitude, longitude
        )
        return found
