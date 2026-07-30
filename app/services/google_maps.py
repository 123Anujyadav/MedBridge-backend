"""
Google Maps, optional by design.

The key is not available yet, and the platform has to ship, deploy and run an
emergency workflow without it. So every method here has two honest outcomes:
the real answer, or nothing. There is no third outcome where a plausible
address, a nearby hospital or an ETA is invented to fill the gap — an estimated
ETA on an emergency screen is a number somebody plans around, and a fabricated
hospital is an address an ambulance is sent to.

**Deferred activation.** The key is read from settings *at call time*, not
captured at import. Adding `GOOGLE_MAPS_API_KEY` to the environment and
restarting turns reverse geocoding, hospital search and distance/ETA on with no
code change anywhere. `is_enabled()` is the single place that decides, and the
"not configured" warning is logged once per process rather than once per
request, so an unconfigured deployment does not fill its logs.

The map *link* is the exception: `https://www.google.com/maps/search/?api=1&…`
takes no key, so a responder always gets a working link to the coordinates even
with nothing configured.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

MAPS_SEARCH_TEMPLATE = "https://www.google.com/maps/search/?api=1&query={lat},{lng}"
MAPS_DIRECTIONS_TEMPLATE = (
    "https://www.google.com/maps/dir/?api=1&origin={o_lat},{o_lng}"
    "&destination={d_lat},{d_lng}&travelmode=driving"
)

DEFAULT_SEARCH_RADIUS_METRES = 10_000
REQUEST_TIMEOUT_SECONDS = 8.0
"""
Short on purpose. This runs while an emergency is being raised; a slow map
lookup must never be the reason a responder is alerted late. A timeout is
treated exactly like a missing key — no data, carry on.
"""


class GoogleMapsService:
    """Everything that needs the Maps platform, and nothing that pretends to."""

    def __init__(self) -> None:
        self._warned = False
        self._rejections: set[tuple[str, str]] = set()

    # ── availability ─────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        """
        Whether a key is configured *right now*.

        Read from settings on every call rather than cached, which is what
        makes activation a configuration change instead of a deployment.
        """
        return bool((settings.GOOGLE_MAPS_API_KEY or "").strip())

    def _unavailable(self, operation: str) -> None:
        """Log the first skip per process, then stay quiet."""
        if not self._warned:
            logger.warning(
                "[MAPS_DISABLED] GOOGLE_MAPS_API_KEY is not set; %s and every "
                "later Maps lookup will be skipped. Coordinates and the plain "
                "map link continue to work.", operation,
            )
            self._warned = True

    async def _get(self, url: str, params: dict) -> Optional[dict]:
        """
        One request, with every failure flattened to None.

        A network error, a timeout, a quota rejection and a malformed body are
        the same thing to every caller here: no data. None of them may raise
        into an emergency being raised.
        """
        if not self.is_enabled():
            return None

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    url, params={**params, "key": settings.GOOGLE_MAPS_API_KEY}
                )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            logger.warning("[MAPS_REQUEST_FAILED] %s: %s", url, exc)
            return None

        status = body.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            # REQUEST_DENIED, OVER_QUERY_LIMIT, INVALID_REQUEST — all mean the
            # answer is unusable, and are worth seeing in the log because they
            # are configuration problems rather than absence of data. Google's
            # own message names the fix, so it is passed through verbatim.
            #
            # Logged once per API and status, not once per call: a key whose
            # project has an API switched off is rejected identically forever,
            # and three warnings per emergency would bury the emergencies.
            seen = (url, status)
            if seen not in self._rejections:
                self._rejections.add(seen)
                logger.warning(
                    "[MAPS_REJECTED] %s -> %s: %s (logged once per status; "
                    "the emergency continues without map data)", url, status,
                    body.get("error_message", ""),
                )
            return None
        return body

    # ── links (no key required) ──────────────────────────────────────────

    def build_maps_url(self, latitude: float, longitude: float) -> str:
        """A link to the coordinates. Works with or without a key."""
        return MAPS_SEARCH_TEMPLATE.format(lat=latitude, lng=longitude)

    def build_directions_url(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> str:
        """Driving directions between two points. Also key-free."""
        return MAPS_DIRECTIONS_TEMPLATE.format(
            o_lat=origin[0], o_lng=origin[1],
            d_lat=destination[0], d_lng=destination[1],
        )

    # ── key-backed lookups ───────────────────────────────────────────────

    async def reverse_geocode(
        self, latitude: float, longitude: float
    ) -> Optional[str]:
        """The street address for a coordinate pair, or None."""
        if not self.is_enabled():
            self._unavailable("reverse geocoding")
            return None

        body = await self._get(GEOCODE_URL, {"latlng": f"{latitude},{longitude}"})
        if not body:
            return None
        for result in body.get("results") or []:
            formatted = result.get("formatted_address")
            if formatted:
                return formatted
        return None

    async def find_nearby_hospitals(
        self, latitude: float, longitude: float,
        radius_metres: int = DEFAULT_SEARCH_RADIUS_METRES, limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Hospitals near a point, nearest first, or an empty list.

        An empty list means "we do not know", never "there are none nearby" —
        callers must not present it as the latter.
        """
        if not self.is_enabled():
            self._unavailable("nearby hospital search")
            return []

        body = await self._get(PLACES_NEARBY_URL, {
            "location": f"{latitude},{longitude}",
            "radius": radius_metres,
            "type": "hospital",
            "rankby": "prominence",
        })
        if not body:
            return []

        hospitals: list[dict[str, Any]] = []
        for place in (body.get("results") or [])[:limit]:
            location = (place.get("geometry") or {}).get("location") or {}
            if location.get("lat") is None or location.get("lng") is None:
                continue
            hospitals.append({
                "place_id": place.get("place_id"),
                "name": place.get("name"),
                "address": place.get("vicinity") or place.get("formatted_address"),
                "latitude": location["lat"],
                "longitude": location["lng"],
                "rating": place.get("rating"),
                "open_now": (place.get("opening_hours") or {}).get("open_now"),
            })
        return hospitals

    async def distance_matrix(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> Optional[dict[str, Any]]:
        """
        Driving distance and duration between two points, or None.

        Returns kilometres and whole minutes — the units the emergency record
        stores — so no caller has to convert and get it wrong.
        """
        if not self.is_enabled():
            self._unavailable("distance and ETA")
            return None

        body = await self._get(DISTANCE_MATRIX_URL, {
            "origins": f"{origin[0]},{origin[1]}",
            "destinations": f"{destination[0]},{destination[1]}",
            "mode": "driving",
        })
        if not body:
            return None

        try:
            element = body["rows"][0]["elements"][0]
        except (KeyError, IndexError):
            return None
        if element.get("status") != "OK":
            return None

        metres = (element.get("distance") or {}).get("value")
        seconds = (element.get("duration") or {}).get("value")
        if metres is None or seconds is None:
            return None

        return {
            "distance_km": round(metres / 1000.0, 2),
            "duration_minutes": max(1, round(seconds / 60.0)),
            "distance_text": (element.get("distance") or {}).get("text"),
            "duration_text": (element.get("duration") or {}).get("text"),
        }

    async def nearest_hospital_with_eta(
        self, latitude: float, longitude: float
    ) -> Optional[dict[str, Any]]:
        """
        The nearest hospital and how long it takes to reach it, or None.

        Two lookups rather than one because Places ranks by prominence, not by
        driving time; the distance matrix is what turns a list into an answer.
        If the second call fails the hospital is still returned, with the ETA
        left null — a known facility with an unknown ETA is useful, an invented
        ETA is not.
        """
        hospitals = await self.find_nearby_hospitals(latitude, longitude, limit=5)
        if not hospitals:
            return None

        best: Optional[dict[str, Any]] = None
        for hospital in hospitals:
            metrics = await self.distance_matrix(
                (latitude, longitude), (hospital["latitude"], hospital["longitude"])
            )
            candidate = {**hospital, **(metrics or {})}
            if metrics is None:
                best = best or candidate
                continue
            if best is None or metrics["duration_minutes"] < best.get(
                "duration_minutes", float("inf")
            ):
                best = candidate
        return best


_service: GoogleMapsService | None = None


def get_google_maps_service() -> GoogleMapsService:
    """The process-wide service. Replaceable in tests."""
    global _service
    if _service is None:
        _service = GoogleMapsService()
    return _service


def set_google_maps_service(service: GoogleMapsService | None) -> None:
    global _service
    _service = service
