"""
Mapping, geocoding and routing — OpenStreetMap, Nominatim and OpenRouteService.

Replaces the Google Maps service and deliberately keeps its exact public
surface: `is_enabled`, `reverse_geocode`, `find_nearby_hospitals`,
`distance_matrix`, `nearest_hospital_with_eta`, `build_maps_url` and
`build_directions_url`, with identical return shapes. Every existing caller —
emergency comms, SOS timeline, pharmacy ranking, delivery routing — keeps
working with only its import line changed. That is what makes this a migration
rather than a rewrite.

Three upstreams, each with a different failure posture:

* **ORS Matrix** — distance and duration. Needs `ORS_API_KEY`.
* **ORS Directions** — route geometry for polylines. Same key.
* **Nominatim + Overpass** — geocoding and place search. No key, but a strict
  usage policy: a real User-Agent is required and requests must be rate
  limited, so a shared limiter serialises them.

Every failure flattens to None or an empty list, exactly as before. An empty
result means "we do not know", never "there are none" — the callers were
written against that contract and it is preserved verbatim.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── endpoints ────────────────────────────────────────────────────────────

ORS_BASE = "https://api.openrouteservice.org"
ORS_MATRIX_URL = f"{ORS_BASE}/v2/matrix/driving-car"
# The `/geojson` variant is required: the bare endpoint answers with a
# `routes[]` array carrying an encoded polyline, while this one returns
# `features[]` with decoded [lng, lat] coordinates ready to draw.
ORS_DIRECTIONS_URL = f"{ORS_BASE}/v2/directions/driving-car/geojson"

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Public OSM links, replacing the google.com/maps equivalents. Both are
# key-free and open in any browser.
OSM_SEARCH_TEMPLATE = (
    "https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map=17/{lat}/{lng}"
)
OSM_DIRECTIONS_TEMPLATE = (
    "https://www.openstreetmap.org/directions?engine=fossgis_osrm_car"
    "&route={o_lat}%2C{o_lng}%3B{d_lat}%2C{d_lng}"
)

DEFAULT_SEARCH_RADIUS_METRES = 10_000
REQUEST_TIMEOUT_SECONDS = 8.0
"""
Short on purpose. This runs while an emergency is being raised; a slow map
lookup must never be the reason a responder is alerted late. A timeout is
treated exactly like a missing key — no data, carry on.
"""

# Nominatim's usage policy caps anonymous callers at one request a second and
# requires an identifying User-Agent. Both are honoured here rather than left
# to chance, because being blocked would take reverse geocoding down platform-wide.
NOMINATIM_MIN_INTERVAL_SECONDS = 1.1

# Overpass is a free, heavily shared instance that answers complex queries and
# sheds load by timing out rather than by returning an error. Its usage policy
# asks for gentle, serialised access, so it gets its own gap and one retry.
OVERPASS_MIN_INTERVAL_SECONDS = 1.0
OVERPASS_ATTEMPTS = 2

# Overpass needs a far longer budget than the routing APIs: an `around:` query
# over a dense city regularly runs for ten seconds or more. The server-side
# budget is declared in the query itself, and the client timeout must exceed
# it — otherwise we hang up on a request the server is still allowed to be
# working on, which is what the shared 8s timeout was doing.
OVERPASS_QUERY_BUDGET_SECONDS = 20
OVERPASS_TIMEOUT_SECONDS = 25.0
OVERPASS_BACKOFF_SECONDS = 2.0
OVERPASS_BACKOFF_CAP_SECONDS = 10.0

OVERPASS_TOTAL_BUDGET_SECONDS = 12.0
"""
A hard ceiling on the whole lookup, retries and backoff included.

Per-attempt timeouts do not bound a retrying call: two 25s attempts either side
of a backoff is nearly a minute, and this sits on the emergency path where the
module's own rule is that a slow map lookup must never delay a responder. The
budget is what actually enforces that rule; exceeding it degrades to "unknown",
which every caller already handles.
"""
NOMINATIM_USER_AGENT = "MedBridge-Healthcare/1.0 (+https://medbridge.health)"

REVERSE_GEOCODE_CACHE_SIZE = 512
FORWARD_GEOCODE_CACHE_SIZE = 256
PLACES_CACHE_SIZE = 128
"""
Coordinates and search terms repeat constantly — the same pharmacy, the same
patient address, the same hospital, the same city typed into an autocomplete —
so these caches remove most calls and keep us well inside the rate limits.
"""

# Addresses and hospital locations do not move. Long TTLs are both correct and
# what Nominatim's and Overpass's usage policies ask of heavy callers.
GEOCODE_CACHE_TTL_SECONDS = 60 * 60 * 24
PLACES_CACHE_TTL_SECONDS = 60 * 60 * 6


class _RateLimiter:
    """Serialises calls to a shared upstream with a minimum gap between them."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()


class _SingleFlightCache:
    """
    A bounded cache that also collapses concurrent misses into one request.

    A plain dict cache is only a cache under sequential access. Under load the
    interesting case is a hundred requests for the same key arriving while the
    first is still in flight: every one of them misses, every one calls
    upstream, and the cache fills with a hundred identical answers *after* the
    damage is done. Behind a rate limiter that turns into a queue that grows
    faster than it drains.

    Storing the in-flight task itself — not just the finished value — means
    later callers await the request already running. One upstream call per key,
    however many callers ask for it.
    """

    def __init__(self, capacity: int, ttl_seconds: float) -> None:
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._values: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()
        self._in_flight: dict[Any, asyncio.Task] = {}

    def _get_fresh(self, key: Any) -> tuple[bool, Any]:
        entry = self._values.get(key)
        if entry is None:
            return False, None
        stored_at, value = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._values[key]
            return False, None
        # Refresh recency so the eviction below drops the coldest key rather
        # than an entry that is merely old but still in constant use.
        self._values.move_to_end(key)
        return True, value

    def _store(self, key: Any, value: Any) -> None:
        self._values[key] = (time.monotonic(), value)
        self._values.move_to_end(key)
        while len(self._values) > self._capacity:
            self._values.popitem(last=False)

    async def get_or_load(self, key: Any, loader) -> Any:
        """
        The cached value for `key`, loading it at most once concurrently.

        Only successful loads are stored. Caching a failure would be far worse
        than not caching at all: one transient Overpass outage would answer
        "no hospitals nearby" for the next six hours, on the emergency path,
        with no way to tell it from the truth. A repeated miss is the cheap
        failure mode here.
        """
        hit, value = self._get_fresh(key)
        if hit:
            return value

        existing = self._in_flight.get(key)
        if existing is not None:
            # Shielded: this caller giving up (client disconnect, cancellation)
            # must not cancel the shared request every other caller is waiting on.
            return await asyncio.shield(existing)

        task = asyncio.ensure_future(loader())
        self._in_flight[key] = task
        try:
            value = await asyncio.shield(task)
        finally:
            # Only the owner clears the slot, and only once the task is settled,
            # so a late waiter never adopts a task that is about to be dropped.
            if self._in_flight.get(key) is task and task.done():
                self._in_flight.pop(key, None)
        if value is not None:
            self._store(key, value)
        return value

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


def _retry_after_seconds(exc: Exception) -> float:
    """
    How long to wait before retrying a shed-load response.

    Uses the server's own `Retry-After` when it sends one — it knows when the
    slot frees up better than a fixed constant does — and falls back to a
    plain delay otherwise. Capped either way: these lookups sit behind an
    emergency screen, where a stale answer beats a long stall.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("Retry-After") if response else None
    if header:
        try:
            return min(float(header), OVERPASS_BACKOFF_CAP_SECONDS)
        except (TypeError, ValueError):
            # Retry-After may be an HTTP date rather than seconds; not worth
            # parsing for a retry hint when a sane default is right there.
            pass
    return OVERPASS_BACKOFF_SECONDS


class MapsService:
    """Everything that needs mapping, and nothing that pretends to."""

    def __init__(self) -> None:
        self._warned = False
        self._rejections: set[tuple[str, str]] = set()
        self._nominatim = _RateLimiter(NOMINATIM_MIN_INTERVAL_SECONDS)
        self._overpass = _RateLimiter(OVERPASS_MIN_INTERVAL_SECONDS)
        self._reverse_cache = _SingleFlightCache(
            REVERSE_GEOCODE_CACHE_SIZE, GEOCODE_CACHE_TTL_SECONDS
        )
        self._forward_cache = _SingleFlightCache(
            FORWARD_GEOCODE_CACHE_SIZE, GEOCODE_CACHE_TTL_SECONDS
        )
        self._places_cache = _SingleFlightCache(
            PLACES_CACHE_SIZE, PLACES_CACHE_TTL_SECONDS
        )

    # ── availability ─────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        """
        Whether routing is configured *right now*.

        Read from settings on every call rather than cached, which is what
        makes activation a configuration change instead of a deployment.

        Reports on the ORS key specifically: geocoding and place search run
        keyless through OSM and keep working regardless, but distance and ETA
        are what every caller means when it asks whether maps are on.
        """
        return bool((settings.ORS_API_KEY or "").strip())

    def _unavailable(self, operation: str) -> None:
        """Log the first skip per process, then stay quiet."""
        if not self._warned:
            logger.warning(
                "[MAPS_DISABLED] ORS_API_KEY is not set; %s and every later "
                "routing lookup will be skipped. Coordinates, addresses and the "
                "plain map link continue to work.", operation,
            )
            self._warned = True

    def _reject_once(self, url: str, status: str, message: str) -> None:
        """
        Log a configuration-level rejection once per (endpoint, status).

        A key whose quota is exhausted is rejected identically forever, and one
        warning per emergency would bury the emergencies.
        """
        seen = (url, status)
        if seen not in self._rejections:
            self._rejections.add(seen)
            logger.warning("[MAPS_REJECTED] %s -> %s: %s", url, status, message[:300])

    # ── transport ────────────────────────────────────────────────────────

    async def _ors_post(
        self, url: str, payload: dict, *, accept: str = "application/json"
    ) -> Optional[dict]:
        """
        One ORS request, with every failure flattened to None.

        A network error, a timeout, a quota rejection and a malformed body are
        the same thing to every caller here: no data. None of them may raise
        into an emergency being raised.
        """
        if not self.is_enabled():
            return None

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": settings.ORS_API_KEY,
                        "Content-Type": "application/json",
                        # The /geojson endpoint rejects `application/json` with
                        # a 406, so the caller states what it expects.
                        "Accept": accept,
                    },
                    json=payload,
                )
        except Exception as exc:
            logger.warning("[MAPS_REQUEST_FAILED] %s: %s", url, exc)
            return None

        if response.status_code >= 400:
            # 401/403 is a bad key, 429 is quota. Both are configuration
            # problems worth seeing, and ORS names the fix in its own message.
            self._reject_once(url, str(response.status_code), response.text)
            return None

        try:
            return response.json()
        except Exception as exc:
            logger.warning("[MAPS_BAD_BODY] %s: %s", url, exc)
            return None

    async def _osm_get(self, url: str, params: dict) -> Optional[Any]:
        """
        One keyless OSM request, rate limited and failure-flattened.

        The limiter is shared across the process, so concurrent callers queue
        rather than tripping Nominatim's policy and getting the whole platform
        blocked.
        """
        await self._nominatim.wait()
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={
                        "User-Agent": NOMINATIM_USER_AGENT,
                        "Accept": "application/json",
                    },
                )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("[MAPS_OSM_FAILED] %s: %s", url, exc)
            return None

    # ── links (no key required) ──────────────────────────────────────────

    def build_maps_url(self, latitude: float, longitude: float) -> str:
        """A link to the coordinates on OpenStreetMap. Works with or without a key."""
        return OSM_SEARCH_TEMPLATE.format(lat=latitude, lng=longitude)

    def build_directions_url(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> str:
        """Driving directions between two points on OSM. Also key-free."""
        return OSM_DIRECTIONS_TEMPLATE.format(
            o_lat=origin[0], o_lng=origin[1],
            d_lat=destination[0], d_lng=destination[1],
        )

    # ── geocoding ────────────────────────────────────────────────────────

    async def reverse_geocode(
        self, latitude: float, longitude: float
    ) -> Optional[str]:
        """
        The street address for a coordinate pair, or None.

        Keyless — this keeps working even with no ORS key, which is a strict
        improvement on the Google implementation where an unset key meant no
        address at all.
        """
        # Rounded to ~11 metres before caching. Finer precision would make
        # every GPS jitter a cache miss without changing the answer.
        cache_key = (round(latitude, 4), round(longitude, 4))

        async def load() -> Optional[str]:
            body = await self._osm_get(
                NOMINATIM_REVERSE_URL,
                {
                    "lat": latitude,
                    "lon": longitude,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "zoom": 18,
                },
            )
            if isinstance(body, dict):
                return body.get("display_name") or None
            return None

        return await self._reverse_cache.get_or_load(cache_key, load)

    async def forward_geocode(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Search for a place by name or address.

        Powers address autocomplete. Returns an empty list on failure, never
        an exception, so a typeahead cannot break the form it sits in.
        """
        cleaned = (query or "").strip()
        if len(cleaned) < 3:
            # Below three characters Nominatim returns noise, and querying on
            # every keystroke is exactly what its policy asks callers not to do.
            return []

        # Autocomplete is the highest-volume path on the platform: every user,
        # every keystroke past the debounce. The frontend's React Query cache is
        # per-browser and does nothing about the same city being typed by a
        # hundred different people, so the deduplication has to live here.
        async def load() -> Optional[list]:
            return await self._osm_get(
                NOMINATIM_SEARCH_URL,
                {"q": cleaned, "format": "jsonv2", "limit": limit, "addressdetails": 1},
            )

        body = await self._forward_cache.get_or_load((cleaned.lower(), limit), load)
        if not isinstance(body, list):
            return []

        results: list[dict[str, Any]] = []
        for item in body:
            try:
                results.append(
                    {
                        "display_name": item.get("display_name", ""),
                        "latitude": float(item["lat"]),
                        "longitude": float(item["lon"]),
                        "type": item.get("type"),
                        "importance": item.get("importance", 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return results

    # ── place search ─────────────────────────────────────────────────────

    async def find_nearby_hospitals(
        self,
        latitude: float,
        longitude: float,
        radius_metres: int = DEFAULT_SEARCH_RADIUS_METRES,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Hospitals near a point, nearest first, or an empty list.

        An empty list means "we do not know", never "there are none nearby" —
        callers must not present it as the latter. Contract unchanged from the
        Google implementation; the keys of each dict are identical.

        Backed by Overpass, which queries OSM's own hospital tagging. Unlike
        Places it needs no key, so this now works in environments where the
        Google version returned nothing.
        """
        query = f"""
        [out:json][timeout:{OVERPASS_QUERY_BUDGET_SECONDS}];
        (
          node["amenity"="hospital"](around:{radius_metres},{latitude},{longitude});
          way["amenity"="hospital"](around:{radius_metres},{latitude},{longitude});
        );
        out center {max(limit * 4, 20)};
        """

        async def fetch() -> Optional[dict[str, Any]]:
            for attempt in range(1, OVERPASS_ATTEMPTS + 1):
                await self._overpass.wait()
                try:
                    async with httpx.AsyncClient(
                        timeout=OVERPASS_TIMEOUT_SECONDS
                    ) as client:
                        response = await client.post(
                            OVERPASS_URL,
                            data={"data": query},
                            headers={"User-Agent": NOMINATIM_USER_AGENT},
                        )
                    response.raise_for_status()
                    return response.json()
                except asyncio.CancelledError:
                    # The budget expired. Not a fault, and not something to
                    # retry — let it unwind.
                    raise
                except Exception as exc:
                    # The exception type carries the diagnosis here: a shed-load
                    # timeout stringifies to nothing, so logging `exc` alone
                    # leaves an empty, unactionable line.
                    logger.warning(
                        "[MAPS_OVERPASS_FAILED] attempt %d/%d: %s: %s",
                        attempt,
                        OVERPASS_ATTEMPTS,
                        type(exc).__name__,
                        exc,
                    )
                    if attempt < OVERPASS_ATTEMPTS:
                        # Overpass sheds load with 429 and 504. Retrying
                        # instantly is what earns the next 429, so back off —
                        # honouring Retry-After when the server states one.
                        await asyncio.sleep(_retry_after_seconds(exc))
            return None

        # Cached on the rounded query so a street of patients raising SOS in the
        # same area does not re-ask Overpass for the same hospitals each time.
        cache_key = (round(latitude, 3), round(longitude, 3), radius_metres)

        async def load() -> Optional[dict[str, Any]]:
            try:
                return await asyncio.wait_for(
                    fetch(), timeout=OVERPASS_TOTAL_BUDGET_SECONDS
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "[MAPS_OVERPASS_BUDGET] exceeded %.0fs; degrading to unknown",
                    OVERPASS_TOTAL_BUDGET_SECONDS,
                )
                return None

        body = await self._places_cache.get_or_load(cache_key, load)

        if body is None:
            return []

        from app.pharmacy.domain.entities import haversine_km

        hospitals: list[dict[str, Any]] = []
        for element in body.get("elements") or []:
            centre = element.get("center") or element
            lat, lon = centre.get("lat"), centre.get("lon")
            if lat is None or lon is None:
                continue
            tags = element.get("tags") or {}
            hospitals.append(
                {
                    "place_id": f"osm:{element.get('type')}:{element.get('id')}",
                    "name": tags.get("name") or "Hospital",
                    "address": tags.get("addr:full")
                    or ", ".join(
                        filter(None, [tags.get("addr:street"), tags.get("addr:city")])
                    )
                    or None,
                    "latitude": lat,
                    "longitude": lon,
                    # Overpass carries no rating or live opening state. Reported
                    # as None rather than invented — the Google shape had these
                    # keys and callers already tolerate nulls.
                    "rating": None,
                    "open_now": None,
                    "_distance_km": haversine_km(latitude, longitude, lat, lon),
                }
            )

        # Overpass returns in arbitrary order; the contract promises nearest first.
        hospitals.sort(key=lambda h: h["_distance_km"])
        for hospital in hospitals:
            hospital.pop("_distance_km", None)
        return hospitals[:limit]

    # ── routing ──────────────────────────────────────────────────────────

    async def distance_matrix(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> Optional[dict[str, Any]]:
        """
        Driving distance and duration between two points, or None.

        Returns kilometres and whole minutes — the units the emergency record
        stores — so no caller has to convert and get it wrong. Keys are
        identical to the Google implementation.
        """
        if not self.is_enabled():
            self._unavailable("distance and ETA")
            return None

        # ORS takes longitude first. Getting this backwards silently returns a
        # route across the wrong hemisphere rather than an error.
        body = await self._ors_post(
            ORS_MATRIX_URL,
            {
                "locations": [
                    [origin[1], origin[0]],
                    [destination[1], destination[0]],
                ],
                "metrics": ["distance", "duration"],
                "units": "m",
            },
        )
        if not body:
            return None

        try:
            metres = body["distances"][0][1]
            seconds = body["durations"][0][1]
        except (KeyError, IndexError, TypeError):
            return None
        if metres is None or seconds is None:
            return None

        km = round(metres / 1000.0, 2)
        minutes = max(1, round(seconds / 60.0))
        return {
            "distance_km": km,
            "duration_minutes": minutes,
            "distance_text": f"{km} km",
            "duration_text": f"{minutes} min",
        }

    async def distance_matrix_many(
        self,
        origin: tuple[float, float],
        destinations: list[tuple[float, float]],
    ) -> list[Optional[dict[str, Any]]]:
        """
        One origin against many destinations, in a single request.

        This is what a matrix API is for. Asking it for one pair at a time —
        which is what looping over `distance_matrix` does — costs one HTTP
        round trip and one quota unit per destination to obtain exactly the
        same numbers.

        Returns a list positionally aligned with `destinations`, with None in
        any slot ORS could not route to. Entry shape is identical to
        `distance_matrix`, so callers format results the same way.
        """
        if not destinations:
            return []
        if not self.is_enabled():
            self._unavailable("distance and ETA")
            return [None] * len(destinations)

        body = await self._ors_post(
            ORS_MATRIX_URL,
            {
                # Index 0 is the origin; every later index is a destination.
                "locations": [[origin[1], origin[0]]]
                + [[lat_lng[1], lat_lng[0]] for lat_lng in destinations],
                "sources": [0],
                "destinations": list(range(1, len(destinations) + 1)),
                "metrics": ["distance", "duration"],
                "units": "m",
            },
        )
        if not body:
            return [None] * len(destinations)

        try:
            distances = body["distances"][0]
            durations = body["durations"][0]
        except (KeyError, IndexError, TypeError):
            return [None] * len(destinations)

        results: list[Optional[dict[str, Any]]] = []
        for index in range(len(destinations)):
            try:
                metres = distances[index]
                seconds = durations[index]
            except (IndexError, TypeError):
                results.append(None)
                continue
            if metres is None or seconds is None:
                # ORS returns null for a destination it cannot reach by road.
                results.append(None)
                continue
            km = round(metres / 1000.0, 2)
            minutes = max(1, round(seconds / 60.0))
            results.append(
                {
                    "distance_km": km,
                    "duration_minutes": minutes,
                    "distance_text": f"{km} km",
                    "duration_text": f"{minutes} min",
                }
            )
        return results

    async def route_geometry(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> Optional[dict[str, Any]]:
        """
        The driving route between two points, as coordinates for a polyline.

        New capability — Google Directions was never wired in. Returns
        `[[lat, lng], …]` in Leaflet's order so the client can draw it without
        transforming anything.
        """
        if not self.is_enabled():
            self._unavailable("route geometry")
            return None

        body = await self._ors_post(
            ORS_DIRECTIONS_URL,
            {
                "coordinates": [
                    [origin[1], origin[0]],
                    [destination[1], destination[0]],
                ],
                "instructions": False,
            },
            accept="application/geo+json",
        )
        if not body:
            return None

        try:
            feature = body["features"][0]
            coordinates = feature["geometry"]["coordinates"]
            summary = feature["properties"]["summary"]
        except (KeyError, IndexError, TypeError):
            return None

        metres = summary.get("distance")
        seconds = summary.get("duration")
        return {
            # ORS emits [lng, lat]; Leaflet wants [lat, lng].
            "polyline": [[point[1], point[0]] for point in coordinates],
            "distance_km": round((metres or 0) / 1000.0, 2),
            "duration_minutes": max(1, round((seconds or 0) / 60.0)),
        }

    async def nearest_hospital_with_eta(
        self, latitude: float, longitude: float
    ) -> Optional[dict[str, Any]]:
        """
        The nearest hospital and how long it takes to reach it, or None.

        Two lookups rather than one because Overpass ranks by straight-line
        distance, not by driving time; the matrix is what turns a list into an
        answer. If the second call fails the hospital is still returned, with
        the ETA left null — a known facility with an unknown ETA is useful, an
        invented ETA is not. Behaviour identical to the Google version.
        """
        hospitals = await self.find_nearby_hospitals(latitude, longitude, limit=5)
        if not hospitals:
            return None

        # One matrix request covering every candidate, rather than one request
        # per candidate. Same numbers, same ranking, a fifth of the latency on
        # a path where the caller is raising an emergency.
        all_metrics = await self.distance_matrix_many(
            (latitude, longitude),
            [(hospital["latitude"], hospital["longitude"]) for hospital in hospitals],
        )

        best: Optional[dict[str, Any]] = None
        for hospital, metrics in zip(hospitals, all_metrics):
            candidate = {**hospital, **(metrics or {})}
            if metrics is None:
                best = best or candidate
                continue
            if best is None or metrics["duration_minutes"] < best.get(
                "duration_minutes", float("inf")
            ):
                best = candidate
        return best


_service: MapsService | None = None


def get_maps_service() -> MapsService:
    """The process-wide service. Replaceable in tests."""
    global _service
    if _service is None:
        _service = MapsService()
    return _service


def set_maps_service(service: MapsService | None) -> None:
    global _service
    _service = service
