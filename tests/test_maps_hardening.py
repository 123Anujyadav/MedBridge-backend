"""
Production-hardening invariants for the OSM/ORS mapping stack.

These guard defects found during production-readiness verification. Each one
was a real failure, not a hypothetical, so each keeps its own test:

* concurrent misses stampeded straight past the cache to the upstream;
* forward geocoding — the highest-volume path — had no server-side cache;
* a retrying Overpass call had no overall deadline, on the emergency path;
* the nearest-hospital lookup spent one matrix request per candidate.

No network access: every upstream is stubbed. A test that needs the real
services to be up is not a regression test.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import app.services.maps as maps_module
from app.services.maps import MapsService, _SingleFlightCache


@pytest.fixture
def service() -> MapsService:
    return MapsService()


class TestSingleFlightCache:
    async def test_concurrent_misses_collapse_into_one_load(self):
        cache = _SingleFlightCache(capacity=16, ttl_seconds=60)
        calls = {"n": 0}

        async def load():
            calls["n"] += 1
            await asyncio.sleep(0.02)
            return "value"

        results = await asyncio.gather(*(cache.get_or_load("k", load) for _ in range(50)))

        assert results == ["value"] * 50
        assert calls["n"] == 1, "50 concurrent callers must share one upstream request"

    async def test_failures_are_not_cached(self):
        """A cached failure would outlive the outage that caused it."""
        cache = _SingleFlightCache(capacity=16, ttl_seconds=3600)
        state = {"fail": True}

        async def load():
            return None if state["fail"] else "recovered"

        assert await cache.get_or_load("k", load) is None
        state["fail"] = False
        assert await cache.get_or_load("k", load) == "recovered"

    async def test_entries_are_evicted_once_capacity_is_reached(self):
        cache = _SingleFlightCache(capacity=10, ttl_seconds=3600)

        for index in range(500):
            await cache.get_or_load(index, _returning(index))

        assert len(cache) <= 10

    async def test_entries_expire(self):
        cache = _SingleFlightCache(capacity=10, ttl_seconds=0.05)
        calls = {"n": 0}

        async def load():
            calls["n"] += 1
            return "v"

        await cache.get_or_load("k", load)
        await cache.get_or_load("k", load)
        await asyncio.sleep(0.08)
        await cache.get_or_load("k", load)

        assert calls["n"] == 2

    async def test_one_caller_cancelling_does_not_cancel_the_others(self):
        """A disconnecting client must not abort the request others await."""
        cache = _SingleFlightCache(capacity=10, ttl_seconds=60)
        calls = {"n": 0}

        async def load():
            calls["n"] += 1
            await asyncio.sleep(0.1)
            return "value"

        first = asyncio.ensure_future(cache.get_or_load("k", load))
        await asyncio.sleep(0.01)
        second = asyncio.ensure_future(cache.get_or_load("k", load))
        await asyncio.sleep(0.01)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert await second == "value"
        assert calls["n"] == 1


def _returning(value):
    async def loader():
        return value

    return loader


class TestGeocodeCaching:
    async def test_reverse_geocode_deduplicates_concurrent_callers(self, service):
        calls = {"n": 0}

        async def fake_get(url, params):
            calls["n"] += 1
            await asyncio.sleep(0.01)
            return {"display_name": "Connaught Place, New Delhi"}

        with patch.object(service, "_osm_get", side_effect=fake_get):
            results = await asyncio.gather(
                *(service.reverse_geocode(28.6315, 77.2167) for _ in range(40))
            )

        assert all(r == "Connaught Place, New Delhi" for r in results)
        assert calls["n"] == 1

    async def test_gps_jitter_does_not_defeat_the_cache(self, service):
        """A stationary rider's coordinates wobble; the address does not."""
        calls = {"n": 0}

        async def fake_get(url, params):
            calls["n"] += 1
            return {"display_name": "Somewhere"}

        # Jitter oscillates around a fix rather than drifting away from it, so
        # the readings stay inside one rounding bucket — which is the case the
        # 4-decimal-place key exists to absorb.
        with patch.object(service, "_osm_get", side_effect=fake_get):
            for tick in range(60):
                wobble = (tick % 5) * 0.000001
                await service.reverse_geocode(28.61290 + wobble, 77.22950 + wobble)

        assert calls["n"] == 1

    async def test_forward_geocode_is_cached_across_callers(self, service):
        calls = {"n": 0}

        async def fake_get(url, params):
            calls["n"] += 1
            return [{"display_name": "Delhi", "lat": "28.6", "lon": "77.2"}]

        with patch.object(service, "_osm_get", side_effect=fake_get):
            for _ in range(25):
                results = await service.forward_geocode("Delhi")

        assert calls["n"] == 1
        assert results[0]["latitude"] == 28.6

    async def test_forward_geocode_still_rejects_short_queries(self, service):
        with patch.object(service, "_osm_get") as osm:
            assert await service.forward_geocode("de") == []
        osm.assert_not_called()


class TestOverpassResilience:
    async def test_a_hanging_upstream_cannot_outlast_the_budget(self, service, monkeypatch):
        """The emergency path must degrade, never stall."""
        monkeypatch.setattr(maps_module, "OVERPASS_TOTAL_BUDGET_SECONDS", 0.3)

        class Hanging:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, *args, **kwargs):
                await asyncio.sleep(60)

        with patch.object(maps_module.httpx, "AsyncClient", Hanging):
            result = await asyncio.wait_for(
                service.find_nearby_hospitals(28.6, 77.2), timeout=5
            )

        assert result == []

    async def test_a_transient_outage_is_not_cached_as_no_hospitals(
        self, service, monkeypatch
    ):
        """The defect this guards would answer 'none nearby' for hours."""
        monkeypatch.setattr(maps_module, "OVERPASS_BACKOFF_SECONDS", 0.0)
        state = {"fail": True}

        class Flaky:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, *args, **kwargs):
                if state["fail"]:
                    raise ConnectionError("upstream down")
                return _OverpassResponse()

        with patch.object(maps_module.httpx, "AsyncClient", Flaky):
            during_outage = await service.find_nearby_hospitals(28.6, 77.2)
            state["fail"] = False
            after_recovery = await service.find_nearby_hospitals(28.6, 77.2)

        assert during_outage == []
        assert [h["name"] for h in after_recovery] == ["Recovered Hospital"]

    async def test_identical_searches_share_one_overpass_query(self, service):
        calls = {"n": 0}

        class Counting:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, *args, **kwargs):
                calls["n"] += 1
                return _OverpassResponse()

        with patch.object(maps_module.httpx, "AsyncClient", Counting):
            await asyncio.gather(
                *(service.find_nearby_hospitals(28.6, 77.2) for _ in range(30))
            )

        assert calls["n"] == 1

    @pytest.mark.parametrize("status", [429, 502, 504])
    async def test_shed_load_responses_degrade_quietly(
        self, service, monkeypatch, status
    ):
        monkeypatch.setattr(maps_module, "OVERPASS_BACKOFF_SECONDS", 0.0)

        class Rejecting:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, *args, **kwargs):
                return _OverpassResponse(status=status)

        with patch.object(maps_module.httpx, "AsyncClient", Rejecting):
            assert await service.find_nearby_hospitals(28.6, 77.2) == []


class _OverpassResponse:
    def __init__(self, status: int = 200):
        self.status_code = status
        self.headers: dict[str, str] = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self
            )

    def json(self):
        return {
            "elements": [
                {
                    "lat": 28.61,
                    "lon": 77.21,
                    "tags": {"name": "Recovered Hospital", "amenity": "hospital"},
                }
            ]
        }


class TestBatchedMatrix:
    async def test_many_destinations_cost_one_request(self, service, monkeypatch):
        monkeypatch.setattr(maps_module.settings, "ORS_API_KEY", "test-key")
        posted = []

        async def fake_post(url, payload, accept="application/json"):
            posted.append(payload)
            count = len(payload["destinations"])
            return {
                "distances": [[1000.0 * (i + 1) for i in range(count)]],
                "durations": [[60.0 * (i + 1) for i in range(count)]],
            }

        with patch.object(service, "_ors_post", side_effect=fake_post):
            results = await service.distance_matrix_many(
                (28.6, 77.2), [(28.61, 77.21), (28.62, 77.22), (28.63, 77.23)]
            )

        assert len(posted) == 1
        assert [r["duration_text"] for r in results] == ["1 min", "2 min", "3 min"]
        assert [r["distance_km"] for r in results] == [1.0, 2.0, 3.0]

    async def test_an_unroutable_destination_is_none_not_zero(
        self, service, monkeypatch
    ):
        """ORS returns null for a destination it cannot reach. 0 km would be a lie."""
        monkeypatch.setattr(maps_module.settings, "ORS_API_KEY", "test-key")

        async def fake_post(url, payload, accept="application/json"):
            return {"distances": [[1000.0, None]], "durations": [[60.0, None]]}

        with patch.object(service, "_ors_post", side_effect=fake_post):
            results = await service.distance_matrix_many(
                (28.6, 77.2), [(28.61, 77.21), (0.0, 0.0)]
            )

        assert results[0]["distance_km"] == 1.0
        assert results[1] is None

    async def test_no_key_yields_one_none_per_destination(self, service, monkeypatch):
        monkeypatch.setattr(maps_module.settings, "ORS_API_KEY", "")

        results = await service.distance_matrix_many(
            (28.6, 77.2), [(28.61, 77.21), (28.62, 77.22)]
        )

        assert results == [None, None]

    async def test_empty_destinations_makes_no_request(self, service):
        with patch.object(service, "_ors_post") as post:
            assert await service.distance_matrix_many((28.6, 77.2), []) == []
        post.assert_not_called()

    async def test_nearest_hospital_uses_a_single_matrix_request(
        self, service, monkeypatch
    ):
        """Five candidates used to cost five requests on the emergency path."""
        monkeypatch.setattr(maps_module.settings, "ORS_API_KEY", "test-key")

        hospitals = [
            {"name": f"Hospital {i}", "latitude": 28.61 + i * 0.01, "longitude": 77.21}
            for i in range(5)
        ]
        posted = []

        async def fake_post(url, payload, accept="application/json"):
            posted.append(payload)
            # Hospital 3 is the quickest by road despite not being the closest.
            return {
                "distances": [[5000.0, 4000.0, 3000.0, 2000.0, 6000.0]],
                "durations": [[600.0, 480.0, 360.0, 120.0, 720.0]],
            }

        with patch.object(service, "find_nearby_hospitals", return_value=hospitals):
            with patch.object(service, "_ors_post", side_effect=fake_post):
                best = await service.nearest_hospital_with_eta(28.6, 77.2)

        assert len(posted) == 1, "one request must cover every candidate"
        assert best["name"] == "Hospital 3"
        assert best["duration_minutes"] == 2

    async def test_hospital_is_still_returned_when_routing_fails(
        self, service, monkeypatch
    ):
        """A known facility with an unknown ETA beats no answer at all."""
        monkeypatch.setattr(maps_module.settings, "ORS_API_KEY", "test-key")
        hospitals = [{"name": "Only", "latitude": 28.61, "longitude": 77.21}]

        with patch.object(service, "find_nearby_hospitals", return_value=hospitals):
            with patch.object(service, "_ors_post", return_value=None):
                best = await service.nearest_hospital_with_eta(28.6, 77.2)

        assert best["name"] == "Only"
        assert "duration_minutes" not in best


class TestSecrets:
    def test_the_key_is_never_logged(self, service, monkeypatch, caplog):
        monkeypatch.setattr(maps_module.settings, "ORS_API_KEY", "super-secret-key")
        service._reject_once(maps_module.ORS_MATRIX_URL, "401", "Invalid API key")
        service._unavailable("distance and ETA")

        assert "super-secret-key" not in caplog.text

    async def test_the_key_is_sent_as_a_header_not_a_query_parameter(
        self, service, monkeypatch
    ):
        """A key in a URL leaks into access logs, proxies and referrers."""
        monkeypatch.setattr(maps_module.settings, "ORS_API_KEY", "super-secret-key")
        seen = {}

        class Capturing:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, headers=None, json=None, **kwargs):
                seen["url"] = url
                seen["headers"] = headers

                class R:
                    status_code = 200

                    def json(self):
                        return {"distances": [[0, 1000]], "durations": [[0, 60]]}

                return R()

        with patch.object(maps_module.httpx, "AsyncClient", Capturing):
            await service.distance_matrix((28.6, 77.2), (28.61, 77.21))

        assert "super-secret-key" not in seen["url"]
        assert seen["headers"]["Authorization"] == "super-secret-key"
