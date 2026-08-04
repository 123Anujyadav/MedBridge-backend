"""
Patient-facing nearby hospital search.

The emergency screen calls this while someone is having an emergency, so the
behaviour that matters most is what happens when the upstreams are unhappy:
it must answer, quickly, with something the patient can act on, and it must
never surface an internal error or invent a facility.

Every upstream is stubbed. The live integrations are covered in
test_maps_hardening.py; this file is about the endpoint contract.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient
from app.models.user import User
from app.services.maps import get_maps_service

PW = "HospTest#2026"
URL = "/api/v1/patient/hospitals/nearby"
DELHI = {"latitude": 28.6315, "longitude": 77.2167}

HOSPITALS = [
    {
        "place_id": "osm:node:1",
        "name": "Lady Hardinge Medical College",
        "address": "Shaheed Bhagat Singh Marg, New Delhi",
        "latitude": 28.6362,
        "longitude": 77.2094,
        "rating": None,
        "open_now": None,
    },
    {
        "place_id": "osm:node:2",
        "name": "Delhi Heart and Lung Institute",
        "address": None,
        "latitude": 28.6402,
        "longitude": 77.2010,
        "rating": None,
        "open_now": None,
    },
]

METRICS = [
    {
        "distance_km": 2.37,
        "duration_minutes": 3,
        "distance_text": "2.37 km",
        "duration_text": "3 min",
    },
    {
        "distance_km": 4.1,
        "duration_minutes": 9,
        "distance_text": "4.1 km",
        "duration_text": "9 min",
    },
]


@pytest.fixture
async def patient_email(db: AsyncSession) -> str:
    from app.core.security import get_password_hash

    email = f"hosp-{uuid.uuid4().hex[:8]}@example.com"
    user = User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=get_password_hash(PW),
        role="patient",
        is_active=True,
    )
    db.add(user)
    await db.flush()
    db.add(
        Patient(
            id=user.id,
            first_name="Hosp",
            last_name="Tester",
            phone="+911234500111",
            date_of_birth="1990-01-01",
            gender="other",
        )
    )
    await db.commit()
    return email


async def auth(client: AsyncClient, email: str) -> dict:
    from conftest import login_payload

    response = await client.post(
        "/api/v1/auth/login", json=await login_payload(email, PW)
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


class TestAccessControl:
    async def test_requires_authentication(self, client):
        assert (await client.get(URL, params=DELHI)).status_code == 401

    async def test_an_anonymous_caller_never_reaches_the_upstreams(self, client):
        maps = get_maps_service()
        with patch.object(maps, "find_nearby_hospitals") as overpass:
            await client.get(URL, params=DELHI)
        overpass.assert_not_called()


class TestValidation:
    @pytest.mark.parametrize(
        "params",
        [
            {"latitude": 91.0, "longitude": 77.2},
            {"latitude": -91.0, "longitude": 77.2},
            {"latitude": 28.6, "longitude": 181.0},
            {"latitude": 28.6, "longitude": -181.0},
        ],
    )
    async def test_out_of_range_coordinates_are_rejected(
        self, client, patient_email, params
    ):
        headers = await auth(client, patient_email)
        assert (await client.get(URL, params=params, headers=headers)).status_code == 422

    async def test_the_limit_is_capped(self, client, patient_email):
        headers = await auth(client, patient_email)
        response = await client.get(
            URL, params={**DELHI, "limit": 500}, headers=headers
        )
        assert response.status_code == 422


class TestResults:
    async def test_returns_hospitals_with_distance_eta_and_navigation(
        self, client, patient_email
    ):
        headers = await auth(client, patient_email)
        maps = get_maps_service()

        with patch.object(maps, "find_nearby_hospitals", return_value=HOSPITALS):
            with patch.object(maps, "distance_matrix_many", return_value=METRICS):
                response = await client.get(URL, params=DELHI, headers=headers)

        assert response.status_code == 200
        body = response.json()
        assert body["available"] is True
        assert len(body["hospitals"]) == 2

        first = body["hospitals"][0]
        assert first["name"] == "Lady Hardinge Medical College"
        assert first["distance_km"] == 2.37
        assert first["eta_minutes"] == 3
        assert first["duration_text"] == "3 min"
        # Built by the shared helper, not assembled in the endpoint.
        assert first["directions_url"].startswith("https://www.openstreetmap.org")
        assert "28.6315" in first["directions_url"]

    async def test_one_matrix_request_covers_every_candidate(
        self, client, patient_email
    ):
        """Five candidates must not cost five routing requests."""
        headers = await auth(client, patient_email)
        maps = get_maps_service()

        with patch.object(maps, "find_nearby_hospitals", return_value=HOSPITALS):
            with patch.object(
                maps, "distance_matrix_many", return_value=METRICS
            ) as batched:
                with patch.object(maps, "distance_matrix") as single:
                    await client.get(URL, params=DELHI, headers=headers)

        assert batched.call_count == 1
        single.assert_not_called()

    async def test_a_hospital_without_routing_keeps_its_address(
        self, client, patient_email
    ):
        """A known facility with an unknown ETA is useful; an invented one is not."""
        headers = await auth(client, patient_email)
        maps = get_maps_service()

        with patch.object(maps, "find_nearby_hospitals", return_value=HOSPITALS):
            with patch.object(maps, "distance_matrix_many", return_value=[None, None]):
                response = await client.get(URL, params=DELHI, headers=headers)

        body = response.json()
        assert body["available"] is True
        for hospital in body["hospitals"]:
            assert hospital["distance_km"] is None
            assert hospital["eta_minutes"] is None
            assert hospital["directions_url"]
        assert body["hospitals"][0]["name"] == "Lady Hardinge Medical College"


class TestDegradation:
    async def test_overpass_outage_is_not_a_server_error(self, client, patient_email):
        headers = await auth(client, patient_email)
        maps = get_maps_service()

        with patch.object(maps, "find_nearby_hospitals", return_value=[]):
            response = await client.get(URL, params=DELHI, headers=headers)

        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert body["reason"]
        assert body["hospitals"] == []

    async def test_no_internal_detail_reaches_the_patient(self, client, patient_email):
        """The reason is for a phone screen during an emergency, not a log."""
        headers = await auth(client, patient_email)
        maps = get_maps_service()

        with patch.object(maps, "find_nearby_hospitals", return_value=[]):
            response = await client.get(URL, params=DELHI, headers=headers)

        reason = response.json()["reason"].lower()
        for leak in (
            "overpass",
            "openrouteservice",
            "nominatim",
            "traceback",
            "http",
            "timeout",
            "api key",
            "ors",
        ):
            assert leak not in reason, f"{leak!r} leaked into a patient-facing message"

    async def test_routing_outage_still_returns_the_facilities(
        self, client, patient_email
    ):
        headers = await auth(client, patient_email)
        maps = get_maps_service()

        with patch.object(maps, "find_nearby_hospitals", return_value=HOSPITALS):
            with patch.object(maps, "distance_matrix_many", return_value=[]):
                response = await client.get(URL, params=DELHI, headers=headers)

        assert response.status_code == 200
        assert response.json()["available"] is True

    async def test_an_unexpected_upstream_error_does_not_500(
        self, client, patient_email
    ):
        """Whatever Overpass does, this route answers."""
        headers = await auth(client, patient_email)
        maps = get_maps_service()

        with patch.object(
            maps, "find_nearby_hospitals", side_effect=RuntimeError("boom")
        ):
            response = await client.get(URL, params=DELHI, headers=headers)

        assert response.status_code != 500, response.text
