"""
HTTP-level tests for the Nominatim geocoding proxy.

These two routes exist because Nominatim's usage policy requires an
identifying User-Agent and a request rate the browser cannot be trusted to
honour. Proxying it earns an obligation: the endpoints must not become an
open relay that lets anyone use MedBridge's server — and MedBridge's IP
reputation — to hammer a free public service.

The service layer is covered in test_maps_hardening.py; this file is about the
boundary — authentication, input bounds, and never leaking upstream failures
as server errors.
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

PW = "GeoTest#2026"
SEARCH = "/api/v1/pharmacy/geocode/search"
REVERSE = "/api/v1/pharmacy/geocode/reverse"


@pytest.fixture
async def patient_email(db: AsyncSession) -> str:
    from app.core.security import get_password_hash

    email = f"geo-{uuid.uuid4().hex[:8]}@example.com"
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
            first_name="Geo",
            last_name="Tester",
            phone="+911234500000",
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


class TestNotAnOpenProxy:
    async def test_search_requires_authentication(self, client):
        assert (await client.get(SEARCH, params={"q": "Delhi"})).status_code == 401

    async def test_reverse_requires_authentication(self, client):
        response = await client.get(
            REVERSE, params={"latitude": 28.6, "longitude": 77.2}
        )
        assert response.status_code == 401

    async def test_an_anonymous_caller_never_reaches_nominatim(self, client):
        """Rejection must happen before any upstream request is made."""
        with patch.object(get_maps_service(), "_osm_get") as osm:
            await client.get(SEARCH, params={"q": "Delhi"})
            await client.get(REVERSE, params={"latitude": 28.6, "longitude": 77.2})
        osm.assert_not_called()


class TestInputBounds:
    async def test_short_queries_are_rejected_before_the_upstream(
        self, client, patient_email
    ):
        headers = await auth(client, patient_email)
        with patch.object(get_maps_service(), "_osm_get") as osm:
            response = await client.get(SEARCH, params={"q": "de"}, headers=headers)
        assert response.status_code == 422
        osm.assert_not_called()

    @pytest.mark.parametrize(
        "latitude,longitude",
        [(91.0, 77.2), (-91.0, 77.2), (28.6, 181.0), (28.6, -181.0)],
    )
    async def test_out_of_range_coordinates_are_rejected(
        self, client, patient_email, latitude, longitude
    ):
        headers = await auth(client, patient_email)
        response = await client.get(
            REVERSE,
            params={"latitude": latitude, "longitude": longitude},
            headers=headers,
        )
        assert response.status_code == 422

    async def test_the_result_limit_is_capped(self, client, patient_email):
        """An uncapped limit would let one caller pull large result sets."""
        headers = await auth(client, patient_email)
        response = await client.get(
            SEARCH, params={"q": "Delhi", "limit": 500}, headers=headers
        )
        assert response.status_code == 422


class TestResponses:
    async def test_search_returns_normalised_results(self, client, patient_email):
        headers = await auth(client, patient_email)
        upstream = [
            {
                "display_name": "Connaught Place, New Delhi",
                "lat": "28.6315",
                "lon": "77.2167",
                "type": "suburb",
                "importance": 0.6,
            }
        ]
        with patch.object(get_maps_service(), "_osm_get", return_value=upstream):
            response = await client.get(
                SEARCH, params={"q": "Connaught Place"}, headers=headers
            )

        assert response.status_code == 200
        body = response.json()
        assert body[0]["latitude"] == 28.6315
        assert body[0]["longitude"] == 77.2167
        assert body[0]["display_name"] == "Connaught Place, New Delhi"

    async def test_reverse_returns_the_address(self, client, patient_email):
        headers = await auth(client, patient_email)
        upstream = {"display_name": "Outer Circle, Connaught Place, New Delhi"}
        with patch.object(get_maps_service(), "_osm_get", return_value=upstream):
            response = await client.get(
                REVERSE,
                params={"latitude": 28.6315, "longitude": 77.2167},
                headers=headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["address"] == "Outer Circle, Connaught Place, New Delhi"
        assert body["latitude"] == 28.6315

    async def test_an_upstream_outage_is_not_a_server_error(
        self, client, patient_email
    ):
        """
        Nominatim being down is not MedBridge failing. A 5xx here would trip
        alerting and error boundaries over a degraded optional lookup.
        """
        headers = await auth(client, patient_email)
        with patch.object(get_maps_service(), "_osm_get", return_value=None):
            search = await client.get(SEARCH, params={"q": "Delhi"}, headers=headers)
            reverse = await client.get(
                REVERSE,
                params={"latitude": 19.076, "longitude": 72.8777},
                headers=headers,
            )

        assert search.status_code == 200
        assert search.json() == []
        assert reverse.status_code == 200
        assert reverse.json()["address"] is None

    async def test_an_unknown_address_is_null_not_invented(
        self, client, patient_email
    ):
        headers = await auth(client, patient_email)
        with patch.object(get_maps_service(), "_osm_get", return_value={}):
            response = await client.get(
                REVERSE,
                params={"latitude": 0.0, "longitude": 0.0},
                headers=headers,
            )

        assert response.status_code == 200
        assert response.json()["address"] is None
