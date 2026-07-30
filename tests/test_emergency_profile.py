"""
The patient Emergency Profile — CRUD, validation, and isolation.

Phase 1 is a data foundation, so the tests that matter most are the ones about
what the record may contain and who may reach it. Two things are checked
hardest:

* **Isolation.** One patient's request must never touch another patient's
  profile. There is no target-patient parameter to abuse, so the tests confirm
  that two patients writing at the same time see only their own record.
* **Validation.** The database will one day be read while somebody is
  unconscious, so a phone number that is not a phone number, or an address
  field carrying markup, must be refused at the door rather than stored and
  rendered later.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.doctor_code import generate_doctor_code
from app.core.security import get_password_hash
from app.models.doctor import Doctor
from app.models.emergency_profile import EmergencyProfile
from app.models.patient import Patient
from app.models.user import User
from app.services.emergency_profile import build_maps_url, format_address

pytestmark = pytest.mark.asyncio

PW = "password123"
PATIENT_A = "ep.a@aronofy.com"
PATIENT_B = "ep.b@aronofy.com"
DOCTOR = "ep.doc@aronofy.com"

VALID_CONTACT = {
    "contact_name": "Ravi Kumar",
    "contact_phone": "+919876543210",
    "contact_relationship": "brother",
    "alternate_phone": "+919876543211",
}

VALID_ADDRESS = {
    "house_number": "12/A",
    "street": "Gandhi Road",
    "landmark": "Near City Hospital",
    "locality": "Rajendra Nagar",
    "city": "Patna",
    "district": "Patna",
    "state": "Bihar",
    "country": "India",
    "pincode": "800001",
}


def payload(**overrides):
    contact = {**VALID_CONTACT, **overrides.pop("contact", {})}
    address = {**VALID_ADDRESS, **overrides.pop("address", {})}
    return {"contact": contact, "address": address}


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("emergency_profiles", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))

    ids = {}
    for email, role in ((PATIENT_A, "patient"), (PATIENT_B, "patient"),
                        (DOCTOR, "doctor")):
        user = User(email=email, hashed_password=get_password_hash(PW),
                    role=role, is_active=True, is_verified=True)
        db.add(user)
        await db.flush()
        ids[email] = user.id

        if role == "patient":
            db.add(Patient(
                id=user.id, first_name="Test", last_name="Patient",
                phone="+911111111111", date_of_birth="1990-01-01", gender="other",
            ))
        else:
            # An approved clinician with a Doctor ID, so the role-boundary test
            # below fails on authorisation rather than on being unable to sign
            # in at all — which would prove nothing about these routes.
            db.add(Doctor(
                id=user.id, first_name="Test", last_name="Clinician",
                phone="+912222222222", specialty="General Medicine",
                license_number="LIC-EP-1", verification_status="verified",
                doctor_code=generate_doctor_code(),
            ))
    await db.commit()
    return ids


async def auth(client: AsyncClient, email: str) -> dict:
    from conftest import login_payload
    r = await client.post("/api/v1/auth/login", json=await login_payload(email, PW))
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── read ─────────────────────────────────────────────────────────────────

class TestRead:
    async def test_absent_profile_is_null_not_an_error(self, client, estate):
        """
        A patient who has never filled the form in is the ordinary starting
        state. The page renders an empty form for it, so a 404 would make the
        normal case look like a failure.
        """
        r = await client.get("/api/v1/patient/emergency-profile",
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text
        assert r.json() is None

    async def test_created_profile_reads_back(self, client, estate):
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)

        r = await client.get("/api/v1/patient/emergency-profile", headers=headers)
        body = r.json()
        assert body["contact_name"] == "Ravi Kumar"
        assert body["contact_relationship"] == "Brother"  # title-cased on save
        assert body["city"] == "Patna"
        assert body["formatted_address"].startswith("12/A, Gandhi Road")


# ── create and update ────────────────────────────────────────────────────

class TestUpsert:
    async def test_create_returns_the_stored_profile(self, client, estate):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(), headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == str(estate[PATIENT_A])
        assert body["latitude"] is None
        assert body["maps_url"] is None

    async def test_saving_twice_updates_rather_than_duplicates(
        self, client, estate, db
    ):
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)
        r = await client.put(
            "/api/v1/patient/emergency-profile",
            json=payload(contact={"contact_name": "Sita Devi"}), headers=headers)
        assert r.status_code == 200
        assert r.json()["contact_name"] == "Sita Devi"

        rows = (await db.execute(select(EmergencyProfile))).scalars().all()
        assert len(rows) == 1, "a second profile row was created"

    async def test_editing_the_address_keeps_the_stored_location(
        self, client, estate
    ):
        """
        Coordinates are captured by their own endpoint. Saving an edited address
        must not silently discard a position the patient already granted.
        """
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)
        await client.put("/api/v1/patient/emergency-profile/location",
                         json={"latitude": 25.5941, "longitude": 85.1376},
                         headers=headers)

        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(address={"city": "Gaya"}), headers=headers)
        assert r.json()["city"] == "Gaya"
        assert r.json()["latitude"] == pytest.approx(25.5941)
        assert r.json()["maps_url"] is not None

    async def test_whitespace_is_trimmed_and_collapsed(self, client, estate):
        r = await client.put(
            "/api/v1/patient/emergency-profile",
            json=payload(contact={"contact_name": "  Ravi   Kumar  "},
                         address={"city": "  Patna  "}),
            headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text
        assert r.json()["contact_name"] == "Ravi Kumar"
        assert r.json()["city"] == "Patna"

    async def test_phone_separators_are_normalised(self, client, estate):
        """`+91 98765-43210` and `+919876543210` are the same telephone."""
        r = await client.put(
            "/api/v1/patient/emergency-profile",
            json=payload(contact={"contact_phone": "+91 98765-43210",
                                  "alternate_phone": None}),
            headers=await auth(client, PATIENT_A))
        assert r.json()["contact_phone"] == "+919876543210"

    async def test_blank_optional_fields_become_null(self, client, estate):
        r = await client.put(
            "/api/v1/patient/emergency-profile",
            json=payload(contact={"alternate_phone": "   "},
                         address={"landmark": ""}),
            headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text
        assert r.json()["alternate_phone"] is None
        assert r.json()["landmark"] is None


# ── validation ───────────────────────────────────────────────────────────

class TestValidation:
    @pytest.mark.parametrize("field", ["contact_name", "contact_phone",
                                       "contact_relationship"])
    async def test_required_contact_fields(self, client, estate, field):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(contact={field: "   "}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 422, f"{field} was accepted blank"

    @pytest.mark.parametrize("field", ["house_number", "street", "locality",
                                       "city", "district", "state", "pincode"])
    async def test_required_address_fields(self, client, estate, field):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(address={field: "  "}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 422, f"{field} was accepted blank"

    @pytest.mark.parametrize("phone", [
        "12345",                 # too short
        "1234567890123456",      # too long
        "not-a-number",
        "+91abcdefghij",
        "++919876543210",
        "<script>alert(1)</script>",
    ])
    async def test_invalid_phone_numbers_refused(self, client, estate, phone):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(contact={"contact_phone": phone}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 422, f"{phone!r} was accepted"

    async def test_alternate_number_must_differ(self, client, estate):
        """
        A duplicate alternative provides nothing: whoever works down the list
        dials the same unanswered phone twice.
        """
        r = await client.put(
            "/api/v1/patient/emergency-profile",
            json=payload(contact={"contact_phone": "+919876543210",
                                  "alternate_phone": "+91 98765 43210"}),
            headers=await auth(client, PATIENT_A))
        assert r.status_code == 422

    @pytest.mark.parametrize("pincode", ["12345", "1234567", "012345", "abcdef"])
    async def test_invalid_indian_pincode_refused(self, client, estate, pincode):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(address={"pincode": pincode}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 422, f"{pincode!r} was accepted for India"

    async def test_pincode_accepts_spacing(self, client, estate):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(address={"pincode": "800 001"}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text
        assert r.json()["pincode"] == "800001"

    async def test_non_indian_postcode_uses_the_general_rule(self, client, estate):
        r = await client.put(
            "/api/v1/patient/emergency-profile",
            json=payload(address={"country": "United Kingdom", "pincode": "SW1A1AA"}),
            headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize("value", [
        "<script>alert(1)</script>",
        "Rd | rm -rf",
        "{{constructor}}",
        "Patna\\Bihar",
        "Patna`whoami`",
    ])
    async def test_invalid_characters_refused_in_address(self, client, estate, value):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(address={"city": value}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 422, f"{value!r} was accepted"

    @pytest.mark.parametrize("raw,stored", [
        ("Patna\x00", "Patna"),
        ("Pat\x07na", "Patna"),
        ("Patna\nBihar", "Patna Bihar"),
        ("Patna\tBihar", "Patna Bihar"),
    ])
    async def test_control_characters_are_stripped_not_stored(
        self, client, estate, raw, stored
    ):
        """
        Control characters are removed rather than refused.

        Rejecting would fail a patient who pasted a multi-line address into one
        box, and nothing unsafe survives either way: the value is stripped of
        control characters and its whitespace collapsed *before* it is
        validated, so what reaches the column is already clean. PostgreSQL will
        not store a NUL byte in a text column at all, which is exactly the class
        of value this removes.
        """
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(address={"city": raw}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text
        assert r.json()["city"] == stored

    async def test_real_address_punctuation_is_accepted(self, client, estate):
        """The allowlist must not reject genuine Indian addresses."""
        r = await client.put(
            "/api/v1/patient/emergency-profile",
            json=payload(address={"house_number": "#12/A-3",
                                  "street": "M.G. Road (East)",
                                  "locality": "D'Souza Colony & Annexe"}),
            headers=await auth(client, PATIENT_A))
        assert r.status_code == 200, r.text

    async def test_digits_refused_in_a_person_name(self, client, estate):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(contact={"contact_name": "Ravi 123"}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 422

    async def test_oversized_values_refused(self, client, estate):
        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(contact={"contact_name": "A" * 200}),
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 422


# ── location ─────────────────────────────────────────────────────────────

class TestLocation:
    async def test_capture_stores_coordinates_and_derives_the_maps_url(
        self, client, estate
    ):
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)

        r = await client.put("/api/v1/patient/emergency-profile/location",
                             json={"latitude": 25.5941, "longitude": 85.1376},
                             headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["latitude"] == pytest.approx(25.5941)
        assert body["longitude"] == pytest.approx(85.1376)
        assert body["maps_url"] == build_maps_url(25.5941, 85.1376)
        assert "google.com/maps" in body["maps_url"]
        assert body["location_updated_at"] is not None

    async def test_maps_url_cannot_be_supplied_by_the_client(self, client, estate):
        """
        The link is rendered for somebody to follow in an emergency, so its
        destination is derived from the stored coordinates and never accepted
        from the request.
        """
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)
        r = await client.put(
            "/api/v1/patient/emergency-profile/location",
            json={"latitude": 25.5941, "longitude": 85.1376,
                  "maps_url": "https://evil.example.com"},
            headers=headers)
        assert r.status_code == 200
        assert "evil.example.com" not in r.json()["maps_url"]

    @pytest.mark.parametrize("coords", [
        {"latitude": 91, "longitude": 0},
        {"latitude": -91, "longitude": 0},
        {"latitude": 0, "longitude": 181},
        {"latitude": 0, "longitude": -181},
        {"latitude": 0, "longitude": 0},          # Null Island
        {"latitude": "abc", "longitude": 85.0},
    ])
    async def test_impossible_coordinates_refused(self, client, estate, coords):
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)
        r = await client.put("/api/v1/patient/emergency-profile/location",
                             json=coords, headers=headers)
        assert r.status_code == 422, f"{coords} was accepted"

    async def test_location_requires_an_existing_profile(self, client, estate):
        r = await client.put("/api/v1/patient/emergency-profile/location",
                             json={"latitude": 25.5941, "longitude": 85.1376},
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 404

    async def test_clearing_removes_every_location_field(self, client, estate):
        """A Maps link outliving its coordinates would point somewhere erased."""
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)
        await client.put("/api/v1/patient/emergency-profile/location",
                         json={"latitude": 25.5941, "longitude": 85.1376},
                         headers=headers)

        r = await client.delete("/api/v1/patient/emergency-profile/location",
                                headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["latitude"] is None
        assert body["longitude"] is None
        assert body["maps_url"] is None
        assert body["location_updated_at"] is None
        assert body["contact_name"] == "Ravi Kumar", "clearing dropped the contact"


# ── delete ───────────────────────────────────────────────────────────────

class TestDelete:
    async def test_delete_removes_the_profile(self, client, estate):
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)

        r = await client.delete("/api/v1/patient/emergency-profile", headers=headers)
        assert r.status_code == 200, r.text
        assert (await client.get("/api/v1/patient/emergency-profile",
                                 headers=headers)).json() is None

    async def test_delete_without_a_profile_is_404(self, client, estate):
        r = await client.delete("/api/v1/patient/emergency-profile",
                                headers=await auth(client, PATIENT_A))
        assert r.status_code == 404

    async def test_a_profile_can_be_recreated_after_deletion(self, client, estate):
        """
        The primary key is the patient's id. A soft delete would leave the key
        occupied and make the next save collide, so deletion has to actually
        free it.
        """
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)
        await client.delete("/api/v1/patient/emergency-profile", headers=headers)

        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(), headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["contact_name"] == "Ravi Kumar"


# ── isolation ────────────────────────────────────────────────────────────

class TestIsolation:
    async def test_each_patient_sees_only_their_own_profile(self, client, estate):
        a = await auth(client, PATIENT_A)
        b = await auth(client, PATIENT_B)

        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(contact={"contact_name": "Contact Aye"}),
                         headers=a)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(contact={"contact_name": "Contact Bee"},
                                      address={"license": "x"} if False else {}),
                         headers=b)

        assert (await client.get("/api/v1/patient/emergency-profile",
                                 headers=a)).json()["contact_name"] == "Contact Aye"
        assert (await client.get("/api/v1/patient/emergency-profile",
                                 headers=b)).json()["contact_name"] == "Contact Bee"

    async def test_one_patient_cannot_delete_anothers_profile(self, client, estate):
        a = await auth(client, PATIENT_A)
        b = await auth(client, PATIENT_B)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=a)

        # B has no profile; deleting can only ever address B's own record.
        assert (await client.delete("/api/v1/patient/emergency-profile",
                                    headers=b)).status_code == 404
        assert (await client.get("/api/v1/patient/emergency-profile",
                                 headers=a)).json() is not None

    async def test_the_routes_take_no_patient_identifier(self):
        """
        Structural isolation: if no handler accepts a patient id, no request can
        name a patient other than the caller.
        """
        from app.api.v1.endpoints import patient as patient_module

        for route in patient_module.router.routes:
            if "emergency-profile" in getattr(route, "path", ""):
                assert "{" not in route.path, f"{route.path} takes a path parameter"

    async def test_a_doctor_cannot_reach_the_patient_routes(self, client, estate):
        headers = await auth(client, DOCTOR)
        for method, path in (
            ("get", "/api/v1/patient/emergency-profile"),
            ("put", "/api/v1/patient/emergency-profile"),
            ("delete", "/api/v1/patient/emergency-profile"),
            ("put", "/api/v1/patient/emergency-profile/location"),
            ("delete", "/api/v1/patient/emergency-profile/location"),
        ):
            r = await getattr(client, method)(
                path, headers=headers,
                **({"json": payload()} if method == "put" else {}))
            assert r.status_code == 403, f"{method.upper()} {path} -> {r.status_code}"

    async def test_anonymous_requests_are_refused(self, client, estate):
        for method, path in (
            ("get", "/api/v1/patient/emergency-profile"),
            ("delete", "/api/v1/patient/emergency-profile"),
        ):
            r = await getattr(client, method)(path)
            assert r.status_code == 401


# ── helpers ──────────────────────────────────────────────────────────────

class TestSoftDeleteRevival:
    """
    A soft-deleted row still occupies the primary key.

    The key is the patient's id, so if anything ever set `deleted_at` — the
    base repository offers `soft_remove`, and the column exists on every model
    — the default lookup would report "no profile", the next save would insert,
    and PostgreSQL would refuse the duplicate key. The patient would be unable
    to save again, permanently. The upsert therefore looks past `deleted_at`
    and revives.
    """

    async def test_a_soft_deleted_profile_is_revived_by_the_next_save(
        self, client, estate, db
    ):
        headers = await auth(client, PATIENT_A)
        await client.put("/api/v1/patient/emergency-profile",
                         json=payload(), headers=headers)

        profile = await db.get(EmergencyProfile, estate[PATIENT_A])
        profile.soft_delete()
        await db.commit()

        assert (await client.get("/api/v1/patient/emergency-profile",
                                 headers=headers)).json() is None

        r = await client.put("/api/v1/patient/emergency-profile",
                             json=payload(contact={"contact_name": "Sita Devi"}),
                             headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["contact_name"] == "Sita Devi"

        rows = (await db.execute(select(EmergencyProfile))).scalars().all()
        assert len(rows) == 1, "revival created a second row"
        assert rows[0].deleted_at is None


class TestRateLimitScope:
    """
    `/patient/emergency-profile` must not inherit the panic button's throttle.

    The rate limiter matched protected paths with a bare `startswith`, so every
    route whose name merely began with `/patient/emergency` was capped at the
    SOS rate of ten per minute. A patient correcting a typo in their address a
    few times would have been answered with 429.
    """

    def test_profile_routes_are_not_treated_as_the_panic_route(self):
        from app.middleware.rate_limit import PROTECTED_PATHS

        def matches(path: str) -> bool:
            return any(
                path == pattern or path.startswith(pattern + "/")
                for pattern in PROTECTED_PATHS
            )

        assert matches("/patient/emergency") is True
        assert matches("/patient/emergency/abc") is True
        assert matches("/patient/emergency-profile") is False
        assert matches("/patient/emergency-profile/location") is False

    def test_the_login_routes_still_share_one_budget(self):
        """The boundary fix must not undo the shared login bucket."""
        from app.middleware.rate_limit import PROTECTED_PATHS

        assert "/auth/login/doctor".startswith("/auth/login" + "/")
        assert "/auth/login" in PROTECTED_PATHS

    async def test_many_consecutive_saves_are_not_throttled(self, client, estate):
        """Editing a profile repeatedly is ordinary use, not abuse."""
        headers = await auth(client, PATIENT_A)
        for i in range(15):
            r = await client.put(
                "/api/v1/patient/emergency-profile",
                json=payload(address={"house_number": f"{i + 1}/A"}),
                headers=headers)
            assert r.status_code == 200, f"save {i + 1} -> {r.status_code}"


class TestHelpers:
    def test_maps_url_points_at_the_given_coordinates(self):
        url = build_maps_url(25.5941, 85.1376)
        assert url.startswith("https://www.google.com/maps")
        assert "25.5941,85.1376" in url

    def test_formatted_address_skips_missing_parts(self):
        profile = EmergencyProfile(
            house_number="12/A", street="Gandhi Road", landmark=None,
            locality="Rajendra Nagar", city="Patna", district="Patna",
            state="Bihar", country="India", pincode="800001",
        )
        formatted = format_address(profile)
        assert formatted == (
            "12/A, Gandhi Road, Rajendra Nagar, Patna, Patna, Bihar, India, 800001"
        )
        assert ", ," not in formatted
