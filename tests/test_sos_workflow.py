"""
The SOS emergency workflow.

The tests that matter most here are the ones about who can reach what, and
about the state machine — because an emergency record is read by people acting
on it, and a wrong status in it is a wrong action taken.

Nothing in Phase 2 sends an SMS, places a call or dispatches a vehicle. The
notifier is stubbed throughout so these tests assert on the record and on the
audience of each announcement, which is the part that has to be right before
Phase 3 plugs real transports in behind it.
"""

import asyncio
import time
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import event, select, text

from app.core.doctor_code import generate_doctor_code
from app.core.security import get_password_hash
from app.models.doctor import Doctor
from app.models.emergency import EmergencyRequest, EmergencyStatusEvent
from app.models.emergency_profile import EmergencyProfile
from app.models.patient import Patient
from app.models.user import User
from app.services.sos import ALLOWED_TRANSITIONS, sos_service
from app.services.sos_notifications import (
    NullEmergencyNotifier,
    set_emergency_notifier,
)
from app.services.twilio_gateway import set_twilio_gateway
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
PATIENT_A = "sos.a@aronofy.com"
PATIENT_B = "sos.b@aronofy.com"
DOCTOR_A = "sos.doc.a@aronofy.com"
DOCTOR_B = "sos.doc.b@aronofy.com"
ADMIN = "sos.admin@aronofy.com"

PATNA = (25.5941, 85.1376)


class RecordingNotifier:
    """Captures announcements so the audience and payload can be asserted."""

    def __init__(self):
        self.raised: list[dict] = []
        self.updated: list[dict] = []

    async def emergency_raised(self, payload: dict) -> None:
        self.raised.append(payload)

    async def emergency_updated(self, payload: dict) -> None:
        self.updated.append(payload)


@pytest.fixture(autouse=True)
def silent_notifier():
    """Phase 2 announces over the socket; these tests do not need it."""
    set_emergency_notifier(NullEmergencyNotifier())
    yield
    set_emergency_notifier(NullEmergencyNotifier())


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("emergency_status_events", "emergency_requests",
                  "emergency_profiles", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))

    ids = {}
    for email, role in ((PATIENT_A, "patient"), (PATIENT_B, "patient"),
                        (DOCTOR_A, "doctor"), (DOCTOR_B, "doctor"),
                        (ADMIN, "admin")):
        user = User(email=email, hashed_password=get_password_hash(PW),
                    role=role, is_active=True, is_verified=True)
        db.add(user)
        await db.flush()
        ids[email] = user.id

        if role == "patient":
            db.add(Patient(
                id=user.id, first_name="Test", last_name=email[4],
                phone="+911111111111", date_of_birth="1990-05-15",
                gender="other", blood_type="O+",
            ))
        elif role == "doctor":
            db.add(Doctor(
                id=user.id, first_name="Res", last_name="Ponder",
                phone="+912222222222", specialty="Emergency Medicine",
                license_number=f"LIC-SOS-{email}", verification_status="verified",
                doctor_code=generate_doctor_code(),
            ))
    await db.flush()

    # Patient A is ready to raise an SOS; patient B deliberately is not.
    db.add(EmergencyProfile(
        id=ids[PATIENT_A], contact_name="Ravi Kumar",
        contact_phone="+919876543210", contact_relationship="Brother",
        house_number="12/A", street="Gandhi Road", locality="Rajendra Nagar",
        city="Patna", district="Patna", state="Bihar", country="India",
        pincode="800001", latitude=PATNA[0], longitude=PATNA[1],
        maps_url="https://www.google.com/maps/search/?api=1&query=25.5941,85.1376",
    ))
    await db.commit()
    return ids


async def auth(client: AsyncClient, email: str) -> dict:
    r = await client.post("/api/v1/auth/login", json=await login_payload(email, PW))
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def raise_sos(client: AsyncClient, headers: dict, **coords):
    return await client.post("/api/v1/patient/sos", json=coords or {}, headers=headers)


# ── raising ──────────────────────────────────────────────────────────────

class TestTrigger:
    async def test_raises_a_pending_emergency(self, client, estate):
        headers = await auth(client, PATIENT_A)
        r = await raise_sos(client, headers, latitude=PATNA[0], longitude=PATNA[1])
        assert r.status_code == 201, r.text

        body = r.json()
        assert body["status"] == "pending"
        assert body["is_active"] is True
        assert body["created_by"] == "patient"
        assert body["contact_name"] == "Ravi Kumar"
        assert body["blood_type"] == "O+"
        assert body["patient_age"] is not None
        assert "25.5941" in body["maps_url"]

    async def test_nothing_is_claimed_to_be_dispatched(self, client, estate, db):
        """
        The record must not assert that help is on the way.

        The previous implementation set `ambulance_dispatched` true with a
        hard-coded unit and a twelve-minute ETA the moment the button was
        pressed. Nothing had been arranged; a patient reading that would have
        stopped looking for other help.
        """
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)

        row = (await db.execute(select(EmergencyRequest))).scalars().first()
        assert row.ambulance_dispatched is False
        assert row.eta is None
        assert row.hospital_name is None

    async def test_the_first_timeline_entry_is_written(self, client, estate, db):
        headers = await auth(client, PATIENT_A)
        r = await raise_sos(client, headers)
        assert len(r.json()["timeline"]) == 1
        assert r.json()["timeline"][0]["status"] == "pending"
        assert r.json()["timeline"][0]["actor_role"] == "patient"

    async def test_live_coordinates_are_written_back_to_the_profile(
        self, client, estate, db
    ):
        """The most recent known position must outlive the emergency."""
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers, latitude=19.0760, longitude=72.8777)

        profile = await db.get(EmergencyProfile, estate[PATIENT_A])
        await db.refresh(profile)
        assert profile.latitude == pytest.approx(19.0760)
        assert "19.076" in profile.maps_url

    async def test_falls_back_to_the_stored_location_when_gps_is_refused(
        self, client, estate
    ):
        """
        Refusing the browser prompt is not a reason to refuse an ambulance.
        """
        headers = await auth(client, PATIENT_A)
        r = await raise_sos(client, headers)  # no coordinates sent

        assert r.status_code == 201, r.text
        assert r.json()["latitude"] == pytest.approx(PATNA[0])

    async def test_refused_with_no_stored_location_is_a_clear_error(
        self, client, estate, db
    ):
        profile = await db.get(EmergencyProfile, estate[PATIENT_A])
        profile.latitude = None
        profile.longitude = None
        await db.commit()

        r = await raise_sos(client, await auth(client, PATIENT_A))
        assert r.status_code == 422
        assert "location" in r.json()["message"].lower()

    async def test_refused_without_an_emergency_profile(self, client, estate):
        r = await raise_sos(client, await auth(client, PATIENT_B))
        assert r.status_code == 422
        assert "profile" in r.json()["message"].lower()

    @pytest.mark.parametrize("coords", [
        {"latitude": 91, "longitude": 0},
        {"latitude": 0, "longitude": 181},
        {"latitude": "abc", "longitude": 85.0},
    ])
    async def test_impossible_coordinates_refused(self, client, estate, coords):
        r = await raise_sos(client, await auth(client, PATIENT_A), **coords)
        assert r.status_code == 422


# ── duplicates ───────────────────────────────────────────────────────────

class TestDuplicateSuppression:
    async def test_a_second_sos_is_refused_while_one_is_open(self, client, estate):
        headers = await auth(client, PATIENT_A)
        assert (await raise_sos(client, headers)).status_code == 201

        second = await raise_sos(client, headers)
        assert second.status_code == 422
        assert second.json()["message"] == "Emergency already active."

    async def test_only_one_row_exists_after_a_double_press(
        self, client, estate, db
    ):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        await raise_sos(client, headers)

        rows = (await db.execute(select(EmergencyRequest))).scalars().all()
        assert len(rows) == 1

    async def test_a_new_sos_is_allowed_once_the_previous_one_closed(
        self, client, estate
    ):
        headers = await auth(client, PATIENT_A)
        first = await raise_sos(client, headers)
        await client.post(f"/api/v1/patient/sos/{first.json()['id']}/cancel",
                          json={"reason": "False alarm"}, headers=headers)

        again = await raise_sos(client, headers)
        assert again.status_code == 201, again.text

    async def test_one_patients_emergency_does_not_block_another(
        self, client, estate, db
    ):
        db.add(EmergencyProfile(
            id=estate[PATIENT_B], contact_name="Sita Devi",
            contact_phone="+919876500000", contact_relationship="Sister",
            house_number="7", street="Main Road", locality="Kankarbagh",
            city="Patna", district="Patna", state="Bihar", country="India",
            pincode="800020", latitude=PATNA[0], longitude=PATNA[1],
        ))
        await db.commit()

        assert (await raise_sos(client, await auth(client, PATIENT_A))).status_code == 201
        assert (await raise_sos(client, await auth(client, PATIENT_B))).status_code == 201


# ── the state machine ────────────────────────────────────────────────────

class TestStatusTransitions:
    async def _open(self, client, estate):
        headers = await auth(client, PATIENT_A)
        r = await raise_sos(client, headers)
        return r.json()["id"]

    async def test_a_doctor_advances_and_claims_it(self, client, estate):
        emergency_id = await self._open(client, estate)
        headers = await auth(client, DOCTOR_A)

        r = await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                             json={"status": "accepted"}, headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "accepted"
        # Acting on an unclaimed emergency takes it, so the queue stops
        # offering it to every other clinician.
        assert r.json()["assigned_doctor_id"] == str(estate[DOCTOR_A])

    async def test_the_full_forward_path(self, client, estate):
        emergency_id = await self._open(client, estate)
        headers = await auth(client, DOCTOR_A)

        for status_value in ("accepted", "doctor_assigned",
                             "ambulance_dispatched", "hospital_reached",
                             "resolved"):
            r = await client.put(
                f"/api/v1/doctor/emergencies/{emergency_id}/status",
                json={"status": status_value}, headers=headers)
            assert r.status_code == 200, f"{status_value}: {r.text}"

        assert r.json()["status"] == "resolved"
        assert r.json()["resolved_at"] is not None
        assert r.json()["is_active"] is False

    async def test_dispatching_sets_the_ambulance_flag(self, client, estate, db):
        emergency_id = await self._open(client, estate)
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "ambulance_dispatched"},
                         headers=await auth(client, DOCTOR_A))

        row = await db.get(EmergencyRequest, uuid.UUID(emergency_id))
        await db.refresh(row)
        assert row.ambulance_dispatched is True

    async def test_backwards_transitions_are_refused(self, client, estate):
        emergency_id = await self._open(client, estate)
        headers = await auth(client, DOCTOR_A)
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "hospital_reached"}, headers=headers)

        r = await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                             json={"status": "pending"}, headers=headers)
        assert r.status_code == 422
        assert "cannot move" in r.json()["message"].lower()

    async def test_a_resolved_emergency_is_frozen(self, client, estate):
        emergency_id = await self._open(client, estate)
        headers = await auth(client, DOCTOR_A)
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "resolved"}, headers=headers)

        for attempt in ("ambulance_dispatched", "cancelled", "pending"):
            r = await client.put(
                f"/api/v1/doctor/emergencies/{emergency_id}/status",
                json={"status": attempt}, headers=headers)
            assert r.status_code == 422, attempt

    async def test_an_unknown_status_is_refused_by_the_schema(self, client, estate):
        emergency_id = await self._open(client, estate)
        r = await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                             json={"status": "teleported"},
                             headers=await auth(client, DOCTOR_A))
        assert r.status_code == 422

    def test_terminal_states_have_no_exits(self):
        assert ALLOWED_TRANSITIONS["resolved"] == ()
        assert ALLOWED_TRANSITIONS["cancelled"] == ()

    async def test_every_change_appends_to_the_timeline(self, client, estate):
        emergency_id = await self._open(client, estate)
        headers = await auth(client, DOCTOR_A)
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "accepted", "note": "On my way"},
                         headers=headers)
        r = await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                             json={"status": "ambulance_dispatched"},
                             headers=headers)

        timeline = r.json()["timeline"]
        assert [e["status"] for e in timeline] == [
            "pending", "accepted", "ambulance_dispatched"]
        assert timeline[1]["note"] == "On my way"
        assert timeline[1]["actor_role"] == "doctor"

    async def test_a_patient_cannot_drive_the_clinical_status(self, client, estate):
        """
        A patient's one write is cancellation.

        Letting them mark themselves as having reached hospital would put a
        clinical claim in the record that nobody clinical made.
        """
        emergency_id = await self._open(client, estate)
        headers = await auth(client, PATIENT_A)
        for path in (f"/api/v1/doctor/emergencies/{emergency_id}/status",
                     f"/api/v1/admin/emergencies/{emergency_id}/status"):
            r = await client.put(path, json={"status": "resolved"}, headers=headers)
            assert r.status_code == 403, path


# ── cancellation ─────────────────────────────────────────────────────────

class TestCancellation:
    async def test_a_patient_may_cancel_their_own(self, client, estate):
        headers = await auth(client, PATIENT_A)
        emergency_id = (await raise_sos(client, headers)).json()["id"]

        r = await client.post(f"/api/v1/patient/sos/{emergency_id}/cancel",
                              json={"reason": "Pressed by accident"},
                              headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "cancelled"
        assert r.json()["cancelled_at"] is not None
        assert r.json()["cancel_reason"] == "Pressed by accident"
        assert r.json()["is_active"] is False

    async def test_cancelling_twice_is_refused(self, client, estate):
        headers = await auth(client, PATIENT_A)
        emergency_id = (await raise_sos(client, headers)).json()["id"]
        await client.post(f"/api/v1/patient/sos/{emergency_id}/cancel",
                          json={}, headers=headers)

        r = await client.post(f"/api/v1/patient/sos/{emergency_id}/cancel",
                              json={}, headers=headers)
        assert r.status_code == 422

    async def test_a_patient_cannot_cancel_another_patients(self, client, estate, db):
        headers_a = await auth(client, PATIENT_A)
        emergency_id = (await raise_sos(client, headers_a)).json()["id"]

        r = await client.post(f"/api/v1/patient/sos/{emergency_id}/cancel",
                              json={}, headers=await auth(client, PATIENT_B))
        # 404, not 403: answering differently would confirm the id is real.
        assert r.status_code == 404


# ── who can see what ─────────────────────────────────────────────────────

class TestVisibility:
    async def _open(self, client, estate):
        return (await raise_sos(client, await auth(client, PATIENT_A))).json()["id"]

    async def test_a_patient_sees_only_their_own(self, client, estate):
        emergency_id = await self._open(client, estate)

        mine = await client.get("/api/v1/patient/sos",
                                headers=await auth(client, PATIENT_A))
        assert [e["id"] for e in mine.json()] == [emergency_id]

        theirs = await client.get("/api/v1/patient/sos",
                                  headers=await auth(client, PATIENT_B))
        assert theirs.json() == []

    async def test_a_patient_cannot_read_another_patients_by_id(
        self, client, estate
    ):
        emergency_id = await self._open(client, estate)
        r = await client.get(f"/api/v1/patient/sos/{emergency_id}",
                             headers=await auth(client, PATIENT_B))
        assert r.status_code == 404

    async def test_an_unclaimed_emergency_reaches_every_clinician(
        self, client, estate
    ):
        """That queue is how an emergency finds somebody who can accept it."""
        await self._open(client, estate)
        for email in (DOCTOR_A, DOCTOR_B):
            r = await client.get("/api/v1/doctor/emergencies",
                                 headers=await auth(client, email))
            assert r.status_code == 200
            assert len(r.json()) == 1, email

    async def test_a_claimed_emergency_is_hidden_from_other_clinicians(
        self, client, estate
    ):
        emergency_id = await self._open(client, estate)
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "accepted"},
                         headers=await auth(client, DOCTOR_A))

        mine = await client.get("/api/v1/doctor/emergencies",
                                headers=await auth(client, DOCTOR_A))
        assert [e["id"] for e in mine.json()] == [emergency_id]

        theirs = await client.get("/api/v1/doctor/emergencies",
                                  headers=await auth(client, DOCTOR_B))
        assert theirs.json() == []

        direct = await client.get(f"/api/v1/doctor/emergencies/{emergency_id}",
                                  headers=await auth(client, DOCTOR_B))
        assert direct.status_code == 404

    async def test_an_administrator_sees_everything(self, client, estate):
        emergency_id = await self._open(client, estate)
        r = await client.get("/api/v1/admin/emergencies",
                             headers=await auth(client, ADMIN))
        assert [e["id"] for e in r.json()] == [emergency_id]

    async def test_anonymous_requests_are_refused(self, client, estate):
        emergency_id = await self._open(client, estate)
        for path in ("/api/v1/patient/sos", "/api/v1/doctor/emergencies",
                     "/api/v1/admin/emergencies",
                     f"/api/v1/patient/sos/{emergency_id}"):
            assert (await client.get(path)).status_code == 401, path

    async def test_role_boundaries_on_the_queues(self, client, estate):
        await self._open(client, estate)
        patient = await auth(client, PATIENT_A)
        doctor = await auth(client, DOCTOR_A)

        assert (await client.get("/api/v1/admin/emergencies",
                                 headers=patient)).status_code == 403
        assert (await client.get("/api/v1/admin/emergencies",
                                 headers=doctor)).status_code == 403
        assert (await client.get("/api/v1/doctor/emergencies",
                                 headers=patient)).status_code == 403


# ── assignment ───────────────────────────────────────────────────────────

class TestAssignment:
    async def test_an_administrator_assigns_a_clinician(self, client, estate):
        emergency_id = (await raise_sos(
            client, await auth(client, PATIENT_A))).json()["id"]

        r = await client.put(f"/api/v1/admin/emergencies/{emergency_id}/assign",
                             json={"doctor_id": str(estate[DOCTOR_B])},
                             headers=await auth(client, ADMIN))
        assert r.status_code == 200, r.text
        assert r.json()["assigned_doctor_id"] == str(estate[DOCTOR_B])
        assert r.json()["status"] == "doctor_assigned"

    async def test_assignment_makes_it_visible_to_that_clinician(
        self, client, estate
    ):
        emergency_id = (await raise_sos(
            client, await auth(client, PATIENT_A))).json()["id"]
        await client.put(f"/api/v1/admin/emergencies/{emergency_id}/assign",
                         json={"doctor_id": str(estate[DOCTOR_B])},
                         headers=await auth(client, ADMIN))

        theirs = await client.get("/api/v1/doctor/emergencies",
                                  headers=await auth(client, DOCTOR_B))
        assert [e["id"] for e in theirs.json()] == [emergency_id]

    async def test_a_doctor_cannot_assign(self, client, estate):
        emergency_id = (await raise_sos(
            client, await auth(client, PATIENT_A))).json()["id"]
        r = await client.put(f"/api/v1/admin/emergencies/{emergency_id}/assign",
                             json={"doctor_id": str(estate[DOCTOR_B])},
                             headers=await auth(client, DOCTOR_A))
        assert r.status_code == 403

    async def test_an_unapproved_clinician_cannot_be_assigned(
        self, client, estate, db
    ):
        """The same approval rule the doctor portal enforces."""
        doctor = await db.get(Doctor, estate[DOCTOR_B])
        doctor.verification_status = "pending"
        await db.commit()

        emergency_id = (await raise_sos(
            client, await auth(client, PATIENT_A))).json()["id"]
        r = await client.put(f"/api/v1/admin/emergencies/{emergency_id}/assign",
                             json={"doctor_id": str(estate[DOCTOR_B])},
                             headers=await auth(client, ADMIN))
        assert r.status_code == 403


# ── announcements ────────────────────────────────────────────────────────

class TestNotifications:
    async def test_raising_announces_once(self, client, estate):
        notifier = RecordingNotifier()
        set_emergency_notifier(notifier)

        await raise_sos(client, await auth(client, PATIENT_A))
        assert len(notifier.raised) == 1
        assert notifier.raised[0]["status"] == "pending"
        assert notifier.raised[0]["patient_name"]

    async def test_a_refused_sos_announces_nothing(self, client, estate):
        """A responder must never be sent to an emergency that was not created."""
        notifier = RecordingNotifier()
        set_emergency_notifier(notifier)

        r = await raise_sos(client, await auth(client, PATIENT_B))
        assert r.status_code == 422
        assert notifier.raised == []

    async def test_status_changes_announce(self, client, estate):
        emergency_id = (await raise_sos(
            client, await auth(client, PATIENT_A))).json()["id"]

        notifier = RecordingNotifier()
        set_emergency_notifier(notifier)
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "accepted"},
                         headers=await auth(client, DOCTOR_A))

        assert len(notifier.updated) == 1
        assert notifier.updated[0]["status"] == "accepted"

    async def test_the_announced_payload_is_json_serialisable(self, client, estate):
        """
        The payload is built from ORM values, so it carries `UUID` and
        `datetime` objects. `WebSocket.send_json` cannot encode either, and the
        notifier deliberately swallows delivery errors — so this failed
        silently in production: every emergency was recorded correctly and
        announced to nobody.
        """
        import json

        from fastapi.encoders import jsonable_encoder

        notifier = RecordingNotifier()
        set_emergency_notifier(notifier)
        await raise_sos(client, await auth(client, PATIENT_A))

        payload = notifier.raised[0]
        # Raw payload is not encodable — that is the trap.
        with pytest.raises(TypeError):
            json.dumps(payload)
        # The notifier coerces before sending, which is what must never regress.
        json.dumps(jsonable_encoder({"type": "EMERGENCY_SOS_CREATED", **payload}))

    async def test_the_socket_notifier_encodes_before_sending(self):
        import inspect

        from app.services.sos_notifications import WebSocketEmergencyNotifier

        source = inspect.getsource(WebSocketEmergencyNotifier)
        assert "jsonable_encoder" in source, (
            "the payload must be coerced to JSON primitives before send_json"
        )

    async def test_the_socket_notifier_never_broadcasts_globally(self):
        """
        A global broadcast would put a patient's name, blood group and home
        address on every open socket, including other patients'.
        """
        import inspect

        from app.services.sos_notifications import WebSocketEmergencyNotifier

        source = inspect.getsource(WebSocketEmergencyNotifier)
        assert "broadcast_to_role" in source
        assert "websocket_manager.broadcast(" not in source


# ── persistence ──────────────────────────────────────────────────────────

class TestPersistence:
    async def test_the_active_endpoint_survives_a_reload(self, client, estate):
        headers = await auth(client, PATIENT_A)
        created = (await raise_sos(client, headers)).json()

        again = await client.get("/api/v1/patient/sos/active", headers=headers)
        assert again.status_code == 200
        assert again.json()["active"] is True
        assert again.json()["emergency"]["id"] == created["id"]

    async def test_no_active_emergency_reports_cleanly(self, client, estate):
        r = await client.get("/api/v1/patient/sos/active",
                             headers=await auth(client, PATIENT_A))
        assert r.status_code == 200
        assert r.json() == {"active": False, "emergency": None}

    async def test_a_resolved_emergency_leaves_the_active_view(self, client, estate):
        headers = await auth(client, PATIENT_A)
        emergency_id = (await raise_sos(client, headers)).json()["id"]
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "resolved"},
                         headers=await auth(client, DOCTOR_A))

        active = await client.get("/api/v1/patient/sos/active", headers=headers)
        assert active.json()["active"] is False

        history = await client.get("/api/v1/patient/sos", headers=headers)
        assert [e["id"] for e in history.json()] == [emergency_id]

    async def test_events_are_stored_not_derived(self, client, estate, db):
        headers = await auth(client, PATIENT_A)
        emergency_id = (await raise_sos(client, headers)).json()["id"]
        await client.put(f"/api/v1/doctor/emergencies/{emergency_id}/status",
                         json={"status": "accepted"},
                         headers=await auth(client, DOCTOR_A))

        events = (await db.execute(
            select(EmergencyStatusEvent).where(
                EmergencyStatusEvent.emergency_id == uuid.UUID(emergency_id))
        )).scalars().all()
        assert len(events) == 2


# ── legacy route ─────────────────────────────────────────────────────────

class TestLegacyPanicRoute:
    async def test_it_still_answers_in_its_original_shape(self, client, estate):
        r = await client.post(
            "/api/v1/patient/emergency",
            json={"location": {"lat": PATNA[0], "lng": PATNA[1],
                               "address": "Patna"}},
            headers=await auth(client, PATIENT_A))
        assert r.status_code == 201, r.text
        for field in ("id", "patient_id", "patient_name", "patient_phone",
                      "location", "ambulance_dispatched", "status"):
            assert field in r.json()

    async def test_it_no_longer_fabricates_a_dispatch(self, client, estate):
        r = await client.post(
            "/api/v1/patient/emergency",
            json={"location": {"lat": PATNA[0], "lng": PATNA[1],
                               "address": "Patna"}},
            headers=await auth(client, PATIENT_A))
        body = r.json()
        assert body["status"] == "pending"
        assert body["ambulance_dispatched"] is False
        assert body["eta"] is None

    async def test_it_shares_the_duplicate_check(self, client, estate):
        headers = await auth(client, PATIENT_A)
        payload = {"location": {"lat": PATNA[0], "lng": PATNA[1], "address": "Patna"}}
        assert (await client.post("/api/v1/patient/emergency",
                                  json=payload, headers=headers)).status_code == 201
        second = await client.post("/api/v1/patient/emergency",
                                   json=payload, headers=headers)
        assert second.status_code == 422


class TestRequestPathCost:
    """
    What the SOS request costs, pinned so it cannot quietly grow again.

    An SOS response is the most latency-sensitive thing this platform does, and
    against a managed database its cost is almost entirely *round trips* — each
    one is a full network wait, and they are sequential. Wall-clock assertions
    would only measure whoever's laptop ran the suite; the number of round trips
    is a property of the code, so that is what is asserted here.

    The budget below is the whole request: authentication, the patient with
    their emergency profile and the duplicate check in one query, the emergency
    row, its first timeline entry, the position write-back, and the commit.
    """

    BUDGET = 7

    async def test_the_request_path_stays_within_its_round_trip_budget(
        self, client, estate
    ):
        from conftest import test_engine

        headers = await auth(client, PATIENT_A)

        statements: list[str] = []
        commits: list[int] = []
        sync_engine = test_engine.sync_engine

        def before(conn, cursor, stmt, params, context, executemany):
            statements.append(" ".join(stmt.split())[:60])

        def on_commit(conn):
            commits.append(1)

        event.listen(sync_engine, "before_cursor_execute", before)
        event.listen(sync_engine, "commit", on_commit)
        try:
            r = await raise_sos(client, headers, latitude=PATNA[0],
                                longitude=PATNA[1])
        finally:
            event.remove(sync_engine, "before_cursor_execute", before)
            event.remove(sync_engine, "commit", on_commit)

        assert r.status_code == 201, r.text
        trips = len(statements) + len(commits)
        assert trips <= self.BUDGET, (
            f"the SOS request now costs {trips} database round trips, budget is "
            f"{self.BUDGET}. Against a managed database each one is a network "
            f"wait a patient sits through:\n  "
            + "\n  ".join(statements)
        )

    async def test_the_duplicate_check_costs_nothing_extra(self, client, estate):
        """
        The "one emergency at a time" rule is answered by the same query that
        loads the patient, not by a query of its own.
        """
        from conftest import test_engine

        headers = await auth(client, PATIENT_A)
        seen: list[str] = []

        def before(conn, cursor, stmt, params, context, executemany):
            seen.append(" ".join(stmt.split()))

        event.listen(test_engine.sync_engine, "before_cursor_execute", before)
        try:
            assert (await raise_sos(client, headers, latitude=PATNA[0],
                                    longitude=PATNA[1])).status_code == 201
        finally:
            event.remove(test_engine.sync_engine, "before_cursor_execute", before)

        selects_on_emergencies = [
            s for s in seen
            if s.upper().startswith("SELECT") and "emergency_requests" in s
        ]
        assert len(selects_on_emergencies) == 1, (
            "the duplicate check should be folded into the patient query, not "
            f"issued separately:\n  " + "\n  ".join(selects_on_emergencies))
        assert "EXISTS" in selects_on_emergencies[0].upper()

    async def test_the_response_does_not_wait_for_the_fan_out(
        self, client, estate, monkeypatch
    ):
        """
        The property the production check measures, asserted without depending
        on how far away the database is.

        A gateway that takes a second per message would add nine seconds to the
        response if the request waited for the fan-out. Here the fan-out is
        driven explicitly (the suite suppresses the detached task), so what is
        asserted is the endpoint's contract: it returns having issued no vendor
        call at all.
        """
        from app.services.twilio_gateway import DeliveryResult

        called: list[str] = []

        class SlowGateway:
            def is_configured(self):
                return True

            def is_whatsapp_configured(self):
                return True

            async def place_call(self, to, say_text):
                called.append("voice")
                await asyncio.sleep(1.0)
                return DeliveryResult(success=True, sid="CA", provider_status="queued")

            async def send_sms(self, to, body):
                called.append("sms")
                await asyncio.sleep(1.0)
                return DeliveryResult(success=True, sid="SM", provider_status="queued")

            async def send_whatsapp(self, to, body, media_urls=None):
                called.append("whatsapp")
                await asyncio.sleep(1.0)
                return DeliveryResult(success=True, sid="WA", provider_status="queued")

        set_twilio_gateway(SlowGateway())
        try:
            headers = await auth(client, PATIENT_A)
            started = time.perf_counter()
            r = await raise_sos(client, headers, latitude=PATNA[0],
                                longitude=PATNA[1])
            elapsed = time.perf_counter() - started

            assert r.status_code == 201, r.text
            assert called == [], (
                "the SOS request reached the communications vendor before "
                f"responding: {called}")
            # Nine one-second sends would be unmissable; this leaves generous
            # room for a slow CI machine while still failing if the fan-out is
            # ever awaited inline.
            assert elapsed < 3.0, f"{elapsed:.2f}s"
        finally:
            set_twilio_gateway(None)
