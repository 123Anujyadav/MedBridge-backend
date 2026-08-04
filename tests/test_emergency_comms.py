"""
The emergency communication layer.

Two properties dominate these tests, because they are what the feature is for:

* **Nobody is lost.** A row exists for every intended contact before the vendor
  is called, a transient failure is retried with backoff, a permanent one stops
  being retried, and an unconfigured channel is recorded as skipped rather than
  vanishing.
* **Nothing is invented.** With no Google Maps key there is no address, no
  hospital and no ETA — not a placeholder, not an estimate. The emergency still
  works.

No test here touches Twilio or Google. Both are behind interfaces, and the
fakes below stand in for them, so the suite never places a call or spends money.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text

from app.core.config import settings
from app.core.doctor_code import generate_doctor_code
from app.core.security import get_password_hash
from app.models.communication import CommunicationLog
from app.models.doctor import Doctor
from app.models.emergency import EmergencyRequest
from app.models.emergency_profile import EmergencyProfile
from app.models.patient import Patient
from app.models.user import User
from app.services import emergency_templates as templates
from app.services.emergency_comms import emergency_comms_service, mask_phone
from app.services.maps import MapsService, set_maps_service
from app.services.sos_notifications import (
    NullEmergencyNotifier, set_emergency_notifier,
)
from app.services.twilio_gateway import DeliveryResult, set_twilio_gateway
from conftest import login_payload

pytestmark = pytest.mark.asyncio

PW = "password123"
PATIENT_A = "comms.patient@aronofy.com"
DOCTOR_A = "comms.doctor@aronofy.com"
ADMIN_A = "comms.admin@aronofy.com"
PATNA = (25.5941, 85.1376)


# ── doubles ──────────────────────────────────────────────────────────────

class FakeGateway:
    """A Twilio stand-in whose outcome each test chooses."""

    def __init__(self, result: DeliveryResult | None = None):
        self.result = result or DeliveryResult(success=True, sid="SM_ok",
                                               provider_status="queued")
        self.calls: list[tuple[str, str, str]] = []
        self.whatsapp_configured = True

    def is_configured(self) -> bool:
        return True

    def is_whatsapp_configured(self) -> bool:
        return self.whatsapp_configured

    async def place_call(self, to, say_text):
        self.calls.append(("voice", to, say_text))
        return self.result

    async def send_sms(self, to, body):
        self.calls.append(("sms", to, body))
        return self.result

    async def send_whatsapp(self, to, body, media_urls=None):
        self.calls.append(("whatsapp", to, body))
        if not self.whatsapp_configured:
            return DeliveryResult(success=False, skipped=True, retryable=False,
                                  error_code="not_configured",
                                  error_message="Twilio WhatsApp is not configured.")
        return self.result


class FakeMaps(MapsService):
    """Maps with a key, answering fixed data."""

    def __init__(self, hospital=None, address="12 Real Street, Patna"):
        super().__init__()
        self._hospital = hospital
        self._address = address

    def is_enabled(self) -> bool:
        return True

    async def reverse_geocode(self, latitude, longitude):
        return self._address

    async def nearest_hospital_with_eta(self, latitude, longitude):
        return self._hospital


@pytest.fixture(autouse=True)
def isolated_transports():
    """Every test gets its own doubles, and none reaches a real vendor."""
    set_emergency_notifier(NullEmergencyNotifier())
    gateway = FakeGateway()
    set_twilio_gateway(gateway)
    set_maps_service(MapsService())  # no key by default
    yield gateway
    set_twilio_gateway(None)
    set_maps_service(None)


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("communication_logs", "emergency_status_events",
                  "emergency_requests", "emergency_profiles", "doctors",
                  "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))

    ids = {}
    for email, role in ((PATIENT_A, "patient"), (DOCTOR_A, "doctor"),
                        (ADMIN_A, "admin")):
        user = User(email=email, hashed_password=get_password_hash(PW),
                    role=role, is_active=True, is_verified=True)
        db.add(user)
        await db.flush()
        ids[email] = user.id

        if role == "patient":
            db.add(Patient(id=user.id, first_name="Comms", last_name="Patient",
                           phone="+911111111111", date_of_birth="1990-05-15",
                           gender="other", blood_type="O+"))
        elif role == "doctor":
            db.add(Doctor(id=user.id, first_name="Res", last_name="Ponder",
                          phone="+912222222222", specialty="Emergency Medicine",
                          license_number="LIC-COMMS-1",
                          verification_status="verified",
                          doctor_code=generate_doctor_code()))
        else:
            # Administrators are reached on the phone held against their
            # profile, if they have one.
            db.add(Patient(id=user.id, first_name="Ad", last_name="Min",
                           phone="+913333333333", date_of_birth="1980-01-01",
                           gender="other"))

    await db.flush()
    db.add(EmergencyProfile(
        id=ids[PATIENT_A], contact_name="Ravi Kumar",
        contact_phone="+919876543210", contact_relationship="Brother",
        house_number="12/A", street="Gandhi Road", locality="Rajendra Nagar",
        city="Patna", district="Patna", state="Bihar", country="India",
        pincode="800001", latitude=PATNA[0], longitude=PATNA[1],
    ))
    await db.commit()
    return ids


async def auth(client, email):
    r = await client.post("/api/v1/auth/login", json=await login_payload(email, PW))
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def raise_sos(client, headers):
    return await client.post("/api/v1/patient/sos", json={}, headers=headers)


async def emergency_of(db, patient_id):
    return (await db.execute(
        select(EmergencyRequest).where(EmergencyRequest.patient_id == patient_id)
    )).scalars().first()


# ── queueing ─────────────────────────────────────────────────────────────

class TestQueueing:
    async def test_one_row_per_recipient_per_channel(self, client, estate, db):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()

        rows = (await db.execute(select(CommunicationLog))).scalars().all()
        # three recipients × three channels
        assert len(rows) == 9
        assert {r.recipient_role for r in rows} == {
            "emergency_contact", "doctor", "admin"}
        assert {r.channel for r in rows} == {"voice", "sms", "whatsapp"}

    async def test_rows_exist_before_anything_is_sent(self, client, estate, db):
        """
        Durability is the point: a crash between queueing and sending must
        leave a record of what was owed, for the sweep to pick up.
        """
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        rows = await emergency_comms_service.queue_for_emergency(db, emergency)
        assert all(r.status == "queued" for r in rows)
        assert all(r.attempts == 0 for r in rows)
        assert all(r.provider_sid is None for r in rows)

    async def test_queueing_twice_does_not_double_call(self, client, estate, db):
        """Two background runs for one emergency must not ring a family twice."""
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()
        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()

        rows = (await db.execute(select(CommunicationLog))).scalars().all()
        assert len(rows) == 9

    async def test_numbers_come_from_the_database(self, client, estate, db):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])
        # The clinician only becomes a recipient once assigned; a freshly
        # raised emergency has nobody on it yet.
        emergency.assigned_doctor_id = estate[DOCTOR_A]
        await db.commit()
        await emergency_comms_service.queue_for_emergency(db, emergency)

        by_role = {r.recipient_role: r.recipient_phone
                   for r in (await db.execute(select(CommunicationLog))).scalars()}
        assert by_role["emergency_contact"] == "+919876543210"
        assert by_role["doctor"] == "+912222222222"
        assert by_role["admin"] == "+913333333333"

    async def test_a_recipient_with_no_number_is_still_recorded(
        self, client, estate, db
    ):
        """
        "Nobody tried" and "there was no number" are different facts, and an
        incident review has to tell them apart.
        """
        doctor = await db.get(Doctor, estate[DOCTOR_A])
        doctor.phone = ""   # the column is NOT NULL; blank is "none on file"
        await db.commit()

        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])
        emergency.assigned_doctor_id = estate[DOCTOR_A]
        await db.commit()

        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()

        doctor_rows = [r for r in (await db.execute(select(CommunicationLog))).scalars()
                       if r.recipient_role == "doctor"]
        assert len(doctor_rows) == 3
        assert all(not r.recipient_phone for r in doctor_rows)


# ── sending, retrying, failing ───────────────────────────────────────────

class TestDelivery:
    async def _queued(self, client, db, estate, assign_doctor: bool = True):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])
        if assign_doctor:
            emergency.assigned_doctor_id = estate[DOCTOR_A]
            await db.commit()
        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()
        return emergency

    async def test_success_records_the_provider_sid(
        self, client, estate, db, isolated_transports
    ):
        emergency = await self._queued(client, db, estate)
        counts = await emergency_comms_service.dispatch_pending(
            db, emergency_id=emergency.id)

        # A hand-off the provider took, counted honestly as such.
        assert counts["accepted"] == 9
        rows = (await db.execute(select(CommunicationLog))).scalars().all()
        # `accepted`, not `sent`: the provider took the request, which is not
        # evidence a network carried it. Only a status callback can say more.
        assert all(r.status == "accepted" for r in rows)
        assert all(r.provider_sid == "SM_ok" for r in rows)
        assert all(r.sent_at is not None for r in rows)

    async def test_the_sequence_is_contact_then_doctor_then_admin(
        self, client, estate, db, isolated_transports
    ):
        emergency = await self._queued(client, db, estate)
        await emergency_comms_service.dispatch_pending(db, emergency_id=emergency.id)

        # The gateway records what it was asked to do, in order.
        numbers = [to for _, to, _ in isolated_transports.calls]
        assert numbers.index("+919876543210") < numbers.index("+912222222222")
        assert numbers.index("+912222222222") < numbers.index("+913333333333")

    async def test_a_transient_failure_is_retried_with_backoff(
        self, client, estate, db
    ):
        set_twilio_gateway(FakeGateway(DeliveryResult(
            success=False, retryable=True, error_code="500",
            error_message="Twilio unavailable")))

        emergency = await self._queued(client, db, estate)
        counts = await emergency_comms_service.dispatch_pending(
            db, emergency_id=emergency.id)

        assert counts["retrying"] == 9
        row = (await db.execute(select(CommunicationLog))).scalars().first()
        assert row.status == "queued"
        assert row.attempts == 1
        assert row.next_attempt_at is not None
        # First backoff is the configured base, not zero. SQLite hands back a
        # naive datetime where PostgreSQL is timezone-aware, so both are
        # normalised before comparing.
        scheduled = row.next_attempt_at
        if scheduled.tzinfo is None:
            scheduled = scheduled.replace(tzinfo=timezone.utc)
        assert scheduled > datetime.now(timezone.utc) - timedelta(seconds=5)

    async def test_backoff_grows(self, client, estate, db):
        set_twilio_gateway(FakeGateway(DeliveryResult(
            success=False, retryable=True, error_code="500")))
        emergency = await self._queued(client, db, estate)

        delays = []
        for _ in range(2):
            # Make everything due again so the next attempt runs immediately.
            await db.execute(
                text("UPDATE communication_logs SET next_attempt_at = :t"),
                {"t": datetime.now(timezone.utc) - timedelta(seconds=1)})
            await db.commit()
            await emergency_comms_service.dispatch_pending(
                db, emergency_id=emergency.id)
            row = (await db.execute(select(CommunicationLog))).scalars().first()
            await db.refresh(row)
            if row.next_attempt_at:
                delays.append(row.next_attempt_at)

        assert len(delays) == 2
        assert delays[1] > delays[0], "the second backoff must be longer"

    async def test_a_permanent_failure_is_not_retried(self, client, estate, db):
        """
        Twilio has already said the number is unroutable. Retrying it four
        times does not eventually work; it delays somebody noticing.
        """
        set_twilio_gateway(FakeGateway(DeliveryResult(
            success=False, retryable=False, error_code="21211",
            error_message="Invalid 'To' number")))

        emergency = await self._queued(client, db, estate)
        counts = await emergency_comms_service.dispatch_pending(
            db, emergency_id=emergency.id)

        assert counts["failed"] == 9
        row = (await db.execute(select(CommunicationLog))).scalars().first()
        assert row.status == "failed"
        assert row.next_attempt_at is None
        assert row.error_code == "21211"

    async def test_attempts_stop_at_the_configured_ceiling(self, client, estate, db):
        set_twilio_gateway(FakeGateway(DeliveryResult(
            success=False, retryable=True, error_code="500")))
        emergency = await self._queued(client, db, estate)

        for _ in range(settings.EMERGENCY_COMMS_MAX_ATTEMPTS + 2):
            await db.execute(
                text("UPDATE communication_logs SET next_attempt_at = :t "
                     "WHERE status = 'queued'"),
                {"t": datetime.now(timezone.utc) - timedelta(seconds=1)})
            await db.commit()
            await emergency_comms_service.dispatch_pending(
                db, emergency_id=emergency.id)

        rows = (await db.execute(select(CommunicationLog))).scalars().all()
        assert all(r.status == "failed" for r in rows)
        assert all(r.attempts == settings.EMERGENCY_COMMS_MAX_ATTEMPTS for r in rows)

    async def test_an_unconfigured_channel_is_skipped_not_failed(
        self, client, estate, db, isolated_transports
    ):
        """WhatsApp being unconfigured must not stop the call and the SMS."""
        isolated_transports.whatsapp_configured = False

        emergency = await self._queued(client, db, estate)
        await emergency_comms_service.dispatch_pending(db, emergency_id=emergency.id)

        rows = (await db.execute(select(CommunicationLog))).scalars().all()
        whatsapp = [r for r in rows if r.channel == "whatsapp"]
        others = [r for r in rows if r.channel != "whatsapp"]

        assert all(r.status == "skipped" for r in whatsapp)
        assert all(r.next_attempt_at is None for r in whatsapp)
        assert all(r.status == "accepted" for r in others)

    async def test_a_row_is_only_claimed_once(self, client, estate, db):
        """
        Two sweepers may run. The conditional claim means only one owns an
        attempt, so a family is not telephoned twice.
        """
        emergency = await self._queued(client, db, estate)
        row = (await db.execute(select(CommunicationLog))).scalars().first()

        assert await emergency_comms_service._claim(db, row.id) is True
        assert await emergency_comms_service._claim(db, row.id) is False


# ── Google Maps: absent and present ──────────────────────────────────────

class TestMapsDeferredActivation:
    async def test_no_key_means_no_address_no_hospital_no_eta(
        self, client, estate, db
    ):
        """
        The deployed state today. The emergency must work completely, and
        nothing may be invented to fill the gap.
        """
        headers = await auth(client, PATIENT_A)
        response = await raise_sos(client, headers)
        assert response.status_code == 201, response.text

        emergency = await emergency_of(db, estate[PATIENT_A])
        changed = await emergency_comms_service.enrich_location(db, emergency)

        assert changed is False
        assert emergency.resolved_address is None
        assert emergency.hospital_name is None
        assert emergency.hospital_distance_km is None
        assert emergency.eta is None
        # The key-free map link still works, because it needs no key.
        assert emergency.maps_url and "25.5941" in emergency.maps_url

    async def test_adding_a_key_turns_everything_on_with_no_code_change(
        self, client, estate, db
    ):
        """
        The activation contract: the only change is configuration.
        """
        set_maps_service(FakeMaps(hospital={
            "name": "Patna Medical College Hospital",
            "latitude": 25.6100, "longitude": 85.1400,
            "distance_km": 2.4, "duration_minutes": 7,
        }))

        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        changed = await emergency_comms_service.enrich_location(db, emergency)
        assert changed is True
        assert emergency.resolved_address == "12 Real Street, Patna"
        assert emergency.hospital_name == "Patna Medical College Hospital"
        assert emergency.hospital_distance_km == 2.4
        assert emergency.eta == 7

    async def test_a_hospital_without_an_eta_leaves_the_eta_null(
        self, client, estate, db
    ):
        """A known facility with an unknown ETA beats a guessed one."""
        set_maps_service(FakeMaps(hospital={
            "name": "Some Hospital", "latitude": 25.61, "longitude": 85.14,
        }))
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        await emergency_comms_service.enrich_location(db, emergency)
        assert emergency.hospital_name == "Some Hospital"
        assert emergency.eta is None

    def test_the_service_reads_the_key_at_call_time(self, monkeypatch):
        """
        Not captured at import — that is what makes activation a restart
        rather than a deployment.
        """
        service = MapsService()
        monkeypatch.setattr(settings, "ORS_API_KEY", "")
        assert service.is_enabled() is False
        monkeypatch.setattr(settings, "ORS_API_KEY", "ors-something")
        assert service.is_enabled() is True

    def test_map_links_never_need_a_key(self, monkeypatch):
        service = MapsService()
        monkeypatch.setattr(settings, "ORS_API_KEY", "")
        url = service.build_maps_url(*PATNA)
        assert "25.5941" in url and "85.1376" in url

    async def test_routing_lookups_return_none_without_a_key(self, monkeypatch):
        """
        The ORS-backed lookups degrade to None, never an exception.

        Geocoding and hospital search are deliberately excluded: they run
        keyless through Nominatim and Overpass, so they keep working when
        routing is unconfigured. That is a change from the Google
        implementation, where an unset key meant no address at all.
        """
        service = MapsService()
        monkeypatch.setattr(settings, "ORS_API_KEY", "")
        assert await service.distance_matrix(PATNA, PATNA) is None
        assert await service.route_geometry(PATNA, PATNA) is None

    async def test_every_lookup_survives_an_upstream_outage(self, monkeypatch):
        """
        With every upstream unreachable, nothing raises and nothing is invented.

        Stubbed rather than left to hit the live network: an emergency test that
        depends on Nominatim being up is neither fast nor trustworthy.
        """
        service = MapsService()
        monkeypatch.setattr(settings, "ORS_API_KEY", "ors-something")

        async def dead_ors(*_args, **_kwargs):
            return None

        async def dead_osm(*_args, **_kwargs):
            return None

        monkeypatch.setattr(service, "_ors_post", dead_ors)
        monkeypatch.setattr(service, "_osm_get", dead_osm)
        monkeypatch.setattr(
            "app.services.maps.OVERPASS_URL", "http://127.0.0.1:9/unreachable"
        )

        assert await service.reverse_geocode(*PATNA) is None
        assert await service.forward_geocode("anywhere") == []
        assert await service.find_nearby_hospitals(*PATNA) == []
        assert await service.distance_matrix(PATNA, PATNA) is None
        assert await service.nearest_hospital_with_eta(*PATNA) is None


# ── templates ────────────────────────────────────────────────────────────

class TestTemplates:
    async def test_sms_carries_the_facts_a_responder_needs(
        self, client, estate, db
    ):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])
        patient = await db.get(Patient, estate[PATIENT_A])

        body = templates.sms_body(templates.build_context(emergency, patient))
        assert "Comms Patient" in body
        assert str(emergency.id)[:8] in body
        assert "25.59" in body
        assert "openstreetmap.org" in body

    async def test_unknown_fields_are_omitted_not_filled_in(
        self, client, estate, db
    ):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        body = templates.sms_body(templates.build_context(emergency))
        assert "N/A" not in body
        assert "None" not in body
        assert "Nearest hospital" not in body  # no hospital known
        assert "ETA" not in body

    async def test_the_voice_script_has_no_url(self, client, estate, db):
        """Nobody can write down a URL read aloud by a synthetic voice."""
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        ctx = templates.build_context(emergency)
        for key in (templates.TEMPLATE_VOICE_CONTACT,
                    templates.TEMPLATE_VOICE_DOCTOR,
                    templates.TEMPLATE_VOICE_ADMIN):
            script = templates.voice_script(key, ctx)
            assert "http" not in script
            assert len(script) < 400

    async def test_whatsapp_includes_the_map_link(self, client, estate, db):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])

        body = templates.whatsapp_body(templates.build_context(emergency))
        assert "openstreetmap.org" in body
        assert "MEDBRIDGE EMERGENCY SOS" in body


# ── the API surface ──────────────────────────────────────────────────────

class TestEndpoints:
    async def test_communications_endpoint_masks_numbers(
        self, client, estate, db
    ):
        headers = await auth(client, PATIENT_A)
        eid = (await raise_sos(client, headers)).json()["id"]
        emergency = await emergency_of(db, estate[PATIENT_A])
        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()

        r = await client.get(f"/api/v1/patient/sos/{eid}/communications",
                             headers=headers)
        assert r.status_code == 200, r.text
        entries = r.json()["communications"]
        assert len(entries) == 9
        assert all("recipient_phone" not in e for e in entries)
        masked = {e["recipient_phone_masked"] for e in entries}
        assert "+919876543210" not in masked
        assert any("•" in (m or "") for m in masked)

    async def test_timeline_merges_status_and_communications(
        self, client, estate, db
    ):
        headers = await auth(client, PATIENT_A)
        eid = (await raise_sos(client, headers)).json()["id"]
        emergency = await emergency_of(db, estate[PATIENT_A])
        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()

        r = await client.get(f"/api/v1/patient/sos/{eid}/timeline", headers=headers)
        assert r.status_code == 200, r.text
        kinds = {e["kind"] for e in r.json()["entries"]}
        assert "status" in kinds
        assert "communication" in kinds

    async def test_hospital_endpoint_is_honest_when_maps_is_off(
        self, client, estate
    ):
        headers = await auth(client, PATIENT_A)
        eid = (await raise_sos(client, headers)).json()["id"]

        r = await client.get(f"/api/v1/patient/sos/{eid}/hospital", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is False
        assert body["hospital_name"] is None
        assert body["eta_minutes"] is None
        assert "not enabled" in (body["reason"] or "").lower()

    async def test_another_patient_cannot_read_the_communications(
        self, client, estate, db
    ):
        other = User(email="comms.other@aronofy.com",
                     hashed_password=get_password_hash(PW),
                     role="patient", is_active=True, is_verified=True)
        db.add(other)
        await db.flush()
        db.add(Patient(id=other.id, first_name="Other", last_name="Patient",
                       phone="+919999999999", date_of_birth="1990-01-01",
                       gender="other"))
        await db.commit()

        headers = await auth(client, PATIENT_A)
        eid = (await raise_sos(client, headers)).json()["id"]

        intruder = await auth(client, "comms.other@aronofy.com")
        for path in ("communications", "timeline", "hospital"):
            r = await client.get(f"/api/v1/patient/sos/{eid}/{path}",
                                 headers=intruder)
            assert r.status_code == 404, path

    async def test_anonymous_cannot_read_them(self, client, estate):
        headers = await auth(client, PATIENT_A)
        eid = (await raise_sos(client, headers)).json()["id"]
        for path in ("communications", "timeline", "hospital"):
            assert (await client.get(
                f"/api/v1/patient/sos/{eid}/{path}")).status_code == 401


# ── secrets ──────────────────────────────────────────────────────────────

class TestSecrets:
    def test_no_credential_is_hardcoded(self):
        """Everything comes from the environment via settings."""
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "app"
        for module in ("services/twilio_gateway.py", "services/maps.py",
                       "services/emergency_comms.py"):
            source = (root / module).read_text(encoding="utf-8")

            # A literal Twilio SID or Google key would be a committed secret.
            assert not re.search(r"AC[0-9a-fA-F]{32}", source)
            assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", source)
            # And no credential is ever assigned from a literal.
            for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                         "TWILIO_PHONE_NUMBER", "GOOGLE_MAPS_API_KEY"):
                assert not re.search(rf'{name}\s*=\s*["\']', source), name

    def test_credentials_are_read_through_settings(self):
        from pathlib import Path

        gateway = (Path(__file__).resolve().parents[1] / "app" / "services"
                   / "twilio_gateway.py").read_text(encoding="utf-8")
        assert "settings.TWILIO_ACCOUNT_SID" in gateway
        assert "settings.TWILIO_AUTH_TOKEN" in gateway

        maps = (Path(__file__).resolve().parents[1] / "app" / "services"
                / "maps.py").read_text(encoding="utf-8")
        assert "settings.ORS_API_KEY" in maps

    def test_no_phone_number_is_hardcoded(self):
        from pathlib import Path
        import re

        source = (Path(__file__).resolve().parents[1]
                  / "app" / "services" / "emergency_comms.py").read_text(encoding="utf-8")
        # A literal dialable number would be a number somebody actually rings.
        assert not re.search(r'"\+\d{7,}"', source)

    def test_masking_hides_the_number(self):
        assert mask_phone("+919876543210") != "+919876543210"
        assert "9876543" not in mask_phone("+919876543210")
        assert mask_phone(None) is None


# -- delivery lifecycle (Twilio status callbacks) --------------------------

class TestDeliveryLifecycle:
    """
    A provider acknowledgement is not a delivery.

    Creating a Twilio message returns `queued`/`accepted`; everything after
    that arrives asynchronously as a callback. These tests pin the translation,
    the forward-only rule, and the fact that nothing is ever inferred.
    """

    async def _accepted(self, client, db, estate, channel="sms", sid=None):
        sid = sid or f"SID_{channel}"
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])
        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()
        await emergency_comms_service.dispatch_pending(db, emergency_id=emergency.id)
        row = (await db.execute(
            select(CommunicationLog)
            .where(CommunicationLog.channel == channel)
            .where(CommunicationLog.recipient_role == "emergency_contact")
        )).scalars().first()
        row.provider_sid = sid
        await db.commit()
        return row

    async def test_a_message_walks_the_lifecycle(self, client, estate, db):
        from app.services.twilio_callbacks import twilio_callback_service

        row = await self._accepted(client, db, estate)
        assert row.status == "accepted"

        for reported, expected in (("sent", "sent"), ("delivered", "delivered")):
            await twilio_callback_service.record(db, "sms", {
                "MessageSid": "SID_sms", "MessageStatus": reported})
            await db.commit()
            await db.refresh(row)
            assert row.status == expected, reported

        assert row.completed_at is not None
        assert row.next_attempt_at is None

    async def test_every_callback_is_persisted(self, client, estate, db):
        from app.services.twilio_callbacks import twilio_callback_service

        row = await self._accepted(client, db, estate)
        for reported in ("sent", "delivered"):
            await twilio_callback_service.record(db, "sms", {
                "MessageSid": "SID_sms", "MessageStatus": reported})
        await db.commit()
        await db.refresh(row)

        assert [e["provider_status"] for e in row.provider_events] == [
            "sent", "delivered"]
        assert all(e["at"] for e in row.provider_events)

    async def test_a_late_callback_cannot_walk_it_backwards(
        self, client, estate, db
    ):
        """
        Twilio does not guarantee ordering. A `sent` arriving after a
        `delivered` must not un-deliver a message that reached the handset.
        """
        from app.services.twilio_callbacks import twilio_callback_service

        row = await self._accepted(client, db, estate)
        await twilio_callback_service.record(db, "sms", {
            "MessageSid": "SID_sms", "MessageStatus": "delivered"})
        await twilio_callback_service.record(db, "sms", {
            "MessageSid": "SID_sms", "MessageStatus": "sent"})
        await db.commit()
        await db.refresh(row)

        assert row.status == "delivered"
        # The late callback is still recorded -- it happened.
        assert len(row.provider_events) == 2

    async def test_undelivered_is_kept_apart_from_failed(
        self, client, estate, db
    ):
        from app.services.twilio_callbacks import twilio_callback_service

        row = await self._accepted(client, db, estate)
        await twilio_callback_service.record(db, "sms", {
            "MessageSid": "SID_sms", "MessageStatus": "undelivered",
            "ErrorCode": "30003"})
        await db.commit()
        await db.refresh(row)

        assert row.status == "undelivered"
        assert row.error_code == "30003"
        assert row.next_attempt_at is None

    async def test_a_call_records_the_duration_it_reports(
        self, client, estate, db
    ):
        from app.services.twilio_callbacks import twilio_callback_service

        row = await self._accepted(client, db, estate, channel="voice",
                                   sid="CA_life")
        await twilio_callback_service.record(db, "voice", {
            "CallSid": "CA_life", "CallStatus": "completed",
            "CallDuration": "37"})
        await db.commit()
        await db.refresh(row)

        assert row.status == "delivered"
        assert row.duration_seconds == 37

    async def test_a_callback_for_a_sid_we_never_sent_is_ignored(
        self, db, estate
    ):
        from app.services.twilio_callbacks import twilio_callback_service

        assert await twilio_callback_service.record(db, "sms", {
            "MessageSid": "SM_not_ours", "MessageStatus": "delivered"}) is None

    @pytest.mark.parametrize("reported,expected", [
        ("queued", "queued"), ("initiated", "accepted"), ("ringing", "sent"),
        ("in-progress", "sent"), ("completed", "delivered"),
        ("busy", "undelivered"), ("no-answer", "undelivered"),
        ("canceled", "canceled"), ("failed", "failed"),
    ])
    async def test_call_status_translation(self, reported, expected):
        from app.services.twilio_callbacks import translate

        assert translate("voice", reported) == expected

    @pytest.mark.parametrize("reported,expected", [
        ("accepted", "accepted"), ("sent", "sent"), ("delivered", "delivered"),
        ("read", "delivered"), ("undelivered", "undelivered"),
        ("failed", "failed"), ("canceled", "canceled"),
    ])
    async def test_message_status_translation(self, reported, expected):
        from app.services.twilio_callbacks import translate

        assert translate("sms", reported) == expected

    async def test_an_unrecognised_provider_status_is_not_invented(self):
        from app.services.twilio_callbacks import translate

        assert translate("sms", "teleported") is None


class TestCallbackSecurity:
    """
    The endpoint is public -- Twilio cannot present a bearer token -- so the
    signature is the only thing between it and anyone who guesses a SID.
    """

    async def test_an_unsigned_callback_is_refused(self, client, estate):
        r = await client.post("/api/v1/webhooks/twilio/sms-status",
                              data={"MessageSid": "SM_x",
                                    "MessageStatus": "delivered"})
        assert r.status_code == 403

    async def test_a_wrongly_signed_callback_is_refused(self, client, estate):
        r = await client.post("/api/v1/webhooks/twilio/sms-status",
                              data={"MessageSid": "SM_x",
                                    "MessageStatus": "delivered"},
                              headers={"X-Twilio-Signature": "nonsense"})
        assert r.status_code == 403

    async def test_verification_fails_closed_without_a_token(self, monkeypatch):
        from app.services.twilio_callbacks import twilio_callback_service

        monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "")
        assert twilio_callback_service.verify_signature(
            "https://x/y", {}, "sig") is False

    async def test_verification_fails_closed_without_a_signature(self, monkeypatch):
        from app.services.twilio_callbacks import twilio_callback_service

        monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "t" * 32)
        assert twilio_callback_service.verify_signature(
            "https://x/y", {}, None) is False


class TestCallbackActivation:
    async def test_no_public_url_means_no_callback_is_requested(self, monkeypatch):
        """
        Pointing Twilio at a host it cannot reach makes it retry and fills the
        log with failures that say nothing about the message.
        """
        from app.services.twilio_callbacks import callback_url

        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
        assert callback_url("sms") is None

    async def test_setting_the_public_url_activates_callbacks(self, monkeypatch):
        """Configuration only -- no code change."""
        from app.services.twilio_callbacks import callback_url

        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://api.example.com")
        assert callback_url("sms") == (
            "https://api.example.com/api/v1/webhooks/twilio/sms-status")
        assert callback_url("voice").endswith("/voice-status")
        assert callback_url("whatsapp").endswith("/whatsapp-status")

    async def test_a_trailing_slash_does_not_double_up(self, monkeypatch):
        from app.services.twilio_callbacks import callback_url

        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://api.example.com/")
        assert "//api/v1" not in callback_url("sms")

    async def test_without_a_callback_url_the_attempt_stops_at_accepted(
        self, client, estate, db, monkeypatch
    ):
        """
        Honesty over optimism: with nowhere to hear back from, the record stops
        at the provider's acknowledgement rather than claiming a delivery
        nobody confirmed.
        """
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
        row = await self._stop(client, db, estate)
        assert row.status == "accepted"
        assert row.completed_at is not None   # nothing more will ever arrive

    async def _stop(self, client, db, estate):
        headers = await auth(client, PATIENT_A)
        await raise_sos(client, headers)
        emergency = await emergency_of(db, estate[PATIENT_A])
        await emergency_comms_service.queue_for_emergency(db, emergency)
        await db.commit()
        await emergency_comms_service.dispatch_pending(db, emergency_id=emergency.id)
        return (await db.execute(
            select(CommunicationLog).where(CommunicationLog.channel == "sms")
        )).scalars().first()

    async def test_with_a_callback_url_the_attempt_stays_open(
        self, client, estate, db, monkeypatch
    ):
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://api.example.com")
        row = await self._stop(client, db, estate)
        assert row.status == "accepted"
        assert row.completed_at is None      # waiting for the callback


class TestWhatsAppActivation:
    """
    `TWILIO_WHATSAPP_NUMBER` degrades safely while absent and activates later
    with no code change.
    """

    def _twilio(self, monkeypatch, whatsapp=""):
        monkeypatch.setattr(settings, "EMERGENCY_COMMS_ENABLED", True)
        monkeypatch.setattr(settings, "TWILIO_ACCOUNT_SID", "AC" + "1" * 32)
        monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "t" * 32)
        monkeypatch.setattr(settings, "TWILIO_PHONE_NUMBER", "+15550000000")
        monkeypatch.setattr(settings, "TWILIO_WHATSAPP_NUMBER", whatsapp)

    async def test_an_absent_number_disables_only_whatsapp(self, monkeypatch):
        from app.services.twilio_gateway import TwilioGateway

        self._twilio(monkeypatch, whatsapp="")
        gateway = TwilioGateway()
        assert gateway.is_configured() is True          # voice and SMS still go
        assert gateway.is_whatsapp_configured() is False

    async def test_adding_the_number_activates_whatsapp(self, monkeypatch):
        from app.services.twilio_gateway import TwilioGateway

        self._twilio(monkeypatch, whatsapp="+14155238886")
        assert TwilioGateway().is_whatsapp_configured() is True

    async def test_either_sender_form_is_accepted(self):
        from app.services.twilio_gateway import TwilioGateway

        assert TwilioGateway._whatsapp_address("+14155238886") == \
            "whatsapp:+14155238886"
        assert TwilioGateway._whatsapp_address("whatsapp:+14155238886") == \
            "whatsapp:+14155238886"

    async def test_an_unconfigured_whatsapp_records_a_reason(self, monkeypatch):
        from app.services.twilio_gateway import TwilioGateway

        monkeypatch.setattr(settings, "TWILIO_WHATSAPP_NUMBER", "")
        result = await TwilioGateway().send_whatsapp("+15005550001", "body")
        assert result.skipped is True
        assert result.retryable is False
        assert "not configured" in (result.error_message or "").lower()
