"""
HTTP integration tests for the Medical Case Intake Agent.

Exercises the real FastAPI app, real JWT auth, and the real SQLite test
database. Only the LLM is faked, so persistence, RBAC, error mapping and the
full multi-turn flow are genuinely covered.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.security import get_password_hash
from app.intake.dependencies import get_llm
from app.main import app
from app.models.case import Case, Symptom
from app.models.doctor import Doctor
from app.models.intake import IntakeExtractedEntity, IntakeSessionRecord
from app.models.patient import Patient
from app.models.user import User
from tests.intake.conftest import DeadLLM, ScriptedLLM, complete_extraction, extraction

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/ai/intake"
COMPLETE_SYMPTOMS = "I have had moderate chest discomfort for 3 days now."


@pytest.fixture
async def seeded(db):
    """A patient, a cardiologist, and a second patient for isolation checks."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in (
        "intake_extracted_entities",
        "intake_sessions",
        "symptoms",
        "cases",
        "doctors",
        "patients",
        "users",
    ):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    patient_user = User(
        email="intake.patient@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    other_user = User(
        email="intake.other@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    doctor_user = User(
        email="intake.cardio@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True,
    )
    db.add_all([patient_user, other_user, doctor_user])
    await db.flush()

    db.add_all(
        [
            Patient(
                id=patient_user.id,
                first_name="Asha",
                last_name="Verma",
                phone="+911234567890",
                date_of_birth="1990-04-12",
                gender="female",
            ),
            Patient(
                id=other_user.id,
                first_name="Other",
                last_name="Patient",
                phone="+911111111111",
                date_of_birth="1988-01-01",
                gender="male",
            ),
            Doctor(
                id=doctor_user.id,
                first_name="Rajesh",
                last_name="Sharma",
                phone="+919876543210",
                specialty="Cardiology",
                license_number="MD-INTAKE-001",
                availability="available",
                verification_status="verified",
                rating=4.9,
                years_of_experience=18,
                hospital_name="Max Super Speciality Hospital",
            ),
        ]
    )
    await db.commit()

    return {
        "patient_id": str(patient_user.id),
        "other_id": str(other_user.id),
        "doctor_id": str(doctor_user.id),
    }


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "password123"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _use_llm(llm) -> None:
    app.dependency_overrides[get_llm] = lambda: llm


# ------------------------------------------------------------- Auth / RBAC


class TestAccessControl:
    async def test_requires_authentication(self, client, seeded):
        resp = await client.post(f"{BASE}/sessions", json={"symptoms": "headache"})
        assert resp.status_code == 401

    async def test_doctors_cannot_start_patient_intake(self, client, seeded):
        _use_llm(ScriptedLLM())
        headers = await _login(client, "intake.cardio@aronofy.com")
        resp = await client.post(
            f"{BASE}/sessions", json={"symptoms": "headache"}, headers=headers
        )
        assert resp.status_code == 403

    async def test_patient_cannot_read_another_patients_session(self, client, seeded):
        _use_llm(ScriptedLLM({"extraction": complete_extraction()}))
        owner = await _login(client, "intake.patient@aronofy.com")
        start = await client.post(
            f"{BASE}/sessions", json={"symptoms": COMPLETE_SYMPTOMS}, headers=owner
        )
        session_id = start.json()["session_id"]

        intruder = await _login(client, "intake.other@aronofy.com")
        resp = await client.get(f"{BASE}/sessions/{session_id}", headers=intruder)
        assert resp.status_code == 403


# ------------------------------------------------------------- Validation


class TestRequestValidation:
    @pytest.mark.parametrize("payload", [{}, {"symptoms": ""}, {"symptoms": "   "}])
    async def test_invalid_start_payloads_are_rejected(self, client, seeded, payload):
        _use_llm(ScriptedLLM())
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.post(f"{BASE}/sessions", json=payload, headers=headers)
        assert resp.status_code == 422

    async def test_oversized_input_is_rejected(self, client, seeded):
        _use_llm(ScriptedLLM())
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/sessions", json={"symptoms": "a" * 5000}, headers=headers
        )
        assert resp.status_code == 422

    async def test_unknown_session_returns_404(self, client, seeded):
        _use_llm(ScriptedLLM())
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.get(f"{BASE}/sessions/does-not-exist", headers=headers)
        assert resp.status_code == 404

    async def test_malformed_doctor_id_is_rejected(self, client, seeded):
        _use_llm(ScriptedLLM({"extraction": complete_extraction()}))
        headers = await _login(client, "intake.patient@aronofy.com")
        start = await client.post(
            f"{BASE}/sessions", json={"symptoms": COMPLETE_SYMPTOMS}, headers=headers
        )
        session_id = start.json()["session_id"]

        resp = await client.post(
            f"{BASE}/sessions/{session_id}/select-doctor",
            json={"doctor_id": "not-a-uuid"},
            headers=headers,
        )
        assert resp.status_code == 422


# --------------------------------------------------------- End-to-end flow


class TestFullIntakeFlow:
    async def test_complete_flow_persists_case_and_audit(self, client, seeded, db):
        _use_llm(
            ScriptedLLM(
                {
                    "extraction": extraction(
                        ("symptom", "chest discomfort", 0.93, "chest discomfort"),
                        ("duration", "3 days", 0.91, "for 3 days"),
                        ("severity", "moderate", 0.85, "moderate"),
                    ),
                    "case": {
                        "chief_complaint": "Chest discomfort on exertion",
                        "differential_considerations": ["Stable angina"],
                        "recommended_specialty": "Cardiology",
                        "urgency": "high",
                        "summary_for_doctor": "Requires cardiology review.",
                    },
                }
            )
        )
        headers = await _login(client, "intake.patient@aronofy.com")

        # 1. Start
        start = await client.post(
            f"{BASE}/sessions", json={"symptoms": COMPLETE_SYMPTOMS}, headers=headers
        )
        assert start.status_code == 201
        body = start.json()
        assert body["status"] == "awaiting_doctor_selection"
        assert body["medical_case"]["recommended_specialty"] == "Cardiology"
        assert body["medical_case"]["urgency"] == "high"
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["doctor_id"] == seeded["doctor_id"]

        session_id = body["session_id"]

        # 2. Read back
        fetched = await client.get(f"{BASE}/sessions/{session_id}", headers=headers)
        assert fetched.status_code == 200
        assert fetched.json()["session_id"] == session_id

        # 3. Route to the doctor
        routed = await client.post(
            f"{BASE}/sessions/{session_id}/select-doctor",
            json={"doctor_id": seeded["doctor_id"]},
            headers=headers,
        )
        assert routed.status_code == 201, routed.text
        routing = routed.json()
        assert routing["specialty"] == "Cardiology"
        assert routing["urgency"] == "high"

        # 4. A real clinical case row exists and is assigned
        case = (
            await db.execute(
                select(Case).where(Case.id == uuid.UUID(routing["case_id"]))
            )
        ).scalars().first()
        assert case is not None
        assert str(case.doctor_id) == seeded["doctor_id"]
        assert str(case.patient_id) == seeded["patient_id"]
        assert case.status == "routed"
        assert case.urgency_level == "high"
        assert case.specialty == "Cardiology"
        assert "chest discomfort" in case.ai_extracted_symptoms

        # 5. Symptom rows written
        symptoms = (
            await db.execute(select(Symptom).where(Symptom.case_id == case.id))
        ).scalars().all()
        assert [s.name for s in symptoms] == ["chest discomfort"]
        assert symptoms[0].severity in ("mild", "moderate", "severe")

        # 6. Audit trail written with provenance
        record = (
            await db.execute(
                select(IntakeSessionRecord).where(
                    IntakeSessionRecord.session_key == session_id
                )
            )
        ).scalars().first()
        assert record is not None
        assert record.status == "routed"
        assert str(record.routed_case_id) == routing["case_id"]

        entities = (
            await db.execute(
                select(IntakeExtractedEntity).where(
                    IntakeExtractedEntity.session_id == record.id
                )
            )
        ).scalars().all()
        assert len(entities) == 3
        assert all(e.evidence_quote for e in entities)
        assert all(0.0 <= e.confidence <= 1.0 for e in entities)
        assert all(e.was_accepted for e in entities)

    async def test_multi_turn_flow_completes(self, client, seeded):
        _use_llm(
            ScriptedLLM(
                {"extraction": extraction(("symptom", "malaise", 0.6, "I feel unwell"))}
            )
        )
        headers = await _login(client, "intake.patient@aronofy.com")

        start = await client.post(
            f"{BASE}/sessions", json={"symptoms": "I feel unwell"}, headers=headers
        )
        body = start.json()
        assert body["status"] == "collecting"
        assert body["awaiting_input"] is True
        assert body["pending_question"]
        session_id = body["session_id"]

        _use_llm(
            ScriptedLLM(
                {
                    "extraction": extraction(
                        ("symptom", "malaise", 0.85, "I feel unwell"),
                        ("duration", "3 days", 0.9, "about 3 days"),
                        ("severity", "moderate", 0.85, "moderate"),
                    ),
                    "case": {"recommended_specialty": "Cardiology", "urgency": "medium"},
                }
            )
        )
        turn = await client.post(
            f"{BASE}/sessions/{session_id}/turns",
            json={"answer": "about 3 days and it is moderate"},
            headers=headers,
        )
        assert turn.status_code == 200
        assert turn.json()["status"] == "awaiting_doctor_selection"

    async def test_cannot_route_before_case_is_ready(self, client, seeded):
        _use_llm(
            ScriptedLLM(
                {"extraction": extraction(("symptom", "malaise", 0.6, "I feel unwell"))}
            )
        )
        headers = await _login(client, "intake.patient@aronofy.com")
        start = await client.post(
            f"{BASE}/sessions", json={"symptoms": "I feel unwell"}, headers=headers
        )
        session_id = start.json()["session_id"]

        resp = await client.post(
            f"{BASE}/sessions/{session_id}/select-doctor",
            json={"doctor_id": seeded["doctor_id"]},
            headers=headers,
        )
        assert resp.status_code == 422


# --------------------------------------------------------------- Emergency


class TestEmergencyOverHttp:
    async def test_emergency_escalates_and_blocks_routing(self, client, seeded):
        _use_llm(ScriptedLLM({"extraction": complete_extraction()}))
        headers = await _login(client, "intake.patient@aronofy.com")

        start = await client.post(
            f"{BASE}/sessions",
            json={"symptoms": "I have crushing chest pain and cannot breathe"},
            headers=headers,
        )
        body = start.json()

        assert body["status"] == "emergency_escalated"
        assert body["is_emergency"] is True
        assert body["pending_question"] is None
        assert body["medical_case"]["urgency"] == "critical"
        assert len(body["red_flags"]) >= 2
        assert any("emergency" in n.casefold() for n in body["notices"])

        resp = await client.post(
            f"{BASE}/sessions/{body['session_id']}/select-doctor",
            json={"doctor_id": seeded["doctor_id"]},
            headers=headers,
        )
        assert resp.status_code == 422

    async def test_hindi_emergency_escalates(self, client, seeded):
        _use_llm(ScriptedLLM())
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/sessions",
            json={"symptoms": "मुझे सीने में दर्द है और सांस नहीं आ रही"},
            headers=headers,
        )
        assert resp.json()["status"] == "emergency_escalated"


# ------------------------------------------------------- Fabrication / infra


class TestSafetyOverHttp:
    async def test_fabricated_allergy_never_reaches_the_case(self, client, seeded, db):
        _use_llm(
            ScriptedLLM(
                {
                    "extraction": extraction(
                        ("symptom", "chest discomfort", 0.93, "chest discomfort"),
                        ("duration", "3 days", 0.91, "for 3 days"),
                        ("severity", "moderate", 0.85, "moderate"),
                        (
                            "allergy",
                            "penicillin",
                            0.99,
                            "I am severely allergic to penicillin",
                        ),
                    ),
                    "case": {"recommended_specialty": "Cardiology", "urgency": "medium"},
                }
            )
        )
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/sessions", json={"symptoms": COMPLETE_SYMPTOMS}, headers=headers
        )
        body = resp.json()

        assert body["medical_case"]["allergies"] == []
        assert body["rejected_extraction_count"] == 1
        assert not any(
            "penicillin" in str(e).casefold() for e in body["medical_case"]["symptoms"]
        )

    async def test_llm_outage_degrades_without_500(self, client, seeded):
        _use_llm(DeadLLM())
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/sessions", json={"symptoms": "my head hurts"}, headers=headers
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "collecting"
        assert body["degraded"] is True
        assert body["pending_question"]

    async def test_database_failure_does_not_silently_succeed(
        self, client, seeded, monkeypatch
    ):
        """
        A failed clinical write must surface as an error and must not leave the
        session marked as routed.

        The exception is asserted directly rather than as a 500 response:
        Starlette's ServerErrorMiddleware re-raises through ASGI transport in
        tests, so propagation here is what a 500 looks like in production.
        """
        _use_llm(ScriptedLLM({"extraction": complete_extraction()}))
        headers = await _login(client, "intake.patient@aronofy.com")
        start = await client.post(
            f"{BASE}/sessions", json={"symptoms": COMPLETE_SYMPTOMS}, headers=headers
        )
        session_id = start.json()["session_id"]

        async def boom(*args, **kwargs):
            raise RuntimeError("simulated database outage")

        monkeypatch.setattr(
            "app.intake.infrastructure.repositories.SqlCaseRepository.persist_case",
            boom,
        )

        with pytest.raises(RuntimeError, match="simulated database outage"):
            await client.post(
                f"{BASE}/sessions/{session_id}/select-doctor",
                json={"doctor_id": seeded["doctor_id"]},
                headers=headers,
            )

        monkeypatch.undo()

        # The session must remain routable rather than being left half-finished.
        after = await client.get(f"{BASE}/sessions/{session_id}", headers=headers)
        assert after.json()["status"] == "awaiting_doctor_selection"
        assert after.json()["routed_case_id"] is None


# ------------------------------------------------------------------ Health


class TestIntakeHealth:
    async def test_reports_healthy_llm(self, client, seeded):
        _use_llm(ScriptedLLM())
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.get(f"{BASE}/health", headers=headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["graph_nodes"] == 9
        assert body["max_followup_rounds"] >= 1

    async def test_reports_degraded_when_llm_unreachable(self, client, seeded):
        _use_llm(DeadLLM())
        headers = await _login(client, "intake.patient@aronofy.com")
        resp = await client.get(f"{BASE}/health", headers=headers)
        assert resp.json()["status"] == "degraded"


# ---------------------------------------------------- Legacy non-regression


class TestLegacyEndpointUntouched:
    async def test_legacy_symptom_intake_route_still_exists(self, client, seeded):
        """The pre-existing single-shot endpoint must remain mounted."""
        paths = app.openapi()["paths"]
        assert "/api/v1/ai/symptom-intake" in paths
        assert "/api/v1/ai/chat" in paths
