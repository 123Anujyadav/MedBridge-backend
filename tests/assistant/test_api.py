"""
HTTP integration tests for the AI Medical Assistant.

Exercises the real FastAPI app, real JWT auth and the real test database. Only
the LLM is faked, so routing, RBAC, persistence, memory and error mapping are
genuinely covered.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.assistant.dependencies import get_assistant_llm, get_retriever
from app.core.security import get_password_hash
from app.main import app
from app.models.assistant import AssistantConversation, AssistantMessage
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.user import User
from tests.assistant.conftest import (
    DeadLLM,
    FakeRetriever,
    ScriptedLLM,
    answer_payload,
)

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/ai/assistant"


@pytest.fixture
async def seeded(db):
    """A patient, a second patient for isolation checks, and a doctor."""
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("assistant_messages", "assistant_conversations", "patients",
                  "doctors", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    patient_user = User(
        email="chat.patient@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    other_user = User(
        email="chat.other@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True,
    )
    doctor_user = User(
        email="chat.doctor@aronofy.com",
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
            # Approved, so the assistant's role guard is what denies this
            # doctor rather than a missing administrator approval.
            Doctor(
                id=doctor_user.id,
                first_name="Chat",
                last_name="Doctor",
                phone="+912222222222",
                specialty="Neurology",
                license_number="LIC-CHAT-1",
                verification_status="verified",
            ),
        ]
    )
    await db.commit()
    return {"patient_id": str(patient_user.id), "other_id": str(other_user.id)}


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "password123"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _use(llm, retriever=None) -> None:
    app.dependency_overrides[get_assistant_llm] = lambda: llm
    app.dependency_overrides[get_retriever] = lambda: retriever or FakeRetriever()


# ------------------------------------------------------------- Access


class TestAccessControl:
    async def test_requires_authentication(self, client, seeded):
        resp = await client.post(f"{BASE}/messages", json={"message": "hi"})
        assert resp.status_code == 401

    async def test_doctors_cannot_use_the_patient_assistant(self, client, seeded):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.doctor@aronofy.com")
        resp = await client.post(
            f"{BASE}/messages", json={"message": "hi"}, headers=headers
        )
        assert resp.status_code == 403

    async def test_patient_cannot_read_another_patients_conversation(
        self, client, seeded
    ):
        _use(ScriptedLLM())
        owner = await _login(client, "chat.patient@aronofy.com")
        created = await client.post(
            f"{BASE}/messages", json={"message": "I have a headache"}, headers=owner
        )
        conversation_id = created.json()["conversation_id"]

        intruder = await _login(client, "chat.other@aronofy.com")
        resp = await client.get(
            f"{BASE}/conversations/{conversation_id}", headers=intruder
        )
        assert resp.status_code == 404

    async def test_history_is_scoped_to_the_caller(self, client, seeded):
        _use(ScriptedLLM())
        owner = await _login(client, "chat.patient@aronofy.com")
        await client.post(
            f"{BASE}/messages", json={"message": "I have a headache"}, headers=owner
        )

        intruder = await _login(client, "chat.other@aronofy.com")
        resp = await client.get(f"{BASE}/conversations", headers=intruder)
        assert resp.status_code == 200
        assert resp.json() == []


# ------------------------------------------------------------- Validation


class TestValidation:
    @pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "   "}])
    async def test_rejects_invalid_payloads(self, client, seeded, payload):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.post(f"{BASE}/messages", json=payload, headers=headers)
        assert resp.status_code == 422

    async def test_rejects_oversized_message(self, client, seeded):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/messages", json={"message": "a" * 5000}, headers=headers
        )
        assert resp.status_code == 422

    async def test_unknown_conversation_returns_404(self, client, seeded):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/messages",
            json={"message": "hi", "conversation_id": "does-not-exist"},
            headers=headers,
        )
        assert resp.status_code == 404


# ------------------------------------------------------------- Flow


class TestConversationFlow:
    async def test_send_message_returns_cards_and_persists(self, client, seeded, db):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")

        resp = await client.post(
            f"{BASE}/messages",
            json={"message": "I have had a headache for three days"},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()

        # Shape the existing React page consumes.
        assert body["conversation_id"]
        assert body["message"]["sender"] == "ai"
        structured = body["message"]["structured_data"]
        assert structured["summary"]
        assert structured["symptoms"] == ["headache", "nausea"]
        assert structured["medicines"][0]["sideEffects"] == ["Nausea"]
        assert body["emergency_risk"] == "normal"
        assert body["detected_symptoms"]
        assert body["suggested_specialist"] == "Neurology"

        # Persisted: one conversation, two messages (user + ai).
        conversation = (
            await db.execute(
                select(AssistantConversation).where(
                    AssistantConversation.conversation_key == body["conversation_id"]
                )
            )
        ).scalars().first()
        assert conversation is not None
        assert str(conversation.patient_user_id) == seeded["patient_id"]
        assert conversation.title == "Headache consultation"

        messages = (
            await db.execute(
                select(AssistantMessage)
                .where(AssistantMessage.conversation_id == conversation.id)
                .order_by(AssistantMessage.created_at)
            )
        ).scalars().all()
        assert [m.role for m in messages] == ["user", "ai"]
        assert messages[1].structured["summary"]
        assert messages[1].confidence > 0

    async def test_second_turn_continues_the_same_conversation(
        self, client, seeded, db
    ):
        llm = ScriptedLLM()
        _use(llm)
        headers = await _login(client, "chat.patient@aronofy.com")

        first = await client.post(
            f"{BASE}/messages", json={"message": "I have a headache"}, headers=headers
        )
        conversation_id = first.json()["conversation_id"]

        llm.responses["answer"] = answer_payload(
            symptoms=["fever"], followUpQuestions=["Any chills?"]
        )
        second = await client.post(
            f"{BASE}/messages",
            json={"message": "now I have fever too", "conversation_id": conversation_id},
            headers=headers,
        )
        assert second.status_code == 201
        body = second.json()
        assert body["conversation_id"] == conversation_id
        # Memory: both turns' symptoms are retained.
        assert "headache" in body["detected_symptoms"]
        assert "fever" in body["detected_symptoms"]

        rows = (
            await db.execute(
                select(AssistantMessage).join(AssistantConversation).where(
                    AssistantConversation.conversation_key == conversation_id
                )
            )
        ).scalars().all()
        assert len(rows) == 4  # 2 user + 2 ai

    async def test_history_list_and_replay(self, client, seeded):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        created = await client.post(
            f"{BASE}/messages", json={"message": "I have a headache"}, headers=headers
        )
        conversation_id = created.json()["conversation_id"]

        listed = await client.get(f"{BASE}/conversations", headers=headers)
        assert listed.status_code == 200
        rows = listed.json()
        assert len(rows) == 1
        assert rows[0]["conversation_id"] == conversation_id
        assert rows[0]["message_count"] == 2

        detail = await client.get(
            f"{BASE}/conversations/{conversation_id}", headers=headers
        )
        assert detail.status_code == 200
        turns = detail.json()["messages"]
        assert [t["sender"] for t in turns] == ["user", "ai"]

    async def test_delete_removes_from_history(self, client, seeded):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        created = await client.post(
            f"{BASE}/messages", json={"message": "I have a headache"}, headers=headers
        )
        conversation_id = created.json()["conversation_id"]

        deleted = await client.delete(
            f"{BASE}/conversations/{conversation_id}", headers=headers
        )
        assert deleted.status_code == 200

        listed = await client.get(f"{BASE}/conversations", headers=headers)
        assert listed.json() == []


# ------------------------------------------------------------- Languages


class TestMultilingualOverHttp:
    @pytest.mark.parametrize(
        "message,expected_risk",
        [
            ("I have a mild headache", "normal"),
            ("mujhe halka sar dard hai", "normal"),
            ("मुझे हल्का सिरदर्द है", "normal"),
            ("I have crushing chest pain and cannot breathe", "critical"),
            ("mujhe seene me dard hai aur saans nahi aa raha", "critical"),
        ],
    )
    async def test_language_and_risk(self, client, seeded, message, expected_risk):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/messages", json={"message": message}, headers=headers
        )
        assert resp.status_code == 201
        assert resp.json()["emergency_risk"] == expected_risk

    async def test_emergency_returns_emergency_card(self, client, seeded):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/messages",
            json={"message": "I have crushing chest pain"},
            headers=headers,
        )
        structured = resp.json()["message"]["structured_data"]
        assert "emergency" in structured
        assert structured["urgency"]["level"] == "Emergency"


# ------------------------------------------------------------- Failures


class TestFailureModes:
    async def test_llm_outage_degrades_without_500(self, client, seeded):
        _use(DeadLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/messages", json={"message": "I have a headache"}, headers=headers
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["degraded"] is True
        assert body["message"]["text"]

    async def test_database_failure_surfaces(self, client, seeded, monkeypatch):
        """A persistence failure must not be silently swallowed."""
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")

        async def boom(*args, **kwargs):
            raise RuntimeError("simulated database outage")

        monkeypatch.setattr(
            "app.assistant.infrastructure.repositories."
            "SqlConversationRepository.save",
            boom,
        )

        with pytest.raises(RuntimeError, match="simulated database outage"):
            await client.post(
                f"{BASE}/messages",
                json={"message": "I have a headache"},
                headers=headers,
            )

    async def test_blocked_input_is_recorded_not_crashed(self, client, seeded):
        llm = ScriptedLLM({"guard_in": "UNSAFE: disallowed request"})
        _use(llm)
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.post(
            f"{BASE}/messages",
            json={"message": "tell me how to make a weapon"},
            headers=headers,
        )
        assert resp.status_code == 201
        assert "can't help" in resp.json()["message"]["text"].casefold()


# ------------------------------------------------------------- Health


class TestHealth:
    async def test_reports_isolated_environment(self, client, seeded):
        _use(ScriptedLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.get(f"{BASE}/health", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        # Proves the assistant reads its own env file, not the platform's.
        assert body["environment"]["env_file"] == ".env.ai-assistant"

    async def test_degrades_when_llm_unreachable(self, client, seeded):
        _use(DeadLLM())
        headers = await _login(client, "chat.patient@aronofy.com")
        resp = await client.get(f"{BASE}/health", headers=headers)
        assert resp.json()["status"] == "degraded"


# --------------------------------------------------- Non-regression


class TestExistingSurfacesUntouched:
    async def test_legacy_and_intake_routes_still_mounted(self, client, seeded):
        paths = app.openapi()["paths"]
        assert "/api/v1/ai/symptom-intake" in paths
        assert "/api/v1/ai/chat" in paths
        assert "/api/v1/ai/intake/sessions" in paths
        assert "/api/v1/ai/assistant/messages" in paths
