import pytest
import uuid
from app.models.user import User
from app.core.security import get_password_hash
from conftest import login_payload

class MockWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent_messages = []
        self.received_messages = []
        self.closed = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self.received_messages:
            from fastapi.websockets import WebSocketDisconnect
            raise WebSocketDisconnect()
        return self.received_messages.pop(0)

    async def send_json(self, data: dict):
        self.sent_messages.append(data)

    async def close(self, code: int = 1000):
        self.closed = True
        self.close_code = code

@pytest.fixture
async def setup_websocket_data(db):
    # Ensure a patient and a doctor exist
    from sqlalchemy import text
    
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    await db.execute(text("DELETE FROM appointments;"))
    await db.execute(text("DELETE FROM users;"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    # Create Patient User
    patient = User(
        email="patient.ws@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="patient",
        is_verified=True
    )
    db.add(patient)

    # Create Doctor User
    doctor = User(
        email="doctor.ws@aronofy.com",
        hashed_password=get_password_hash("password123"),
        role="doctor",
        is_verified=True
    )
    db.add(doctor)

    await db.flush()

    # The live socket carries clinical broadcasts, so it is gated on the same
    # administrator approval as the doctor APIs — this fixture is an approved
    # clinician.
    from app.models.doctor import Doctor

    db.add(Doctor(
        id=doctor.id, first_name="Ws", last_name="Doctor", phone="+9133",
        specialty="Cardiology", license_number="LIC-WS-1",
        verification_status="verified",
    ))
    await db.commit()
    return {
        "patient_id": patient.id,
        "doctor_id": doctor.id
    }

@pytest.mark.asyncio
async def test_websocket_auth_and_echo(client, setup_websocket_data, db):
    # 1. Login to get Patient token
    login_resp = await client.post(
        "/api/v1/auth/login",
        json=await login_payload("patient.ws@aronofy.com", "password123")
    )
    assert login_resp.status_code == 200
    patient_token = login_resp.json()["access_token"]

    # 2. Login to get Doctor token
    login_resp = await client.post(
        "/api/v1/auth/login",
        json=await login_payload("doctor.ws@aronofy.com", "password123")
    )
    assert login_resp.status_code == 200
    doctor_token = login_resp.json()["access_token"]

    from app.api.v1.endpoints.websocket import websocket_endpoint

    # 3. Test patient connection with valid token
    ws_patient = MockWebSocket()
    ws_patient.received_messages.append("Hello WebSocket")
    await websocket_endpoint(ws_patient, token=patient_token, db=db)
    assert ws_patient.accepted is True
    assert len(ws_patient.sent_messages) == 1
    assert ws_patient.sent_messages[0]["status"] == "acknowledged"
    assert ws_patient.sent_messages[0]["received"] == "Hello WebSocket"

    # 4. Test doctor connection with trigger emergency
    ws_doctor = MockWebSocket()
    ws_doctor.received_messages.append("trigger_emergency")
    await websocket_endpoint(ws_doctor, token=doctor_token, db=db)
    assert ws_doctor.accepted is True
    # Verify doctor received both the emergency broadcast and the acknowledgment
    assert len(ws_doctor.sent_messages) == 2
    assert any(m.get("type") == "emergency" for m in ws_doctor.sent_messages)
    assert any(m.get("status") == "acknowledged" for m in ws_doctor.sent_messages)

    # 5. Test invalid token connection rejection
    ws_fail = MockWebSocket()
    await websocket_endpoint(ws_fail, token="invalid_jwt_claims_token", db=db)
    assert ws_fail.accepted is False
    assert ws_fail.closed is True

