from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth, patient, doctor, admin, shared, websocket, health, ai, intake,
    assistant, webhooks,
)

api_router = APIRouter()

# Register Authentication endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(patient.router, prefix="/patient", tags=["Patient Portal"])
api_router.include_router(doctor.router, prefix="/doctor", tags=["Doctor Portal"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin Portal"])
api_router.include_router(shared.router, prefix="/shared", tags=["Shared Services"])
api_router.include_router(ai.router, prefix="/ai", tags=["AI Brain Agent"])

# Stateful multi-turn intake agent. Mounted alongside the legacy single-shot
# /ai/symptom-intake endpoint above, which is left untouched.
api_router.include_router(
    intake.router, prefix="/ai/intake", tags=["AI Medical Case Intake Agent"]
)
api_router.include_router(
    intake.monitor_router, prefix="/ai/intake", tags=["AI Medical Case Intake Agent"]
)

# Conversational assistant. Runs on its own isolated .env.ai-assistant
# credentials, separate from the platform environment used everywhere else.
api_router.include_router(
    assistant.router, prefix="/ai/assistant", tags=["AI Medical Assistant"]
)
api_router.include_router(
    assistant.monitor_router, prefix="/ai/assistant", tags=["AI Medical Assistant"]
)
api_router.include_router(health.router, tags=["Health & Monitoring"])
api_router.include_router(websocket.router, tags=["WebSockets"])
# Provider callbacks. Public, signature-verified — see the module docstring.
api_router.include_router(
    webhooks.router, prefix="/webhooks", tags=["Provider Callbacks"]
)



