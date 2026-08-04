from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth, patient, doctor, admin, shared, websocket, health, ai, intake,
    assistant, webhooks, prescriptions, pharmacy, pharmacy_admin,
    pharmacy_portal, delivery,
)

api_router = APIRouter()

# Register Authentication endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(patient.router, prefix="/patient", tags=["Patient Portal"])
api_router.include_router(doctor.router, prefix="/doctor", tags=["Doctor Portal"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin Portal"])
api_router.include_router(shared.router, prefix="/shared", tags=["Shared Services"])
# Prescription document, AI safety review and printable PDF. Shared by the
# patient and doctor portals, so it is mounted at the top level rather than
# duplicated under both.
api_router.include_router(
    prescriptions.router, prefix="/prescriptions", tags=["Prescriptions"]
)
# Pharmacy discovery, medicine availability, ordering and delivery tracking.
api_router.include_router(
    pharmacy.router, prefix="/pharmacy", tags=["Pharmacy & Medicine Orders"]
)
# Pharmacy administration. Mounted under /admin and gated to the admin role on
# the router itself, so no patient or doctor token can reach any route in it.
api_router.include_router(
    pharmacy_admin.router,
    prefix="/admin/pharmacies",
    tags=["Admin — Pharmacy Network"],
)
# Pharmacy Owner Portal. Every route is gated by `require_verified_pharmacy`,
# which checks role, store link and live approval on each request.
api_router.include_router(
    pharmacy_portal.router,
    prefix="/pharmacy-portal",
    tags=["Pharmacy Owner Portal"],
)
# Delivery & Logistics. The rider router gates each route on an approved
# profile; the admin sub-router is role-gated on the router itself.
api_router.include_router(
    delivery.router, prefix="/delivery", tags=["Delivery & Logistics"]
)
api_router.include_router(
    delivery.admin_router, prefix="/delivery/admin", tags=["Delivery — Fleet Admin"]
)
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



