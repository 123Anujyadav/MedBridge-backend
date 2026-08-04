"""
Prescription document, safety review and printable PDF.

Mounted under `/api/v1/prescriptions`. Every route is additive; nothing here
replaces an existing prescription endpoint on the patient or doctor routers.

Access rule, applied identically on all three routes: a prescription may be
read by the patient it was written for, the doctor who wrote it, or an admin.
Anyone else gets 403 — including other clinicians, who have no standing
relationship to a prescription they did not issue.
"""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_active_user, get_db
from app.core.exceptions import AuthorizationException, EntityNotFoundException
from app.models.prescription import Prescription
from app.models.user import User
from app.schemas.rx_safety_api import (
    MedicationLineResponse,
    PrescriberCardResponse,
    PrescriptionDocumentResponse,
    PrescriptionVerificationResponse,
)
from app.services.prescription_safety import prescription_safety_service

logger = logging.getLogger(__name__)

router = APIRouter()


async def _authorised_prescription(
    db: AsyncSession, prescription_id: uuid.UUID, user: User
) -> Prescription:
    result = await db.execute(
        select(Prescription)
        .where(Prescription.id == prescription_id)
        .options(selectinload(Prescription.medications))
    )
    rx = result.scalar_one_or_none()
    if not rx:
        raise EntityNotFoundException("Prescription", str(prescription_id))

    role = getattr(user, "role", None)
    if role == "admin":
        return rx
    if role == "patient" and rx.patient_id == user.id:
        return rx
    if role == "doctor" and rx.doctor_id == user.id:
        return rx

    # Logged because an authenticated user reaching for someone else's
    # prescription is worth seeing, whether it is a bug or a probe.
    logger.warning(
        "[RX_ACCESS_DENIED] user=%s role=%s rx=%s", user.id, role, prescription_id
    )
    raise AuthorizationException("You are not authorized to view this prescription.")


def _prescriber_card(rx: Prescription) -> PrescriberCardResponse:
    return PrescriberCardResponse(
        doctor_id=rx.doctor_id,
        doctor_name=rx.doctor_name,
        specialty=rx.doctor_specialty,
        qualification=rx.doctor_qualification,
        hospital=rx.doctor_hospital,
        registration_number=rx.doctor_registration_number,
        experience_years=rx.doctor_experience_years,
        avatar_url=rx.doctor_avatar_url,
        consultation_date=rx.consultation_date,
        signed_at=rx.signed_at,
        signature_url=rx.doctor_signature_url,
        # The consultation is complete once a prescription exists for it; the
        # case moves to `prescribed` in the same transaction that writes this row.
        consultation_completed=rx.status in ("active", "verified", "completed"),
        prescription_signed=rx.signed_at is not None,
    )


@router.get("/{prescription_id}", response_model=PrescriptionDocumentResponse)
async def get_prescription_document(
    prescription_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    The full prescription: prescriber card, medication lines and the most
    recent safety review, in one round trip.

    The review is read, never run — issuing a request for a document should not
    block on two external drug APIs. Use POST /verify to run one.
    """
    rx = await _authorised_prescription(db, prescription_id, current_user)
    verification = await prescription_safety_service.latest_verification(
        db, prescription_id
    )

    return PrescriptionDocumentResponse(
        id=rx.id,
        status=rx.status,
        diagnosis=rx.diagnosis,
        notes=rx.notes or "",
        follow_up_date=rx.follow_up_date,
        created_at=rx.created_at,
        prescriber=_prescriber_card(rx),
        medications=[MedicationLineResponse.model_validate(m) for m in rx.medications],
        verification=(
            PrescriptionVerificationResponse.model_validate(verification)
            if verification
            else None
        ),
        pdf_url=rx.pdf_url,
        prescription_image_url=rx.prescription_image_url,
    )


@router.post(
    "/{prescription_id}/verify",
    response_model=PrescriptionVerificationResponse | None,
    status_code=status.HTTP_200_OK,
)
async def verify_prescription(
    prescription_id: uuid.UUID,
    refresh: bool = Query(
        False,
        description="Force a new review instead of returning the stored one.",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Run (or fetch) the AI safety review.

    Advisory only: this never alters the prescription. Returns null when the
    feature is switched off via RXSAFETY_ENABLED, which callers should render
    as "not checked" rather than as "no problems found".
    """
    await _authorised_prescription(db, prescription_id, current_user)

    if refresh:
        verification = await prescription_safety_service.verify(db, prescription_id)
    else:
        verification = await prescription_safety_service.get_or_run_verification(
            db, prescription_id
        )

    if verification is None:
        return None

    await db.commit()
    await db.refresh(verification, ["findings"])
    return PrescriptionVerificationResponse.model_validate(verification)


@router.get(
    "/{prescription_id}/pdf",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}, "description": "Prescription PDF"}},
)
async def download_prescription_pdf(
    prescription_id: uuid.UUID,
    disposition: str = Query("attachment", pattern="^(inline|attachment)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    The printable prescription.

    `inline` serves it for preview; `attachment` downloads it. Rendered on
    demand so the document always reflects the current safety review.
    """
    rx = await _authorised_prescription(db, prescription_id, current_user)

    result = await prescription_safety_service.generate_pdf(db, prescription_id)
    await db.commit()

    file_path = result["file_path"]
    if not os.path.exists(file_path):
        raise EntityNotFoundException("Prescription PDF", str(prescription_id))

    filename = f"prescription-{str(rx.id)[:8]}.pdf"
    if disposition == "inline":
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "private, no-cache",
            },
        )

    return FileResponse(
        path=file_path, media_type="application/pdf", filename=filename
    )
