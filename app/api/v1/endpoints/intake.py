"""
AI Medical Case Intake Agent endpoints.

Thin async controllers. Each one validates input via Pydantic, delegates to a
single use case, and maps the result to a response model — no database access,
no AI logic, no business rules. Domain errors propagate to the global handlers
registered in `app/middleware/exceptions.py`, which already map them to the
correct status codes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, status

from app.api.deps import RoleChecker, get_current_active_user
from app.intake.application.dto import StartIntakeCommand, SubmitAnswerCommand
from app.intake.application.ports import LLMPort
from app.intake.application.use_cases import (
    GetSessionUseCase,
    SelectDoctorUseCase,
    StartIntakeUseCase,
    SubmitAnswerUseCase,
)
from app.intake.dependencies import (
    get_llm,
    get_patient_user_id,
    get_select_doctor_use_case,
    get_session_use_case,
    get_start_intake_use_case,
    get_submit_answer_use_case,
)
from app.intake.domain.policies import MAX_FOLLOWUP_ROUNDS, MIN_OVERALL_CONFIDENCE
from app.schemas.intake_api import (
    IntakeHealthResponse,
    IntakeSessionResponse,
    RoutingResponse,
    SelectDoctorRequest,
    StartIntakeRequest,
    SubmitAnswerRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(RoleChecker(["patient"]))])
"""Patient-facing intake routes. Only patients may open or advance a session."""

monitor_router = APIRouter()
"""Health route, available to any authenticated user for dashboards/probes."""


@router.post(
    "/sessions",
    response_model=IntakeSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a medical intake conversation",
)
async def start_intake(
    request: StartIntakeRequest,
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: StartIntakeUseCase = Depends(get_start_intake_use_case),
) -> Any:
    """
    Open an intake session from the patient's first symptom description.

    Accepts English, Hindi, Hinglish or mixed input. The response either carries
    a follow-up question (`status='collecting'`), a completed case with
    specialist options (`status='awaiting_doctor_selection'`), or emergency
    guidance (`status='emergency_escalated'`).
    """
    view = await use_case.execute(
        StartIntakeCommand(
            patient_user_id=patient_user_id,
            text=request.symptoms,
            age=request.age,
            gender=request.gender,
        )
    )
    return IntakeSessionResponse.from_view(view)


@router.post(
    "/sessions/{session_id}/turns",
    response_model=IntakeSessionResponse,
    summary="Answer the agent's follow-up question",
)
async def submit_answer(
    session_id: str,
    request: SubmitAnswerRequest,
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: SubmitAnswerUseCase = Depends(get_submit_answer_use_case),
) -> Any:
    """
    Record the patient's reply and re-run the workflow.

    The agent asks at most `MAX_FOLLOWUP_ROUNDS` questions; after that it
    generates the case with any unresolved fields marked `Unknown` rather than
    continuing to ask.
    """
    view = await use_case.execute(
        SubmitAnswerCommand(
            patient_user_id=patient_user_id,
            session_id=session_id,
            text=request.answer,
        )
    )
    return IntakeSessionResponse.from_view(view)


@router.get(
    "/sessions/{session_id}",
    response_model=IntakeSessionResponse,
    summary="Read the current state of an intake session",
)
async def get_session(
    session_id: str,
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: GetSessionUseCase = Depends(get_session_use_case),
) -> Any:
    """Fetch a session the caller owns, including extracted entities and case."""
    view = await use_case.execute(session_id, patient_user_id)
    return IntakeSessionResponse.from_view(view)


@router.post(
    "/sessions/{session_id}/select-doctor",
    response_model=RoutingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Select a specialist and route the case",
)
async def select_doctor(
    session_id: str,
    request: SelectDoctorRequest,
    patient_user_id: str = Depends(get_patient_user_id),
    use_case: SelectDoctorUseCase = Depends(get_select_doctor_use_case),
) -> Any:
    """
    Finalise the intake.

    Persists the structured case into the main `cases` table, writes the
    extraction audit trail, and assigns the case to the chosen doctor. Valid
    only while the session is `awaiting_doctor_selection`.
    """
    result = await use_case.execute(
        session_id=session_id,
        patient_user_id=patient_user_id,
        doctor_id=str(request.doctor_id),
    )
    return RoutingResponse(
        session_id=result.session_id,
        case_id=result.case_id,
        doctor_id=result.doctor_id,
        doctor_name=result.doctor_name,
        specialty=result.specialty,
        urgency=result.urgency,
    )


@monitor_router.get(
    "/health",
    response_model=IntakeHealthResponse,
    summary="Intake agent dependency health",
)
async def intake_health(
    llm: LLMPort = Depends(get_llm),
    _=Depends(get_current_active_user),
) -> Any:
    """Report LLM reachability and the active safety thresholds."""
    llm_health = await llm.health()
    llm_status = str(llm_health.get("status", "unhealthy"))

    return IntakeHealthResponse(
        status="healthy" if llm_status == "healthy" else "degraded",
        llm=llm_health,
        graph_nodes=9,
        max_followup_rounds=MAX_FOLLOWUP_ROUNDS,
        min_overall_confidence=MIN_OVERALL_CONFIDENCE,
    )
