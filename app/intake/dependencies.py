"""
Composition root for the Medical Case Intake Agent.

The single place where abstract ports are bound to concrete adapters. Keeping
the wiring here means controllers declare only the use case they need, and tests
can swap any collaborator through FastAPI's `dependency_overrides` without
touching application code.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user, get_db
from app.core.redis import get_redis
from app.intake.application.ports import (
    DoctorDirectoryPort,
    IntakeWorkflowPort,
    LLMPort,
    SessionStorePort,
)
from app.intake.application.use_cases import (
    GetSessionUseCase,
    SelectDoctorUseCase,
    StartIntakeUseCase,
    SubmitAnswerUseCase,
)
from app.intake.infrastructure.doctor_directory import SqlDoctorDirectory
from app.intake.infrastructure.llm_groq import GroqJSONAdapter
from app.intake.infrastructure.repositories import SqlCaseRepository, SqlIntakeAudit
from app.intake.infrastructure.session_store import RedisSessionStore
from app.intake.workflow.graph import LangGraphIntakeWorkflow
from app.models.user import User

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> LLMPort:
    """
    Process-wide LLM adapter.

    Cached so the underlying async HTTP client and its connection pool are
    reused across requests instead of being rebuilt per call.
    """
    logger.info("[INTAKE_DI] initialising Groq adapter")
    return GroqJSONAdapter()


def get_doctor_directory(db: AsyncSession = Depends(get_db)) -> DoctorDirectoryPort:
    """Request-scoped: bound to the caller's database session."""
    return SqlDoctorDirectory(db)


def get_session_store(redis: Any = Depends(get_redis)) -> SessionStorePort:
    return RedisSessionStore(redis)


def get_workflow(
    llm: LLMPort = Depends(get_llm),
    doctors: DoctorDirectoryPort = Depends(get_doctor_directory),
) -> IntakeWorkflowPort:
    """
    Build the intake workflow for this request.

    Constructed per-request because the doctor directory carries the
    request-scoped DB session. Graph compilation is pure in-memory DAG assembly
    with no I/O or model loading, so this is cheap enough to do per call and
    avoids the lifetime hazard of caching a graph that closes over a session.
    """
    return LangGraphIntakeWorkflow(llm=llm, doctors=doctors)


def get_start_intake_use_case(
    sessions: SessionStorePort = Depends(get_session_store),
    workflow: IntakeWorkflowPort = Depends(get_workflow),
) -> StartIntakeUseCase:
    return StartIntakeUseCase(sessions=sessions, workflow=workflow)


def get_submit_answer_use_case(
    sessions: SessionStorePort = Depends(get_session_store),
    workflow: IntakeWorkflowPort = Depends(get_workflow),
) -> SubmitAnswerUseCase:
    return SubmitAnswerUseCase(sessions=sessions, workflow=workflow)


def get_session_use_case(
    sessions: SessionStorePort = Depends(get_session_store),
) -> GetSessionUseCase:
    return GetSessionUseCase(sessions=sessions)


def get_select_doctor_use_case(
    sessions: SessionStorePort = Depends(get_session_store),
    doctors: DoctorDirectoryPort = Depends(get_doctor_directory),
    db: AsyncSession = Depends(get_db),
) -> SelectDoctorUseCase:
    return SelectDoctorUseCase(
        sessions=sessions,
        doctors=doctors,
        cases=SqlCaseRepository(db),
        audit=SqlIntakeAudit(db),
    )


def get_patient_user_id(
    current_user: User = Depends(get_current_active_user),
) -> str:
    """The authenticated caller's id, used as the session owner."""
    return str(current_user.id)
