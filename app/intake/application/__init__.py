"""
Application layer for the Medical Case Intake Agent.

Defines the ports (abstract collaborators) the use cases depend on, and the use
cases themselves. Depends on `domain`; knows nothing about FastAPI, SQLAlchemy,
Redis or LangGraph.
"""

from app.intake.application.dto import (
    DoctorRef,
    RoutingResult,
    SessionView,
    StartIntakeCommand,
    SubmitAnswerCommand,
    WorkflowResult,
)
from app.intake.application.ports import (
    CaseRepositoryPort,
    DoctorDirectoryPort,
    IntakeAuditPort,
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

__all__ = [
    "CaseRepositoryPort",
    "DoctorDirectoryPort",
    "DoctorRef",
    "GetSessionUseCase",
    "IntakeAuditPort",
    "IntakeWorkflowPort",
    "LLMPort",
    "RoutingResult",
    "SelectDoctorUseCase",
    "SessionStorePort",
    "SessionView",
    "StartIntakeCommand",
    "StartIntakeUseCase",
    "SubmitAnswerCommand",
    "SubmitAnswerUseCase",
    "WorkflowResult",
]
