"""
Infrastructure adapters for the Medical Case Intake Agent.

Concrete implementations of the application ports: Groq for the LLM, Redis for
session state, SQLAlchemy for clinical persistence and the audit trail. This is
the only layer permitted to import a driver.
"""

from app.intake.infrastructure.doctor_directory import SqlDoctorDirectory
from app.intake.infrastructure.llm_groq import GroqJSONAdapter
from app.intake.infrastructure.repositories import SqlCaseRepository, SqlIntakeAudit
from app.intake.infrastructure.session_store import RedisSessionStore

__all__ = [
    "GroqJSONAdapter",
    "RedisSessionStore",
    "SqlCaseRepository",
    "SqlDoctorDirectory",
    "SqlIntakeAudit",
]
