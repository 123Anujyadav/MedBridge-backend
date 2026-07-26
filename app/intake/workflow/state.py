"""
Graph state for the intake workflow.

The `IntakeSession` aggregate is carried through the graph directly rather than
being flattened into scalars: nodes mutate the domain object and LangGraph
threads it between them, which keeps a single source of truth for the
conversation. Everything else here is transient per-run bookkeeping.
"""

from __future__ import annotations

from typing import TypedDict

# Imported at runtime, not under TYPE_CHECKING: LangGraph evaluates TypedDict
# annotations when it builds the state schema, so a forward reference here fails
# at graph construction. Safe from cycles — `application` never imports
# `workflow`.
from app.intake.application.ports import DoctorDirectoryPort
from app.intake.domain.entities import ExtractedEntity, IntakeSession


class IntakeState(TypedDict, total=False):
    """State passed between intake graph nodes."""

    session: IntakeSession
    """The aggregate under construction. Every node reads and may mutate it."""

    doctors: DoctorDirectoryPort
    """
    Request-scoped doctor directory.

    Travels in state rather than being captured at node construction, so the
    compiled graph can be a process-level singleton instead of being rebuilt for
    every request around a short-lived database session.
    """

    latest_text: str
    """Sanitised text of the newest patient turn, for this pass only."""

    llm_degraded: bool
    """
    Set when a model call returned nothing usable.

    Downstream nodes check this to choose deterministic fallbacks instead of
    silently emitting empty clinical data.
    """

    rejected_entities: list[ExtractedEntity]
    """Extractions dropped for failing evidence grounding. Surfaced for audit."""

    notices: list[str]
    """Operator-facing notes about degradation or safety action taken."""
