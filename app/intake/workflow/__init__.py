"""
LangGraph orchestration for the Medical Case Intake Agent.

The only package that knows an LLM exists. Each workflow stage is an
independent node in `nodes.py`; `graph.py` wires them into a `StateGraph` and
exposes it behind `IntakeWorkflowPort` so the application layer stays free of
LangGraph.
"""

from app.intake.workflow.graph import LangGraphIntakeWorkflow, build_intake_graph
from app.intake.workflow.state import IntakeState

__all__ = ["IntakeState", "LangGraphIntakeWorkflow", "build_intake_graph"]
