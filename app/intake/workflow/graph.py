"""
LangGraph assembly for the intake agent.

    START
      -> receive_input        (sanitise + deterministic red-flag scan)
      -> detect_language
      -> detect_intent
      -> [emergency?] ------------------> escalate_emergency -> END
      -> extract_entities     (+ evidence grounding)
      -> evaluate_confidence
      -> [ready?] --no--> generate_followup ------------------> END
                  --yes-> generate_case -> recommend_specialist -> END

Both branch points are pure functions over domain state, so the routing logic is
unit-testable without running the graph or touching a model.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.intake.application.dto import WorkflowResult
from app.intake.application.ports import DoctorDirectoryPort, LLMPort
from app.intake.domain.entities import IntakeSession
from app.intake.domain.enums import IntentType, SessionStatus
from app.intake.domain.policies import evaluate_readiness
from app.intake.workflow.nodes import IntakeNodes
from app.intake.workflow.state import IntakeState

logger = logging.getLogger(__name__)

_EMERGENCY_BRANCH = "emergency"
_CONTINUE_BRANCH = "continue"
_READY_BRANCH = "ready"
_NEEDS_INFO_BRANCH = "needs_info"


def _route_after_intent(state: IntakeState) -> str:
    """Divert to emergency handling when red flags or emergency intent are present."""
    session = state["session"]
    if session.red_flags or session.intent is IntentType.EMERGENCY:
        return _EMERGENCY_BRANCH
    return _CONTINUE_BRANCH


def _route_after_confidence(state: IntakeState) -> str:
    """Generate the case only when mandatory fields clear the confidence floor."""
    verdict = evaluate_readiness(state["session"])
    return _READY_BRANCH if verdict.is_ready else _NEEDS_INFO_BRANCH


@lru_cache(maxsize=8)
def build_intake_graph(llm: LLMPort):
    """
    Construct and compile the intake StateGraph.

    Cached per LLM instance: compilation costs ~15ms, which is material on every
    request. The graph closes over nothing request-scoped — the doctor directory
    arrives through `IntakeState` — so a single compiled instance is safe to
    share across concurrent requests.
    """
    nodes = IntakeNodes(llm=llm)
    graph: StateGraph = StateGraph(IntakeState)

    graph.add_node("receive_input", nodes.receive_input)
    graph.add_node("detect_language", nodes.detect_language)
    graph.add_node("detect_intent", nodes.detect_intent)
    graph.add_node("extract_entities", nodes.extract_entities)
    graph.add_node("evaluate_confidence", nodes.evaluate_confidence)
    graph.add_node("generate_followup", nodes.generate_followup)
    graph.add_node("generate_case", nodes.generate_case)
    graph.add_node("recommend_specialist", nodes.recommend_specialist)
    graph.add_node("escalate_emergency", nodes.escalate_emergency)

    graph.add_edge(START, "receive_input")
    graph.add_edge("receive_input", "detect_language")
    graph.add_edge("detect_language", "detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        _route_after_intent,
        {
            _EMERGENCY_BRANCH: "escalate_emergency",
            _CONTINUE_BRANCH: "extract_entities",
        },
    )

    graph.add_edge("extract_entities", "evaluate_confidence")

    graph.add_conditional_edges(
        "evaluate_confidence",
        _route_after_confidence,
        {
            _READY_BRANCH: "generate_case",
            _NEEDS_INFO_BRANCH: "generate_followup",
        },
    )

    graph.add_edge("generate_case", "recommend_specialist")
    graph.add_edge("recommend_specialist", END)
    graph.add_edge("generate_followup", END)
    graph.add_edge("escalate_emergency", END)

    return graph.compile()


class LangGraphIntakeWorkflow:
    """
    `IntakeWorkflowPort` implementation backed by the compiled graph.

    Cheap to construct: the compiled graph is shared via `build_intake_graph`'s
    cache, and this object only binds it to the request's doctor directory.
    """

    def __init__(self, *, llm: LLMPort, doctors: DoctorDirectoryPort) -> None:
        self._graph = build_intake_graph(llm)
        self._doctors = doctors

    async def run_detailed(self, session: IntakeSession) -> WorkflowResult:
        """
        Run the graph and return the session plus this pass's diagnostics.

        A graph failure is contained: the session is left in a coherent
        `COLLECTING` state with a notice, so the patient can retry rather than
        losing the conversation to a 500.
        """
        initial: IntakeState = {
            "session": session,
            "doctors": self._doctors,
            "latest_text": "",
            "llm_degraded": False,
            "rejected_entities": [],
            "notices": [],
        }

        try:
            final: IntakeState = await self._graph.ainvoke(initial)
        except Exception:
            logger.exception(
                "[INTAKE_GRAPH_FAILED] session=%s — returning session unchanged",
                session.session_id,
            )
            if not session.status.is_terminal:
                session.status = SessionStatus.COLLECTING
                session.pending_question = (
                    session.pending_question
                    or "Sorry, something went wrong on our side. "
                    "Could you describe your symptoms once more?"
                )
            session.touch()
            return WorkflowResult(
                session=session,
                notices=["The intake service hit an internal error. Please try again."],
                rejected_count=0,
                degraded=True,
            )

        return WorkflowResult(
            session=final["session"],
            notices=list(final.get("notices") or []),
            rejected_count=len(final.get("rejected_entities") or []),
            degraded=bool(final.get("llm_degraded")),
        )
