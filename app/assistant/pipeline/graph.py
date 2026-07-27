"""
LangGraph assembly for the AI Medical Assistant.

    START
      -> detect_language     (script heuristics + deterministic red-flag scan)
      -> detect_intent
      -> [emergency?] ----------------------> emergency -> END
      -> extract_entities    (+ evidence grounding)
      -> retrieve_knowledge  (RAG, optional)
      -> generate_answer     -> END

Mirrors the routing idea from the source project's `agent_decision.py`
(analyse -> route -> agent -> guardrails), rebuilt on LangGraph 1.x. Output
guardrails run in the use case rather than as a node, so they also cover the
deterministic emergency and fallback replies.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.assistant.application.ports import (
    AssistantLLMPort,
    KnowledgeRetrievalPort,
)
from app.assistant.domain.entities import AssistantAnswer, Conversation
from app.assistant.domain.enums import IntentType
from app.assistant.pipeline.nodes import AssistantNodes
from app.assistant.pipeline.state import AssistantState

logger = logging.getLogger(__name__)

_EMERGENCY = "emergency"
_CONTINUE = "continue"


def _route_after_intent(state: AssistantState) -> str:
    """Divert to emergency handling on red flags or emergency intent."""
    if state.get("red_flags") or state.get("intent") is IntentType.EMERGENCY:
        return _EMERGENCY
    return _CONTINUE


@lru_cache(maxsize=4)
def build_assistant_graph(llm: AssistantLLMPort):
    """
    Compile the assistant pipeline.

    Cached per LLM instance: compilation is measurable per-request overhead and
    the graph closes over nothing request-scoped — the retriever arrives through
    `AssistantState`.
    """
    nodes = AssistantNodes(llm=llm)
    graph: StateGraph = StateGraph(AssistantState)

    graph.add_node("detect_language", nodes.detect_language_node)
    graph.add_node("detect_intent", nodes.detect_intent_node)
    graph.add_node("extract_entities", nodes.extract_entities_node)
    graph.add_node("retrieve_knowledge", nodes.retrieve_knowledge_node)
    graph.add_node("generate_answer", nodes.generate_answer_node)
    graph.add_node("emergency", nodes.emergency_node)

    graph.add_edge(START, "detect_language")
    graph.add_edge("detect_language", "detect_intent")
    graph.add_conditional_edges(
        "detect_intent",
        _route_after_intent,
        {_EMERGENCY: "emergency", _CONTINUE: "extract_entities"},
    )
    graph.add_edge("extract_entities", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("emergency", END)

    return graph.compile()


class LangGraphAssistantPipeline:
    """`AssistantPipelinePort` implementation backed by the compiled graph."""

    def __init__(
        self, *, llm: AssistantLLMPort, retriever: KnowledgeRetrievalPort
    ) -> None:
        self._graph = build_assistant_graph(llm)
        self._retriever = retriever

    async def run(
        self, conversation: Conversation, user_text: str
    ) -> AssistantAnswer:
        """
        Produce one structured answer.

        Never raises: a pipeline failure returns an honest degraded answer so
        the patient keeps their conversation instead of receiving a 500.
        """
        initial: AssistantState = {
            "conversation": conversation,
            "user_text": user_text,
            "retriever": self._retriever,
            "entities": [],
            "snippets": [],
            "red_flags": [],
            "degraded": False,
            "soft_failures": [],
            "rejected_entities": 0,
        }

        try:
            final: AssistantState = await self._graph.ainvoke(initial)
        except Exception as exc:
            # Full traceback plus the exception type: an unhandled pipeline error
            # is a code defect and must never be reduced to a one-line message.
            logger.exception(
                "[ASSISTANT_PIPELINE_FAILED] conversation=%s error=%s: %s",
                conversation.conversation_id,
                type(exc).__name__,
                exc,
            )
            return AssistantAnswer(
                reply_text=(
                    "Sorry — something went wrong while analysing your message. "
                    "Please try again."
                ),
                summary="The assistant hit an internal error.",
                language=conversation.language,
                confidence=0.0,
                conversation_title=conversation.title,
                degraded=True,
            )

        answer = final.get("answer")
        if answer is None:
            logger.error(
                "[ASSISTANT_NO_ANSWER] conversation=%s", conversation.conversation_id
            )
            return AssistantAnswer(
                reply_text="Sorry — I could not produce a response. Please try again.",
                language=conversation.language,
                conversation_title=conversation.title,
                degraded=True,
            )
        return answer
