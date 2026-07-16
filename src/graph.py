"""LangGraph assembly for the triage copilot."""
from langgraph.graph import StateGraph, START, END

from src.state import TriageState
from src.nodes.intake import intake_node
from src.nodes.classify import make_classify_node
from src.nodes.route import route_node


def build_graph(prompt_version="v1"):
    """Compile the Phase 1 pipeline: intake -> classify -> route.

    No checkpointer: Phase 2 adds one when interrupt() needs persisted state.
    """
    b = StateGraph(TriageState)
    b.add_node("intake", intake_node)
    b.add_node("classify", make_classify_node(prompt_version))
    b.add_node("route", route_node)
    b.add_edge(START, "intake")
    b.add_edge("intake", "classify")
    b.add_edge("classify", "route")
    b.add_edge("route", END)
    return b.compile()
