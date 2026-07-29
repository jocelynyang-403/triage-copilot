"""LangGraph assembly for the triage copilot."""
from langgraph.graph import StateGraph, START, END

from src.state import TriageState
from src.nodes.intake import intake_node
from src.nodes.classify import make_classify_node
from src.nodes.knowledge import make_knowledge_node
from src.nodes.route import make_route_node


def build_graph(prompt_version="v1", routing_version="v1"):
    """Compile the Phase 2 pipeline: intake -> classify -> knowledge -> route.

    prompt_version selects the classifier prompt; routing_version selects the
    auto_reply-vs-escalate strategy (v1 threshold, v2 LLM grounding). No checkpointer
    yet - added when interrupt()/human review lands in a later PR.
    """
    b = StateGraph(TriageState)
    b.add_node("intake", intake_node)
    b.add_node("classify", make_classify_node(prompt_version))
    b.add_node("knowledge", make_knowledge_node())
    b.add_node("route", make_route_node(routing_version))
    b.add_edge(START, "intake")
    b.add_edge("intake", "classify")
    b.add_edge("classify", "knowledge")
    b.add_edge("knowledge", "route")
    b.add_edge("route", END)
    return b.compile()
