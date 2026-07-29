"""Route node: decide auto_reply vs escalate (+ destination).

Two routing strategies, selected by routing_version so Phase 4 can compare them:
  v1 - pure kb_confidence threshold.
  v2 - LLM grounding check: do the retrieved KB excerpts actually ANSWER the ticket?
Both share an allowlist priority guard: only P2/P3 tickets are ever eligible for
auto_reply (every auto_reply golden row is P2/P3, and P0 recall is too low to trust a
denylist). The grounding check exists because probe_collision.py showed a single
confidence threshold cannot separate answerable from actionable tickets (best cutoff
misses 3/11 contested dev rows; topical similarity != answerability).
"""
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import CLASSIFY_MODEL  # Haiku: cheap, supports temperature=0

_CATEGORY_TO_DESTINATION = {
    "billing": "Billing Ops",
    "bug": "Engineering",
    "access_request": "IT / Access",
    "sales_lead": "Sales",
    "other": "Support",
}
_FALLBACK_DESTINATION = "Human Review"

# Dev-tuned: best single threshold from probe_collision.py. Ceiling 8/11 on the
# contested P2/P3 dev rows - that ceiling is exactly why v2 exists. Tune only on dev.
THRESHOLD_V1 = 0.58

# Allowlist, not denylist: P0 recall is only ~5/9, so a `priority == "P0"` denylist
# would leak un-caught P0s into auto_reply. Every auto_reply golden row is P2/P3.
_AUTO_ELIGIBLE = ("P2", "P3")

GROUNDING_SYSTEM = (
    "You are a strict grounding checker for an ops-support triage system. "
    "You are given one customer ticket and excerpts retrieved from an internal knowledge base. "
    "Decide whether the excerpts contain the SPECIFIC information needed to FULLY resolve the "
    "ticket with an automated reply that requires no human action, approval, provisioning, or "
    "account-specific lookup. "
    "Set answerable=true ONLY if a complete, correct reply can be written using ONLY these "
    "excerpts. If the ticket needs a human decision, an account-specific action, or information "
    "not present in the excerpts, set answerable=false. "
    "Topical similarity is NOT sufficient: the excerpts must actually answer THIS request."
)


class Grounding(BaseModel):
    """Structured grounding verdict - unparseable output is impossible by construction."""
    answerable: bool = Field(description="True only if the excerpts fully answer the ticket.")
    reasoning: str = Field(description="One sentence.")


def _destination_for(category):
    return _CATEGORY_TO_DESTINATION.get(category, _FALLBACK_DESTINATION)


def guard_allows(priority):
    """Allowlist: only P2/P3 are ever eligible for auto_reply."""
    return priority in _AUTO_ELIGIBLE


def route_v1(priority, kb_confidence):
    if not guard_allows(priority):
        return "escalate"
    return "auto_reply" if (kb_confidence or 0.0) >= THRESHOLD_V1 else "escalate"


def make_grounding_checker():
    """Load the Haiku grounding LLM once; return a fn(ticket_text, kb_chunks) -> Grounding."""
    llm = ChatAnthropic(
        model=CLASSIFY_MODEL,
        temperature=0,
        max_tokens=512,
    ).with_structured_output(Grounding)

    def check(ticket_text, kb_chunks):
        excerpts = "\n\n---\n\n".join(kb_chunks)
        human = "TICKET:\n{}\n\nKNOWLEDGE BASE EXCERPTS:\n{}".format(ticket_text, excerpts)
        return llm.invoke([("system", GROUNDING_SYSTEM), ("human", human)])

    return check


def route_v2(priority, ticket_text, kb_chunks, grounding_checker):
    if not guard_allows(priority):
        return "escalate"
    if not kb_chunks:
        return "escalate"
    return "auto_reply" if grounding_checker(ticket_text, kb_chunks).answerable else "escalate"


def make_route_node(routing_version="v1"):
    """Return a route node closure for the chosen strategy (mirrors make_classify_node)."""
    if routing_version not in ("v1", "v2"):
        raise ValueError("Invalid routing_version: {!r} (expected 'v1' or 'v2')".format(routing_version))

    checker = make_grounding_checker() if routing_version == "v2" else None

    def route_node(state):
        category = state.get("category")
        priority = state.get("priority")
        destination = _destination_for(category)

        if routing_version == "v1":
            action = route_v1(priority, state.get("kb_confidence"))
        else:
            ticket_text = state.get("normalized") or state["raw_text"]
            action = route_v2(priority, ticket_text, state.get("kb_chunks") or [], checker)

        return {
            "destination": destination,
            "action": action,
            "draft_reply": None,  # draft generation lands in the next PR
            "trace": ["route[{}]: {} -> {} ({})".format(routing_version, category, destination, action)],
        }

    return route_node
