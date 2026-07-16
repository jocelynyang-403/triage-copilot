"""Route node: decide auto-reply vs. escalation and target destination."""

# Derived from evals/golden.jsonl by majority vote per category.
# Ceiling with this table = 39/52 destination accuracy. Refined in Phase 2.
_CATEGORY_TO_DESTINATION = {
    "billing": "Billing Ops",         # 10/11 rows
    "bug": "Engineering",             #  6/10 rows
    "access_request": "IT / Access",  #  7/10 rows
    "sales_lead": "Sales",            # 10/10 rows
    "other": "Support",               #  6/11 rows
}

# "Human Review" is the fallback for an unknown/absent category. The two golden rows
# labeled Human Review (notfound-01/02) are `other`+P3 and are NOT reachable by a
# rule-based router — they are separated only by kb_confidence in Phase 2. This is
# expected; do not hack the table to chase them.
_FALLBACK_DESTINATION = "Human Review"


def route_node(state):
    category = state.get("category")
    # `priority` is read but unused in Phase 1: the golden set does not support a
    # priority-dependent destination rule. Kept as a Phase 2 extension point.
    destination = _CATEGORY_TO_DESTINATION.get(category, _FALLBACK_DESTINATION)
    return {
        "destination": destination,
        "action": "escalate",
        "draft_reply": None,
        "trace": ["route: {} -> {} (escalate)".format(category, destination)],
    }
