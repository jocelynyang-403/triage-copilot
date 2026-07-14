"""Generate a synthetic, VARIED ops-ticket dataset with model-guessed DRAFT labels.

Writes JSONL to evals/golden.draft.jsonl. The labels are guesses only: a human must
review and correct every one, then rename the file to evals/golden.jsonl before it is
used in evals. This script refuses to overwrite an existing draft so a re-run cannot
clobber human-reviewed work.

Run from the repo root:  python scripts/gen_dataset.py
"""

import json
import os
import sys

from langchain_anthropic import ChatAnthropic

# Allow running as a plain script (`python scripts/gen_dataset.py`) by putting the
# repo root on the path so `src` is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CATEGORIES, DATASET_MODEL  # noqa: E402

OUTPUT_PATH = os.path.join("evals", "golden.draft.jsonl")
TICKETS_PER_CATEGORY = 10

# Two short hand-written seed examples per category to anchor tone and variety.
# These are illustrative only; the model should produce fresh, varied tickets.
SEED_EXAMPLES = {
    "billing": [
        "I was charged twice for order #48211 this month — can I get one of the "
        "charges refunded?",
        "Why did my plan jump from $29 to $79? I never upgraded and want this fixed "
        "before the next cycle.",
    ],
    "bug": [
        "The export button throws error code ERR_500 every time I click it on the "
        "reports page.",
        "App keeps logging me out after ~30 seconds on iOS 18. Started yesterday, "
        "totally unusable now.",
    ],
    "access_request": [
        "Hi, new hire on the data team — could I get read access to the analytics "
        "dashboard?",
        "Please add me to the #deploys Slack channel and the staging environment, "
        "manager approved.",
    ],
    "sales_lead": [
        "We're a 200-person company evaluating your Enterprise tier — can someone "
        "walk us through pricing?",
        "Interested in a demo for our support org, roughly 40 seats. What's the next "
        "step?",
    ],
    "other": [
        "Just wanted to say the new dark mode looks great, thanks team!",
        "Is your office open on the public holiday next Monday?",
    ],
}


def build_prompt(category: str) -> str:
    """Build a per-category generation prompt with seed examples embedded."""
    seeds = "\n".join(f"- {s}" for s in SEED_EXAMPLES[category])
    return f"""You are generating synthetic ops-support tickets for the category \
"{category}".

Here are two seed examples to anchor tone and variety (do NOT copy them verbatim):
{seeds}

Generate 10 tickets that are clearly different from one another. Vary the
phrasing, the message length (from a single line to a full paragraph), the
tone (frustrated / polite / terse / confused), and the urgency level. Some
tickets should include concrete details such as order IDs or error codes;
others should be vague and underspecified. Do not reuse sentence templates
or openings across tickets.

For each ticket, also propose DRAFT labels (these are best-effort guesses):
- "id": a short unique slug, e.g. "{category}-01"
- "raw_text": the ticket message text
- "category": must be exactly "{category}"
- "priority": one of "P0", "P1", "P2", "P3" (P0 = most urgent)
- "expected_action": one of "auto_reply" or "escalate"
- "expected_destination": a plausible team name to route to, e.g. "Billing Ops",
  "Engineering", "IT / Access", "Sales", "Support"

Return ONLY a JSON array of {TICKETS_PER_CATEGORY} objects with exactly those keys.
No prose, no markdown fences — just the raw JSON array."""


def _extract_json_array(text) -> list:
    """Parse a JSON array from the model response, tolerating stray formatting."""
    # response.content may be a plain string or a list of content blocks
    # (e.g. [{"type": "text", "text": "..."}]); normalize to a single string.
    if isinstance(text, list):
        pieces = []
        for item in text:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                pieces.append(item.get("text", ""))
            elif hasattr(item, "text"):
                pieces.append(getattr(item, "text"))
            else:
                pieces.append(str(item))
        text = "".join(pieces)
    text = text.strip()
    if text.startswith("```"):
        # Strip a leading ```json / ``` fence and the trailing fence if present.
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON array found in model response. Normalized text was: "
            f"{text[:800]!r}"
        )
    return json.loads(text[start : end + 1])


def main() -> None:
    if os.path.exists(OUTPUT_PATH):
        print(
            f"Refusing to overwrite existing {OUTPUT_PATH}. "
            "Move or delete it first if you really want to regenerate."
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Sonnet has deprecated the temperature parameter; diversity is driven by the
    # prompt instead (see build_prompt).
    llm = ChatAnthropic(model=DATASET_MODEL, max_tokens=4096)

    all_tickets = []
    for category in CATEGORIES:
        print(f"Generating {TICKETS_PER_CATEGORY} tickets for category: {category} ...")
        response = llm.invoke(build_prompt(category))
        tickets = _extract_json_array(response.content)
        for i, ticket in enumerate(tickets, start=1):
            # Enforce the category and guarantee a unique id even if the model slips.
            ticket["category"] = category
            ticket.setdefault("id", f"{category}-{i:02d}")
            all_tickets.append(ticket)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ticket in all_tickets:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_tickets)} draft tickets to {OUTPUT_PATH}")
    print(
        "DRAFT labels are model-guessed and NOT trustworthy. Review/correct every\n"
        "label by hand, then rename to evals/golden.jsonl before using in evals."
    )


if __name__ == "__main__":
    main()
