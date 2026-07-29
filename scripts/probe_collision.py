"""Measure whether kb_confidence alone can separate escalate from auto_reply.

The Phase 2 go/no-go experiment. The allowlist guard escalates every P0/P1 up front,
so retrieval confidence only has to carry the P2/P3 rows. If the P2/P3 escalate rows
score in the same band as the P2/P3 auto_reply rows, no single threshold can split
them — the empirical case for routing v2 (an LLM grounding check) over a pure cutoff.

Dev split ONLY (16 rows). No API key: local embeddings + regex intake.
Run from the repo root:  python -m scripts.probe_collision
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nodes.intake import intake_node  # noqa: E402
from src.nodes.knowledge import make_knowledge_node  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN = os.path.join(ROOT, "evals", "golden.jsonl")
SPLIT = os.path.join(ROOT, "evals", "split.json")


def load_dev():
    with open(SPLIT) as f:
        dev_ids = set(json.load(f)["dev_ids"])
    rows = []
    with open(GOLDEN) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r["id"] in dev_ids]


def confidence_for(node, row):
    state = {"raw_text": row["raw_text"], "requester": "probe", "channel": "probe", "entities": {}}
    state.update(intake_node(state))
    out = node(state)
    return out["kb_confidence"]


def main():
    rows = load_dev()
    node = make_knowledge_node()

    scored = [(confidence_for(node, r), r) for r in rows]

    contested = [s for s in scored if s[1]["priority"] in ("P2", "P3")]
    guard_caught = [s for s in scored if s[1]["priority"] in ("P0", "P1")]

    auto = sorted([s for s in contested if s[1]["expected_action"] == "auto_reply"],
                  key=lambda s: s[0], reverse=True)
    esc = sorted([s for s in contested if s[1]["expected_action"] == "escalate"],
                 key=lambda s: s[0], reverse=True)

    def show(title, group):
        print("\n== {} ({}) ==".format(title, len(group)))
        for conf, r in group:
            print("  {:6.3f}  {:20s} {:3s} {:10s} | {}".format(
                conf, r["id"], r["priority"], r["expected_action"],
                r["raw_text"][:58].replace("\n", " ")))

    show("P2/P3 auto_reply  (want HIGH confidence)", auto)
    show("P2/P3 escalate    (want LOW confidence)", esc)
    show("P0/P1 (pre-caught by allowlist; reference only)",
         sorted(guard_caught, key=lambda s: s[0], reverse=True))

    print("\n--- separability verdict (P2/P3 only) ---")
    if not auto or not esc:
        print("  not enough rows in one group to judge.")
        return
    min_auto = min(s[0] for s in auto)
    max_esc = max(s[0] for s in esc)
    print("  min auto_reply confidence = {:.3f}".format(min_auto))
    print("  max escalate  confidence = {:.3f}".format(max_esc))
    if min_auto > max_esc:
        print("  SEPARABLE: a threshold in ({:.3f}, {:.3f}] splits them.".format(max_esc, min_auto))
        print("  Pure kb_confidence routing (v1) may suffice on dev.")
    else:
        print("  COLLISION: some escalate rows score >= some auto_reply rows.")
        print("  No single threshold separates them -> routing v2 (LLM grounding) is justified.")


if __name__ == "__main__":
    main()
