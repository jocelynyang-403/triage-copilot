"""Compare routing v1 (threshold) vs v2 (LLM grounding) on the dev split.

Isolates the routing MECHANISM: feeds GOLD priority into the allowlist guard so
classifier priority errors (Phase 1: ~65% priority acc) do not contaminate the
comparison. Retrieval is real (intake -> knowledge). v1 and v2 differ ONLY in how a
guard-eligible P2/P3 ticket is decided, so the delta here is pure routing signal.

Needs ANTHROPIC_API_KEY (v2 grounding calls Haiku on the contested rows). Dev split
only. Run from the repo root:  python -m scripts.compare_routing
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nodes.intake import intake_node  # noqa: E402
from src.nodes.knowledge import make_knowledge_node  # noqa: E402
from src.nodes.route import route_v1, guard_allows, make_grounding_checker  # noqa: E402

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


def mark(ok):
    return "\u2713" if ok else "\u2717"


def main():
    rows = load_dev()
    knowledge = make_knowledge_node()
    checker = make_grounding_checker()

    v1_ok = v2_ok = 0
    c_v1_ok = c_v2_ok = c_total = 0

    for r in rows:
        state = {"raw_text": r["raw_text"], "requester": "cmp", "channel": "cmp", "entities": {}}
        state.update(intake_node(state))
        state.update(knowledge(state))

        gold_pri = r["priority"]
        gold_act = r["expected_action"]
        conf = state.get("kb_confidence") or 0.0
        text = state.get("normalized") or state["raw_text"]
        chunks = state.get("kb_chunks") or []

        a1 = route_v1(gold_pri, conf)

        # Replicate route_v2 inline so we can surface the grounding reasoning.
        if not guard_allows(gold_pri):
            a2, reason = "escalate", "guard: P0/P1"
        elif not chunks:
            a2, reason = "escalate", "no chunks"
        else:
            v = checker(text, chunks)
            a2 = "auto_reply" if v.answerable else "escalate"
            reason = v.reasoning

        ok1 = a1 == gold_act
        ok2 = a2 == gold_act
        v1_ok += ok1
        v2_ok += ok2
        contested = gold_pri in ("P2", "P3")
        if contested:
            c_total += 1
            c_v1_ok += ok1
            c_v2_ok += ok2

        tag = " <-- contested" if contested else ""
        print("{:20s} {:3s} gold={:10s} conf={:5.3f} | v1 {}{:10s} | v2 {}{:10s}{}".format(
            r["id"], gold_pri, gold_act, conf, mark(ok1), a1, mark(ok2), a2, tag))
        if contested:
            print("        grounding: {}".format(reason[:80]))

    n = len(rows)
    print("\n--- dev action accuracy ---")
    print("  all dev ({} rows):         v1 {}/{}   v2 {}/{}".format(n, v1_ok, n, v2_ok, n))
    print("  contested P2/P3 ({} rows):  v1 {}/{}    v2 {}/{}".format(
        c_total, c_v1_ok, c_total, c_v2_ok, c_total))
    print("\nContested rows are the only ones the routing mechanism decides;")
    print("P0/P1 are escalated by the allowlist guard in both versions.")


if __name__ == "__main__":
    main()
