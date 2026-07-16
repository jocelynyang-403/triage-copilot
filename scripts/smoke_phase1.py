"""Phase 1 smoke test for the triage graph.

Runs the intake -> classify -> route pipeline over rows from the READ-ONLY golden
eval set and asserts the structural invariants of Phase 1. Fails loudly (non-zero
exit) on any violation. The agreement summary is INFORMATIONAL only — Phase 4 owns
the real metrics.

Run from the repo root:  python -m scripts.smoke_phase1
                    or:  python scripts/smoke_phase1.py
"""

import argparse
import json
import os
import sys

# Allow running as a plain script (`python scripts/smoke_phase1.py`) by putting the
# repo root on the path so `src` is importable — same convention as gen_dataset.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CATEGORIES, PRIORITIES, DESTINATIONS  # noqa: E402
from src.graph import build_graph  # noqa: E402

GOLDEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evals", "golden.jsonl"))


def load_golden():
    """Load the golden eval set read-only (open for reading only)."""
    rows = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows, run_all):
    if run_all:
        return rows
    # One row per category: first-seen per golden category label.
    picked = []
    seen = set()
    for r in rows:
        cat = r.get("category")
        if cat not in seen:
            seen.add(cat)
            picked.append(r)
    return picked


def mark(ok):
    return "\u2713" if ok else "\u2717"


def main():
    parser = argparse.ArgumentParser(description="Phase 1 smoke test for the triage graph.")
    parser.add_argument("--all", action="store_true",
                        help="Run all rows (default: one row per category).")
    parser.add_argument("--prompt-version", default="v1",
                        help="Prompt version to load (default: v1).")
    args = parser.parse_args()

    rows = load_golden()
    selected = select_rows(rows, args.all)

    graph = build_graph(args.prompt_version)

    failures = []
    first_trace = None

    cat_agree = 0
    pri_agree = 0
    dest_agree = 0
    total = len(selected)

    for r in selected:
        out = graph.invoke({
            "raw_text": r["raw_text"],
            "requester": "smoke",
            "channel": "smoke",
            "entities": {},
            # Deliberately NOT passing `trace`, to prove the reducer's default works.
        })

        if first_trace is None:
            first_trace = out.get("trace")

        pred_cat = out.get("category")
        pred_pri = out.get("priority")
        pred_dest = out.get("destination")
        pred_action = out.get("action")
        trace = out.get("trace") or []

        gold_cat = r.get("category")
        gold_pri = r.get("priority")
        gold_dest = r.get("expected_destination")

        # --- assertions (collect failures rather than crashing) ---
        cat_ok = pred_cat in CATEGORIES
        pri_ok = pred_pri in PRIORITIES
        dest_ok = pred_dest is not None and pred_dest in DESTINATIONS
        action_ok = pred_action == "escalate"
        trace_ok = len(trace) == 3

        row_fails = []
        if not cat_ok:
            row_fails.append("category {!r} not in CATEGORIES".format(pred_cat))
        if not pri_ok:
            row_fails.append("priority {!r} not in PRIORITIES".format(pred_pri))
        if not dest_ok:
            row_fails.append("destination {!r} not in DESTINATIONS".format(pred_dest))
        if not action_ok:
            row_fails.append("action {!r} != 'escalate'".format(pred_action))
        if not trace_ok:
            row_fails.append("len(trace) == {} (expected 3)".format(len(trace)))

        if row_fails:
            failures.append((r.get("id"), row_fails))

        # --- informational agreement (NOT a gate) ---
        if pred_cat == gold_cat:
            cat_agree += 1
        if pred_pri == gold_pri:
            pri_agree += 1
        if pred_dest == gold_dest:
            dest_agree += 1

        # --- per-row line ---
        print("{id} | {cm}{pc}/{gc} | {pm}{pp}/{gp} | {dm}{pd}/{gd}".format(
            id=r.get("id"),
            cm=mark(pred_cat == gold_cat), pc=pred_cat, gc=gold_cat,
            pm=mark(pred_pri == gold_pri), pp=pred_pri, gp=gold_pri,
            dm=mark(pred_dest == gold_dest), pd=pred_dest, gd=gold_dest,
        ))

    print()
    print("--- INFORMATIONAL agreement vs. golden (NOT a pass/fail gate; Phase 4 owns metrics) ---")
    print("  category:    {}/{}".format(cat_agree, total))
    print("  priority:    {}/{}".format(pri_agree, total))
    print("  destination: {}/{}".format(dest_agree, total))

    print()
    print("First-row trace ({} entries):".format(len(first_trace) if first_trace else 0))
    for entry in (first_trace or []):
        print("  - {}".format(entry))

    print()
    if failures:
        print("SMOKE FAILED: {} row(s) with assertion failures:".format(len(failures)))
        for row_id, fails in failures:
            for f in fails:
                print("  {}: {}".format(row_id, f))
        sys.exit(1)

    print("SMOKE PASSED: {} row(s), all assertions held.".format(total))
    sys.exit(0)


if __name__ == "__main__":
    main()
