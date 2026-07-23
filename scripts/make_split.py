"""Create a stratified dev/test split of the golden eval set (Phase 2 prep).

The golden set (`evals/golden.jsonl`) is READ-ONLY. This script references rows
BY ID only and never copies row content or writes back to the golden file. It
produces `evals/split.json`, which must be committed BEFORE any kb_confidence /
threshold number is ever computed, so the git timestamp proves the split
predates tuning and there is no train/test leak.

  dev  = 16 rows  -> Phase 2 threshold tuning ONLY
  test = 36 rows  -> Phase 4 headline metrics (never inspected until frozen)

The split is stratified by the (category, expected_action) pair, deterministic
(SEED = 42, local RNG, inputs sorted before shuffling), and forces the two
genuinely KB-uncovered negatives (notfound-01/02) into TEST.

Run from the repo root:  python -m scripts.make_split
                    or:  python scripts/make_split.py
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

SEED = 42
TOTAL = 52
DEV_TARGET = 16
TEST_TARGET = 36
FORCED_TEST = ("notfound-01", "notfound-02")

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "golden.jsonl")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "evals", "split.json")


def load_golden(path):
    # type: (str) -> List[dict]
    """Load the golden eval set read-only (open for reading only)."""
    rows = []  # type: List[dict]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_strata(rows):
    # type: (List[dict]) -> Dict[Tuple[str, str], List[dict]]
    """Group rows by the (category, expected_action) pair."""
    strata = defaultdict(list)  # type: Dict[Tuple[str, str], List[dict]]
    for row in rows:
        key = (row["category"], row["expected_action"])
        strata[key].append(row)
    return strata


def stratum_sort_key(alloc, key):
    # type: (Dict[Tuple[str, str], dict], Tuple[str, str]) -> Tuple[int, str]
    """Deterministic ordering for adjustment: largest stratum first, then by id.

    The min id across the whole stratum (available + forced) is a stable
    secondary key that does not depend on dict or file ordering.
    """
    a = alloc[key]
    all_ids = [r["id"] for r in a["available"]] + [r["id"] for r in a["forced"]]
    return (-a["n"], min(all_ids))


def main():
    # type: () -> None
    parser = argparse.ArgumentParser(
        description="Create a stratified, reproducible dev/test split of the golden eval set."
    )
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output path for split.json (default: evals/split.json).")
    args = parser.parse_args()

    rows = load_golden(GOLDEN_PATH)
    assert len(rows) == TOTAL, "expected {} golden rows, got {}".format(TOTAL, len(rows))

    strata = build_strata(rows)

    # Single local RNG instance (NOT the global `random`) processed over strata in
    # a fixed sorted-key order, so the shuffle stream is fully reproducible.
    rng = random.Random(SEED)

    alloc = {}  # type: Dict[Tuple[str, str], dict]
    for key in sorted(strata.keys()):
        # Sort by id first (stable), then shuffle with the local RNG.
        stratum_rows = sorted(strata[key], key=lambda r: r["id"])
        rng.shuffle(stratum_rows)

        n = len(stratum_rows)
        # Forced rows always land in TEST and are excluded from the dev pool.
        available = [r for r in stratum_rows if r["id"] not in FORCED_TEST]
        forced = [r for r in stratum_rows if r["id"] in FORCED_TEST]

        if n == 1:
            # Can't split a singleton; protect it by putting it in TEST.
            dev_count = 0
        else:
            dev_count = int(round(n * DEV_TARGET / float(TOTAL)))
            # Every stratum with >= 2 rows contributes >= 1 to dev AND >= 1 to test.
            dev_count = max(1, min(dev_count, n - 1))

        # Never assign more to dev than there are non-forced rows available.
        dev_count = min(dev_count, len(available))

        alloc[key] = {
            "n": n,
            "available": available,
            "forced": forced,
            "dev_count": dev_count,
        }

    dev_total = sum(a["dev_count"] for a in alloc.values())

    # Deterministic boundary adjustment until dev == 16 exactly. Largest strata
    # first, then by id (see stratum_sort_key). Not expected to trigger for the
    # current 52-row golden set, but kept correct for robustness.
    while dev_total > DEV_TARGET:
        candidates = [k for k in alloc if alloc[k]["dev_count"] > 1]
        assert candidates, "cannot reduce dev below target without emptying a stratum's dev side"
        candidates.sort(key=lambda k: stratum_sort_key(alloc, k))
        alloc[candidates[0]]["dev_count"] -= 1
        dev_total -= 1

    while dev_total < DEV_TARGET:
        candidates = []
        for k in alloc:
            a = alloc[k]
            new_dev = a["dev_count"] + 1
            if new_dev > len(a["available"]):
                continue
            # Keep at least 1 row on the test side of this stratum.
            test_remaining = (len(a["available"]) - new_dev) + len(a["forced"])
            if test_remaining < 1:
                continue
            candidates.append(k)
        assert candidates, "cannot raise dev to target without emptying a stratum's test side"
        candidates.sort(key=lambda k: stratum_sort_key(alloc, k))
        alloc[candidates[0]]["dev_count"] += 1
        dev_total += 1

    dev_ids = []  # type: List[str]
    test_ids = []  # type: List[str]
    for key in sorted(alloc.keys()):
        a = alloc[key]
        dc = a["dev_count"]
        dev_here = a["available"][:dc]
        test_here = a["available"][dc:] + a["forced"]
        dev_ids.extend(r["id"] for r in dev_here)
        test_ids.extend(r["id"] for r in test_here)

    dev_ids = sorted(dev_ids)
    test_ids = sorted(test_ids)

    # --- Self-checks (fail loudly, non-zero exit) ---
    dev_set = set(dev_ids)
    test_set = set(test_ids)
    all_ids = set(r["id"] for r in rows)

    assert len(dev_ids) == DEV_TARGET, "dev has {} ids (expected {})".format(len(dev_ids), DEV_TARGET)
    assert len(test_ids) == TEST_TARGET, "test has {} ids (expected {})".format(len(test_ids), TEST_TARGET)
    assert len(dev_set) == len(dev_ids), "duplicate id in dev_ids"
    assert len(test_set) == len(test_ids), "duplicate id in test_ids"
    assert dev_set.isdisjoint(test_set), "dev and test overlap: {}".format(sorted(dev_set & test_set))
    assert dev_set | test_set == all_ids, "union of split != all golden ids"
    for fid in FORCED_TEST:
        assert fid in test_set, "forced-test id {!r} did not land in test".format(fid)

    # --- Write split.json (exact shape) ---
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "dev_ids": dev_ids, "test_ids": test_ids}, f, indent=2)
        f.write("\n")

    # --- Stratification report ---
    print("Stratification report (dev / test) by (category, expected_action):")
    print("  {:<34} {:>3} {:>5} {:>6}".format("stratum", "dev", "test", "total"))
    for key in sorted(alloc.keys()):
        a = alloc[key]
        dc = a["dev_count"]
        tc = a["n"] - dc
        label = "{}/{}".format(key[0], key[1])
        print("  {:<34} {:>3} {:>5} {:>6}".format(label, dc, tc, a["n"]))

    print()
    print("Totals: dev = {} / test = {} (of {})".format(len(dev_ids), len(test_ids), len(all_ids)))
    print("Forced-to-test confirmed: {} -> test, {} -> test".format(FORCED_TEST[0], FORCED_TEST[1]))
    print("Wrote {}".format(os.path.abspath(out_path)))


if __name__ == "__main__":
    main()
