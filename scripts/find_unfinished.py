#!/usr/bin/env python3
"""Classify each expected experiment as finished or not, and emit a re-run list.

For every expected (subject, exp, seed) cell it inspects the newest downloaded
run and assigns one of:

  OK          - produced a J48 tree and terminated normally
  MISSING     - no run found in --runs-dir (never uploaded / killed at the cap)
  WALL        - stopped by the internal wall guard (run_meta.wall_hit)
  NO_TREE     - ran but built no tree; sub-split:
                  * two-class  -> a real failure/crash worth re-running
                  * one-class  -> degenerate (no verdict variety); re-run won't help

Re-run candidates = MISSING + WALL + NO_TREE(two-class). These are printed as a
`cells` string to paste into batch.yml's `cells` dispatch input. Degenerate
one-class cells are listed separately (a config problem, not a time problem).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

SUBJECTS = ["AT1", "AT2", "AT51", "AT52", "AT53", "AT54",
            "AT6A", "AT6B", "AT6C", "AT6ABC",
            "CC1", "CC2", "CC3", "CC4", "CC5", "CCX"]
EXPS = ["exp1", "exp3"]
RUN_RE = re.compile(r"outputs/([A-Za-z0-9]+)_(exp\d+)_s(\d+)/([^/]+)$")


def newest_runs(runs_dir: str) -> dict:
    best = {}
    for rep in glob.glob(f"{runs_dir}/**/report.json", recursive=True):
        m = RUN_RE.search(os.path.dirname(rep))
        if not m:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        stamp = m.group(4)
        if key not in best or stamp > best[key][0]:
            best[key] = (stamp, os.path.dirname(rep))
    return best


def has_tree(run_dir: str) -> bool:
    for o in glob.glob(f"{run_dir}/*.out"):
        try:
            if "J48 pruned tree" in open(o, errors="replace").read():
                return True
        except OSError:
            pass
    return False


def classify(run_dir: str) -> str:
    meta = {}
    mp = os.path.join(run_dir, "run_meta.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp))
        except Exception:
            meta = {}
    if meta.get("wall_hit"):
        return "WALL"
    if has_tree(run_dir):
        return "OK"
    # No tree: distinguish a degenerate one-class space from a real failure.
    try:
        r = json.load(open(os.path.join(run_dir, "report.json")))
    except Exception:
        return "NO_TREE_TWOCLASS"
    sat = (r.get("inferred_satisfied", 0) or 0) + (r.get("real_satisfied", 0) or 0)
    viol = (r.get("inferred_violated", 0) or 0) + (r.get("real_violated", 0) or 0)
    # fall back to any positive counts of both verdict kinds
    both = sat > 0 and viol > 0
    return "NO_TREE_TWOCLASS" if both else "NO_TREE_ONECLASS"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--runs-dir", default=os.path.join(repo, "online_results"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--exps", default=",".join(EXPS))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    subjects = SUBJECTS if args.subjects == "all" else args.subjects.split(",")
    exps = args.exps.split(",")
    best = newest_runs(args.runs_dir)

    buckets: dict[str, list[str]] = {}
    for subj in subjects:
        for exp in exps:
            for seed in seeds:
                cell = f"{subj}:{exp}:{seed}"
                run = best.get((subj, exp, seed))
                status = "MISSING" if run is None else classify(run[1])
                buckets.setdefault(status, []).append(cell)

    order = ["OK", "MISSING", "WALL", "NO_TREE_TWOCLASS", "NO_TREE_ONECLASS"]
    for st in order:
        if buckets.get(st):
            print(f"{st:18} {len(buckets[st]):>3}")
    print()

    rerun = buckets.get("MISSING", []) + buckets.get("WALL", []) + buckets.get("NO_TREE_TWOCLASS", [])
    if rerun:
        print("# Re-run these (unfinished / crashed) via batch.yml `cells` input:")
        print("cells=" + ",".join(sorted(rerun)))
    else:
        print("# Nothing to re-run: all expected cells finished.")

    degen = buckets.get("NO_TREE_ONECLASS", [])
    if degen:
        print()
        print("# Degenerate one-class cells (re-run will NOT help — needs a config fix):")
        print("# " + ",".join(sorted(degen)))


if __name__ == "__main__":
    main()
