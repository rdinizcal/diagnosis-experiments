#!/usr/bin/env python3
"""Aggregate decision-tree statistics across a batch of runs.

Input is a directory of collected run artifacts (one subdirectory per matrix
cell, each containing report.json, summary.json, the J48 ``.out`` files, and a
run_meta.json written by run_batch.py). Runs are grouped by experiment
(``<subject>_<exp>``) and reduced across seeds into per-experiment statistics:

  * root-split attribute frequency,
  * normalized root+depth-2 tree-hash distribution (via the package's own
    ``parse_j48_out`` comparator, so it matches the tree_stable stop criterion),
  * per-position threshold mean/std/min/max and 95% CI,
  * recovered ``boundary_at`` statistics,
  * cross-validated precision/recall mean/std,
  * stop-reason and one-class finding counts,
  * count of runs that stopped on the wall guard.

Outputs one CSV per table plus a SUMMARY.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from diagnosis.diagnostics.summary import parse_j48_out

SPLIT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>|=)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
OUT_RE = re.compile(r"J48-data-(\d+)\.out$")


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _experiment_id(run_dir: Path, meta: Optional[dict]) -> str:
    if meta and meta.get("subject") and meta.get("exp"):
        return f"{meta['subject']}_{meta['exp']}"
    # fall back to the <subject>_<exp> token in the path (with or without _s<seed>)
    m = re.search(r"([A-Za-z0-9]+_exp\d+)", str(run_dir))
    return m.group(1) if m else run_dir.name


def _pick_tree_out(run_dir: Path) -> Optional[Path]:
    """The representative tree = the J48 out with the most instances."""
    best, best_qty = None, -1
    for out in run_dir.rglob("*.out"):
        m = OUT_RE.search(out.name)
        qty = int(m.group(1)) if m else 0
        if qty >= best_qty:
            best, best_qty = out, qty
    return best


def _tree_thresholds(out_text: str) -> dict[str, float]:
    """Attribute -> split threshold, one per attribute (first occurrence)."""
    thresholds: dict[str, float] = {}
    for raw in out_text.splitlines():
        line = raw.strip().lstrip("| ").strip()
        m = SPLIT_RE.match(line)
        if m and m.group(1) not in thresholds:
            thresholds[m.group(1)] = float(m.group(3))
    return thresholds


def _ci95(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    n = len(values)
    if n < 2:
        return (None, None)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(n)
    return (mean - half, mean + half)


class RunRecord:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.report = _read_json(run_dir / "report.json") or {}
        self.summary = _read_json(run_dir / "summary.json") or {}
        self.meta = _read_json(run_dir / "run_meta.json")
        self.experiment = _experiment_id(run_dir, self.meta)

        out = _pick_tree_out(run_dir)
        text = out.read_text(encoding="utf-8", errors="replace") if out else ""
        stats = parse_j48_out(text, include_stopping_metrics=True) if text else {}
        self.root_split = stats.get("root_split")
        self.tree_hash = stats.get("tree_hash")
        self.cv_precision = stats.get("cv_precision")
        self.cv_recall = stats.get("cv_recall")
        self.thresholds = _tree_thresholds(text) if text else {}

        self.boundary_at = self.report.get("boundary_at") or {}
        self.one_class = bool(self.report.get("one_class_space"))
        self.wall_hit = bool(self.meta.get("wall_hit")) if self.meta else False

        checks = self.summary.get("ga_stopping_checks") or []
        if checks and isinstance(checks[-1], dict):
            self.stop_reason = checks[-1].get("reason", "unknown")
        else:
            self.stop_reason = "count"

    @property
    def root_attr(self) -> Optional[str]:
        if not self.root_split:
            return None
        m = SPLIT_RE.match(self.root_split.lstrip("| ").strip())
        return m.group(1) if m else None


def aggregate(runs_dir: Path, out_dir: Path) -> str:
    run_dirs = sorted({p.parent for p in runs_dir.rglob("report.json")})
    records = [RunRecord(d) for d in run_dirs]
    by_exp: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_exp[r.experiment].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _w(name: str, header: list[str], rows: list[list[Any]]) -> None:
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    root_rows, hash_rows, thr_rows, bnd_rows = [], [], [], []
    cv_rows, stop_rows, flag_rows = [], [], []

    for exp in sorted(by_exp):
        recs = by_exp[exp]
        n = len(recs)

        for attr, c in Counter(r.root_attr for r in recs if r.root_attr).most_common():
            root_rows.append([exp, attr, c, round(c / n, 4)])

        for h, c in Counter(r.tree_hash for r in recs if r.tree_hash).most_common():
            hash_rows.append([exp, h, c, round(c / n, 4)])

        thr_by_attr: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            for attr, val in r.thresholds.items():
                thr_by_attr[attr].append(val)
        for attr in sorted(thr_by_attr):
            vals = thr_by_attr[attr]
            lo, hi = _ci95(vals)
            thr_rows.append([
                exp, attr, len(vals),
                statistics.fmean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0,
                min(vals), max(vals), lo, hi,
            ])

        bnd_by_pos: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            for pos, val in r.boundary_at.items():
                bnd_by_pos[str(pos)].append(float(val))
        for pos in sorted(bnd_by_pos):
            vals = bnd_by_pos[pos]
            lo, hi = _ci95(vals)
            bnd_rows.append([
                exp, pos, len(vals), statistics.fmean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0,
                min(vals), max(vals), lo, hi,
            ])

        prec = [r.cv_precision for r in recs if r.cv_precision is not None]
        rec = [r.cv_recall for r in recs if r.cv_recall is not None]
        cv_rows.append([
            exp, n,
            statistics.fmean(prec) if prec else None,
            statistics.stdev(prec) if len(prec) > 1 else 0.0,
            statistics.fmean(rec) if rec else None,
            statistics.stdev(rec) if len(rec) > 1 else 0.0,
        ])

        for reason, c in Counter(r.stop_reason for r in recs).most_common():
            stop_rows.append([exp, reason, c])

        flag_rows.append([
            exp, n,
            sum(1 for r in recs if r.one_class),
            sum(1 for r in recs if r.wall_hit),
        ])

    _w("root_split_frequency.csv", ["experiment", "attribute", "count", "fraction"], root_rows)
    _w("tree_hash_distribution.csv", ["experiment", "tree_hash", "count", "fraction"], hash_rows)
    _w("position_thresholds.csv",
       ["experiment", "attribute", "n", "mean", "std", "min", "max", "ci95_low", "ci95_high"], thr_rows)
    _w("boundary_at.csv",
       ["experiment", "position", "n", "mean", "std", "min", "max", "ci95_low", "ci95_high"], bnd_rows)
    _w("cv_metrics.csv",
       ["experiment", "n", "precision_mean", "precision_std", "recall_mean", "recall_std"], cv_rows)
    _w("stop_reasons.csv", ["experiment", "reason", "count"], stop_rows)
    _w("run_flags.csv", ["experiment", "n_runs", "one_class_findings", "wall_guard_hits"], flag_rows)

    lines = ["# Tree aggregation summary", ""]
    lines.append(f"- Runs found: **{len(records)}** across **{len(by_exp)}** experiments")
    lines.append(f"- Wall-guard stops: **{sum(r.wall_hit for r in records)}**")
    lines.append(f"- One-class findings: **{sum(r.one_class for r in records)}**")
    lines.append("")
    lines.append("| experiment | runs | top root split | distinct trees | stop reasons |")
    lines.append("| --- | --- | --- | --- | --- |")
    for exp in sorted(by_exp):
        recs = by_exp[exp]
        top = Counter(r.root_attr for r in recs if r.root_attr).most_common(1)
        top_s = f"{top[0][0]} ({top[0][1]}/{len(recs)})" if top else "-"
        distinct = len({r.tree_hash for r in recs if r.tree_hash})
        reasons = ", ".join(f"{k}:{v}" for k, v in Counter(r.stop_reason for r in recs).most_common())
        lines.append(f"| {exp} | {len(recs)} | {top_s} | {distinct} | {reasons} |")
    summary_md = "\n".join(lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    return summary_md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True, help="directory of collected run artifacts")
    ap.add_argument("--out", default="aggregate_out", help="output directory for CSVs + SUMMARY.md")
    args = ap.parse_args()
    summary = aggregate(Path(args.runs_dir), Path(args.out))
    print(summary)


if __name__ == "__main__":
    main()
