#!/usr/bin/env python3
"""
Stamp the canonical experiment feature profile onto every config in a directory,
then verify the required ON/OFF state. Idempotent.

Active:   verdict cache, two-tier trace-check timeout, adaptive (cv_pr) stopping,
          time quantization.
Inactive: polarity analysis (no switch: dormant unless the two below are on),
          interval inference, adaptive mutation range.

Usage:
    python3 apply_feature_profile.py <config_dir> [--profile experiment_feature_profile.json]
    python3 apply_feature_profile.py <config_dir> --verify-only
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if k == "_comment":
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def check(cfg: dict) -> list[str]:
    """Return a list of violations; empty = compliant."""
    g = lambda *p: _get(cfg, p)
    problems = []
    # ACTIVE
    if g("evaluation", "cache_enabled") is not True:
        problems.append("verdict cache NOT active (evaluation.cache_enabled != true)")
    if g("heuristics", "two_tier_timeout", "enabled") is not True:
        problems.append("two-tier timeout NOT active")
    if g("ga", "stopping", "mode") != "cv_pr":
        problems.append(f"adaptive stop NOT active (ga.stopping.mode = {g('ga','stopping','mode')!r}, want 'cv_pr')")
    if g("heuristics", "time_quantization", "enabled") is not True:
        problems.append("time quantization NOT active")
    # INACTIVE
    if g("heuristics", "interval_inference", "enabled") is True:
        problems.append("interval inference IS active (must be off)")
    if g("heuristics", "adaptive_range", "enabled") is True:
        problems.append("adaptive range IS active (must be off) -> also keeps polarity dormant")
    # CONSISTENCY
    hi = g("heuristics", "two_tier_timeout", "high_sec")
    cap = g("evaluation", "trace_check_timeout_sec")
    if isinstance(hi, (int, float)) and isinstance(cap, (int, float)) and hi > cap:
        problems.append(f"two_tier high_sec ({hi}) > trace_check_timeout_sec ({cap})")
    if g("ga", "stopping", "mode") == "cv_pr":
        ms = g("ga", "stopping", "min_samples")
        if not isinstance(ms, int) or ms < 1:
            problems.append(f"cv_pr needs min_samples >= 1 (got {ms})")
    return problems


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_dir", type=Path)
    ap.add_argument("--profile", type=Path, default=HERE / "experiment_feature_profile.json")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args()

    profile = json.loads(args.profile.read_text())
    files = sorted(args.config_dir.glob(args.glob))
    if not files:
        print(f"No configs matched {args.config_dir}/{args.glob}", file=sys.stderr)
        return 2

    any_bad = False
    for f in files:
        cfg = json.loads(f.read_text())
        if not args.verify_only:
            deep_merge(cfg, profile)
            f.write_text(json.dumps(cfg, indent=1) + "\n")
        problems = check(cfg)
        status = "OK" if not problems else "FAIL"
        any_bad |= bool(problems)
        print(f"[{status}] {f.name}")
        for p in problems:
            print(f"        - {p}")

    print(f"\n{'VERIFY' if args.verify_only else 'APPLY'}: {len(files)} configs, "
          f"{'all compliant' if not any_bad else 'VIOLATIONS ABOVE'}")
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
