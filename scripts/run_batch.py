#!/usr/bin/env python3
"""Run a set of generated configs under a wall-time guard.

Used by the CI batch job. For every config in ``--configs-dir`` this:

  * sets ``evaluation.parallel_workers`` to ``min(4, cpu_count)`` (the seed and
    every other knob come from the config, so results stay deterministic; only
    the worker count is derived from the runner),
  * runs the diagnosis pipeline under a SIGALRM deadline of ``--wall-minutes``,
  * on a normal finish or on the guard firing, writes ``run_meta.json`` next to
    the run's ``report.json`` (recording subject/exp/seed, whether the wall
    guard fired, and the runner hardware) and folds the runner block into
    ``report.json`` for transparency.

The verdict cache (SQLite, in the run's output dir) is the resume mechanism: a
run stopped by the guard leaves a complete, resumable state, and re-running the
same config with the same cache restored finishes the remaining work cheaply.
Timing numbers from CI are not used in the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import sys
from pathlib import Path

from diagnosis.config import load_config
from diagnosis.pipeline import run_diagnostics


class _WallGuard(Exception):
    pass


def _runner_block() -> dict:
    return {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def _name_parts(config_name: str) -> dict:
    m = re.match(r"([A-Za-z0-9]+)_(exp\d+)_s(\d+)", config_name)
    if m:
        return {"subject": m.group(1), "exp": m.group(2), "seed": int(m.group(3))}
    return {"subject": None, "exp": None, "seed": None}


def _latest_run_dir(output_root: Path) -> Path | None:
    reports = sorted(output_root.rglob("report.json"), key=lambda p: p.stat().st_mtime)
    return reports[-1].parent if reports else None


def run_one(config_path: Path, wall_seconds: int, workers: Optional[int],
            cache_dir: Optional[str] = None) -> bool:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data.setdefault("evaluation", {})
    # Only override when --workers was given explicitly; otherwise respect the
    # config (the all-on profile pins parallel_workers to 1). Default to 1 serial.
    if workers is not None:
        data["evaluation"]["parallel_workers"] = workers
    effective_workers = int(data["evaluation"].get("parallel_workers", 1) or 1)
    # Stable verdict-cache path enables resume: a re-run restores this file and
    # already-solved candidates return from cache instead of re-solving.
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        data["evaluation"]["cache_path"] = os.path.join(cache_dir, config_path.stem + ".sqlite")
    patched = config_path.with_suffix(".effective.json")
    patched.write_text(json.dumps(data, indent=2), encoding="utf-8")

    cfg = load_config(patched)
    output_root = Path(data["input"]["output_dir"])

    wall_hit = False

    def _on_alarm(signum, frame):
        raise _WallGuard()

    if hasattr(signal, "SIGALRM") and wall_seconds > 0:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(wall_seconds)
    try:
        run_diagnostics(cfg)
    except _WallGuard:
        wall_hit = True
        print(f"[run_batch] wall guard fired for {config_path.name}; "
              f"leaving resumable state", file=sys.stderr)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    run_dir = _latest_run_dir(output_root)
    parts = _name_parts(config_path.stem)
    meta = {**parts, "config": config_path.name, "wall_hit": wall_hit,
            "parallel_workers": effective_workers, "runner": _runner_block()}
    if run_dir is not None:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        report = run_dir / "report.json"
        if report.exists():
            try:
                rj = json.loads(report.read_text(encoding="utf-8"))
                rj["runner"] = _runner_block()
                # Record the worker count unconditionally (the GA only records it
                # when > 1), so a serial run is verifiable, not just implied.
                rj["parallel_workers"] = effective_workers
                report.write_text(json.dumps(rj, indent=2), encoding="utf-8")
            except Exception:
                pass
    return not wall_hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs-dir", required=True)
    ap.add_argument("--wall-minutes", type=float, default=330)
    ap.add_argument("--glob", default="*.json",
                    help="config filename glob (default '*.json')")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers per config. Default: respect the "
                         "config (the all-on profile pins this to 1 = serial). "
                         "Pass a number to override for all configs.")
    ap.add_argument("--cache-dir", default=None,
                    help="directory for a stable per-experiment verdict cache; "
                         "restore/save it across dispatches to resume runs.")
    args = ap.parse_args()

    configs = sorted(p for p in Path(args.configs_dir).glob(args.glob)
                     if not p.name.endswith(".effective.json"))
    if not configs:
        print(f"[run_batch] no configs matched {args.glob} in {args.configs_dir}")
        return

    # Split the wall budget across configs so the whole job finishes under cap.
    # --wall-minutes 0 disables the guard: runs go to completion (timing mode).
    per_config = int(args.wall_minutes * 60 / len(configs))
    budget = f"{per_config}s wall budget each" if per_config > 0 else "no wall guard (runs to completion)"
    wpol = f"{args.workers} workers (override)" if args.workers is not None else "workers from config (serial)"
    print(f"[run_batch] {len(configs)} configs, {wpol}, {budget}")

    completed = 0
    for cfg in configs:
        ok = run_one(cfg, per_config, args.workers, cache_dir=args.cache_dir)
        completed += int(ok)
        print(f"[run_batch] {cfg.name}: {'complete' if ok else 'wall-stopped (resumable)'}")
    print(f"[run_batch] {completed}/{len(configs)} finished without hitting the guard")


if __name__ == "__main__":
    main()
