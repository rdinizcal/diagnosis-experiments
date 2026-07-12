#!/usr/bin/env python3
"""Generate the evaluation batch configs.

For each (subject, experiment, seed) triple this emits one diagnosis JSON
config: the subject's requirement/traces and mutation block are taken from
``replication/evaluation_inputs/effectiveness/<SUBJECT>/<exp>.json``, and the
fixed "all-on" evaluation profile (verdict cache + cv_pr stopping + interval
inference + two-tier timeout + adaptive range) is overlaid on top. This is the
same profile used for the local batch reported in the paper; see the ALL_ON
constant below and the README section "All-on profile".

The seed comes only from the config (``ga.seed``); nothing else varies with the
runner. Determinism therefore depends solely on the config, not the machine.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

SUBJECTS = [
    "AT1", "AT2", "AT51", "AT52", "AT53", "AT54",
    "AT6A", "AT6B", "AT6C", "AT6ABC",
    "CC1", "CC2", "CC3", "CC4", "CC5", "CCX",
]
EXPERIMENTS = ["exp1", "exp3"]

# Fixed all-on evaluation profile. Overlaid on each subject's input+mutation.
# GA search budget and stopping are part of the profile (bounded, adaptive).
ALL_ON = {
    "ga": {
        "population_size": 50,
        "generations": 100,
        "crossover_rate": 0.95,
        "mutation_rate": 0.9,
        "target_sats": 1000,
        "stopping": {
            "mode": "cv_pr",
            "pr_threshold": 0.995,
            "check_every_generations": 1,
            "patience": 2,
            "min_samples": 50,
            "max_samples": 1000,
        },
    },
    "evaluation": {
        "trace_check_timeout_sec": 3600,
        "cache_enabled": True,
    },
    "heuristics": {
        "interval_inference": {
            "enabled": True,
            "mode": "label",
            "empirical_validation_k": 2,
        },
        "two_tier_timeout": {
            "enabled": True,
            "low_sec": 60,
            "high_sec": 3000,   # <= evaluation.trace_check_timeout_sec (3600)
            "escalation": "once_per_formula",
        },
        "adaptive_range": {
            "enabled": True,
            "exploration_fraction": 0.3,
            "endpoint_init": True,
            "on_one_class": "continue",
            "widen_factor": 1.5,
            "max_widenings": 4,
        },
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_config(subject_cfg: dict, seed: int, out_dir: str) -> dict:
    cfg = {"input": copy.deepcopy(subject_cfg["input"])}
    cfg["input"]["output_dir"] = out_dir

    cfg["ga"] = copy.deepcopy(ALL_ON["ga"])
    cfg["ga"]["seed"] = seed

    cfg["mutation"] = copy.deepcopy(subject_cfg["mutation"])
    cfg["evaluation"] = copy.deepcopy(ALL_ON["evaluation"])
    cfg["heuristics"] = copy.deepcopy(ALL_ON["heuristics"])
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0",
                    help="comma-separated seed list, e.g. '0,1,2' (default '0')")
    ap.add_argument("--subjects", default="all",
                    help="'all' or comma-separated subject list")
    ap.add_argument("--exps", default=",".join(EXPERIMENTS),
                    help="comma-separated experiment list (default 'exp1,exp3')")
    ap.add_argument("--out", default="configs",
                    help="output directory for generated configs")
    ap.add_argument("--output-root", default="outputs",
                    help="root the run output_dir is placed under")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    subjects = SUBJECTS if args.subjects == "all" else args.subjects.split(",")
    exps = args.exps.split(",")

    root = repo_root()
    inputs = root / "replication" / "evaluation_inputs" / "effectiveness"
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = []
    for subject in subjects:
        for exp in exps:
            src = inputs / subject / f"{exp}.json"
            if not src.exists():
                skipped.append(f"{subject}/{exp}")
                continue
            subject_cfg = json.loads(src.read_text(encoding="utf-8"))
            for seed in seeds:
                name = f"{subject}_{exp}_s{seed}"
                out_dir = f"{args.output_root}/{name}"
                cfg = build_config(subject_cfg, seed, out_dir)
                (out / f"{name}.json").write_text(
                    json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                written += 1

    print(f"wrote {written} configs to {out}")
    if skipped:
        print(f"skipped (no input file): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
