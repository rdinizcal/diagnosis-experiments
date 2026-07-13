#!/usr/bin/env python3
"""Plot cross-validated precision/recall vs. generation for downloaded runs.

Discovers every run under --runs-dir (any number of batch-download folders),
keeps the newest run per (subject, exp, seed), and draws small multiples by
subject: one line per instance, blue = exp1 / vermillion = exp3, precision
solid / recall dashed, with the 0.995 cv_pr stop threshold and stop markers.

Re-run this after downloading more artifacts; new folders are picked up
automatically.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C = {"exp1": "#0072B2", "exp3": "#D55E00"}      # CVD-safe categorical pair
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

RUN_RE = re.compile(r"outputs/([A-Za-z0-9]+)_(exp\d+)_s(\d+)/([^/]+)$")


def newest_runs(runs_dir: str) -> dict:
    best = {}
    for rep in glob.glob(f"{runs_dir}/**/summary.json", recursive=True):
        m = RUN_RE.search(os.path.dirname(rep))
        if not m:
            continue
        subj, exp, seed, stamp = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        key = (subj, exp, seed)
        if key not in best or stamp > best[key][0]:
            best[key] = (stamp, rep)
    return best


def collect(best: dict):
    series, gmin = {}, 1.0
    for (subj, exp, seed), (stamp, rep) in best.items():
        try:
            d = json.load(open(rep))
        except Exception:
            continue
        sc = [c for c in (d.get("ga_stopping_checks") or [])
              if isinstance(c, dict) and c.get("cv_precision") is not None]
        if not sc:
            continue
        sc.sort(key=lambda c: c.get("generation", 0))
        gens = [c["generation"] for c in sc]
        prec = [c["cv_precision"] for c in sc]
        rec = [c["cv_recall"] for c in sc]
        stop_gen = next((c["generation"] for c in sc if c.get("stop")), gens[-1])
        gmin = min(gmin, min(prec + rec))
        series.setdefault(subj, []).append(
            dict(exp=exp, seed=seed, gens=gens, prec=prec, rec=rec, stop=stop_gen))
    return series, gmin


def plot(series: dict, gmin: float, out: str) -> None:
    subjects = sorted(series)
    n = len(subjects)
    ncol = 3
    nrow = max(1, math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.0 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    ylo = max(0.0, math.floor(gmin * 20) / 20)

    for ax, subj in zip(axes, subjects):
        ax.axhline(0.995, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
        for s in series[subj]:
            col = C.get(s["exp"], INK)
            ax.plot(s["gens"], s["prec"], color=col, lw=1.0, alpha=0.75, zorder=3)
            ax.plot(s["gens"], s["rec"], color=col, lw=1.0, alpha=0.55,
                    ls=(0, (3, 2)), zorder=2)
            if s["stop"] in s["gens"]:
                i = s["gens"].index(s["stop"])
                ax.plot(s["stop"], s["prec"][i], "o", color=col, ms=3.5,
                        mec="white", mew=0.6, zorder=4)
        ax.set_title(f"{subj}  (n={len(series[subj])})", color=INK, fontsize=11, loc="left")
        ax.set_ylim(ylo, 1.008)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.supxlabel("GA generation (cv_pr check)", color=INK, fontsize=11)
    fig.supylabel("Cross-validated score", color=INK, fontsize=11)
    handles = [
        Line2D([0], [0], color=C["exp1"], lw=2, label="exp1"),
        Line2D([0], [0], color=C["exp3"], lw=2, label="exp3"),
        Line2D([0], [0], color=INK, lw=1.3, ls="-", label="precision"),
        Line2D([0], [0], color=INK, lw=1.3, ls=(0, (3, 2)), label="recall"),
        Line2D([0], [0], color=MUTED, lw=1, ls=(0, (1, 2)), label="0.995 stop threshold"),
        Line2D([0], [0], marker="o", color="none", mec=INK, mfc=INK, ms=5, label="stop generation"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.005), labelcolor=INK)
    fig.suptitle("Cross-validated precision & recall vs. generation — all instances "
                 "(newest run per subject/exp/seed)", color=INK, fontsize=12.5, y=1.045)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=os.path.join(repo, "online_results"),
                    help="directory of downloaded artifacts (default: online_results/)")
    ap.add_argument("--out", default=os.path.join(repo, "analysis", "cv_pr_vs_generation.png"),
                    help="output PNG path")
    args = ap.parse_args()

    best = newest_runs(args.runs_dir)
    series, gmin = collect(best)
    if not series:
        print(f"[plot_cvpr] no runs with CV checks found under {args.runs_dir}")
        return
    plot(series, gmin, args.out)
    total = sum(len(v) for v in series.values())
    print(f"[plot_cvpr] wrote {args.out}")
    print(f"[plot_cvpr] {total} instances across {len(series)} subjects: {', '.join(sorted(series))}")


if __name__ == "__main__":
    main()
