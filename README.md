# diagnosis-experiments

Companion repository for running the evaluation batch of the paper
**"Search-based Trace Diagnostic for Cyber-Physical Systems"** on GitHub Actions.

It bundles the `diagnosis` package (the tool under evaluation), the effectiveness
benchmark inputs, and the scripts and workflows that generate the batch configs,
run them, and aggregate the resulting decision-tree statistics.

> **Timing is not a paper result.** The wall-clock numbers reported in the paper
> come from the controlled local environment described there, not from CI. CI
> runs on shared hosted runners of varying speed; each `report.json` records the
> runner hardware for transparency only. CI is here to reproduce the *tree
> statistics* (splits, thresholds, boundaries, precision/recall), not timings.

## Layout

```
diagnosis/                     the tool (installed with `pip install -e .`)
replication/
  evaluation_inputs/           requirement scripts + per-experiment configs
  analysis/data/               execution traces (tracesAT/CC, traceRR)
scripts/
  gen_batch_configs.py         emit the 32 experiment configs x a seed list
  run_batch.py                 run configs under a wall guard; write run_meta
  aggregate_trees.py           reduce collected artifacts into stat tables
.github/workflows/
  batch.yml                    dispatch the batch (matrix = subject x seed)
  aggregate.yml                aggregate a finished batch into CSVs + summary
  smoke.yml                    unit tests + one short end-to-end on push/PR
```

## Experiments

The batch covers 16 subjects — `AT1, AT2, AT51–AT54, AT6A/B/C, AT6ABC,
CC1–CC5, CCX` — each with two experiments (`exp1`, `exp3`), i.e. **32 configs**
per seed. Each config takes its requirement, traces, and mutation block from
`replication/evaluation_inputs/effectiveness/<subject>/<exp>.json` and overlays
the fixed **all-on profile** below.

### All-on profile

Emitted by `scripts/gen_batch_configs.py` (see the `ALL_ON` constant). Every
config carries:

| Block | Setting |
| --- | --- |
| `ga` | population 50, generations 100, crossover 0.95, mutation 0.9, target_sats 1000 |
| `ga.stopping` | `cv_pr`, pr_threshold 0.995, check_every 1, patience 2, min_samples 50, max_samples 1000 |
| `evaluation` | trace_check_timeout_sec 3600, cache_enabled true; `parallel_workers` set at run time to `min(4, cpu_count)` |
| `heuristics.interval_inference` | enabled, mode `label`, empirical_validation_k 2 |
| `heuristics.two_tier_timeout` | enabled, low 60 s, high 600 s, escalate once per formula |
| `heuristics.adaptive_range` | enabled, exploration_fraction 0.3, endpoint_init, on_one_class `continue`, widen 1.5, max_widenings 4 |

`time_quantization` is left off, matching the local batch. The seed is the only
per-run variable and comes solely from the config, so results are deterministic
given a config.

## Running locally

```bash
pip install -e .
export WEKA_JAR=/path/to/weka-stable-3.8.6.jar          # Weka 3.8.6 + a JRE

python scripts/gen_batch_configs.py --seeds 0 --subjects all --out configs
python scripts/run_batch.py --configs-dir configs --wall-minutes 330
python scripts/aggregate_trees.py --runs-dir outputs --out stats
```

## Running in CI

See [RUNBOOK.md](RUNBOOK.md) for the full push-and-dispatch procedure. In short:
dispatch **batch** (`workflow_dispatch`) with a seed list and subject set, then
dispatch **aggregate** with the batch run id (or let it trigger automatically).
Artifacts (`diagnosis-run-<subject>-s<seed>`, `diagnosis-statistics`) are kept
for 30 days; pull them with `gh run download`.

## License

The tool is released under the MIT license (see `LICENSE`). Weka (GPL) is not
redistributed here; CI downloads `weka-stable-3.8.6.jar` from Maven Central,
pinned by version and sha256.
