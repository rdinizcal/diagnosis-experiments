# RUNBOOK — diagnosis-experiments

Procedure for the repository owner to publish this repository and drive the
evaluation batch on GitHub Actions. Nothing here has been pushed yet.

## 4. Full batch

```bash
gh workflow run batch.yml -f seeds=0 -f subjects=all -f wall_minutes=340
# multiple seeds:
gh workflow run batch.yml -f seeds=0,1,2,3,4 -f subjects=all -f wall_minutes=340
```

Matrix = **subject x exp x seed** (one experiment per cell), `max-parallel: 20`,
`timeout-minutes: 355` (under the 6-hour hosted-runner cap). Each cell runs a
single experiment under the internal wall guard, so every experiment gets the
**full** `wall_minutes` budget (not half of it shared with the other exp).

## 4b. Re-run only the experiments that did not finish

Some experiments (esp. the solve-heavy CC family) can exceed the runner cap.
To find and re-run exactly those:

```bash
# 1. Download the batch artifacts (see step 6), then classify them:
python scripts/find_unfinished.py --runs-dir online_results --seeds 0,1,2,3,4
#   -> prints a `cells=SUBJECT:exp:seed,...` line for MISSING/WALL/crashed cells,
#      and separately lists degenerate one-class cells (a config fix, not a re-run).

# 2. Re-dispatch just those cells with the full per-experiment budget:
gh workflow run batch.yml -f cells="CC1:exp1:0,CC1:exp3:0,CC2:exp1:0,..." -f wall_minutes=340
```

`cells` overrides subjects/seeds/exps. Each listed experiment runs in its own
cell with up to `wall_minutes` (≤355) of solver time. If an experiment still
does not finish at `wall_minutes=340`, it needs more time than a hosted runner
allows — run that one locally with no wall guard (README "Running locally",
`run_batch.py --wall-minutes 0`). Re-runs supersede earlier ones: the analysis
scripts keep only the newest run per (subject, exp, seed).

## 5. Aggregate

```bash
BATCH_ID=$(gh run list --workflow=batch.yml -L1 --json databaseId -q '.[0].databaseId')
gh workflow run aggregate.yml -f run_id=$BATCH_ID
```
`aggregate.yml` also triggers automatically when a batch run completes. It writes
the summary to the job summary and uploads the `diagnosis-statistics` artifact
(CSV per table + `SUMMARY.md`).

## 6. Pull results locally

```bash
gh run download <batch_run_id>       # all diagnosis-run-* artifacts
gh run download <aggregate_run_id>   # diagnosis-statistics
```

## Notes

- **Determinism.** The seed comes only from the config. `batch.yml` runs with
  `--workers 1` (serial), so nothing varies with the runner; worker count does
  not affect results in any case (batch results are applied in population-index
  order). Runner hardware is recorded in each `report.json` under `runner`.
- **Artifacts.** Each `diagnosis-run-*` contains `report.json`, `summary.json`,
  `run_meta.json`, final ARFFs, J48 `.out`, stopping/`hypot` logs, and
  `inference_state.json`. Bulky per-generation population dumps are excluded.
  Retention: 30 days.
- **Resume via the verdict cache.** `batch.yml` passes `--cache-dir vcache`, which
  pins the verdict cache to a stable file (`evaluation.cache_path`) that
  `actions/cache` restores/saves per cell (run-id key + prefix restore-keys, so
  each dispatch saves an updated cache). Re-dispatching a wall-stopped experiment
  replays already-solved candidates from cache and advances further each time
  until it completes. Aggregation keeps only the newest run per (subject, exp,
  seed), so a completed re-run supersedes the wall-stopped one. Note: a resumed
  run reuses cached *verdicts* (each is the verdict a solve would return); for a
  strictly single-shot reproduction of a seed, run it locally with
  `run_batch.py --wall-minutes 0` and no `--cache-dir`.
- **Weka needs bounce.jar.** The Maven `weka-stable-3.8.6.jar` does not bundle
  `org.bounce`, and Weka's package manager references it, so J48 dies with
  `NoClassDefFoundError` and produces no tree unless `third_party/bounce.jar`
  (vendored) is on `WEKA_JAR`. The workflows add it automatically; keep it when
  editing the Weka step. See `third_party/README.md`.
- **Weka checksum.** To refresh:
  ```bash
  curl -fsSL -o weka.jar \
    https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar
  sha256sum weka.jar
  ```
- **Solve-heavy subjects (CC).** CC requirements can need the 600 s high tier
  even locally; on slower runners those SAT solves time out to UNDECIDED, so a
  CC cell may show 0 SAT. Raise `two_tier_timeout.high_sec` (<= trace timeout)
  in `scripts/gen_batch_configs.py` for CI, or run CC locally with no wall guard.
- **Timing.** No timing claim in the paper is derived from CI.
