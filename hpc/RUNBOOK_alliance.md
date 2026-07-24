# Runbook — Diagnosis batch on Alliance Canada (Apptainer + SLURM)

One **joined** campaign: every run produces both the **effectiveness** artifacts (J48
tree, ARFF, `report.json`) and the **efficiency/timing** provenance. To keep timing
comparable, **every task always runs on a single, pinned node model** (`--constraint`).
We execute in **two tiers**:

- **Tier 1 — smoke + timing** (`NSEEDS=1`): one run per config. Confirms the pipeline
  works end-to-end and measures wall time per experiment.
- **Tier 2 — statistics** (`NSEEDS=10`): only after Tier 1 finishes comfortably within
  the wall. Same configs, same node model, 10 seeds for the effectiveness statistics.

No Docker on Alliance clusters; Apptainer is the runtime. Compute nodes have **no
internet** — pip packages, Weka jars and z3 are all baked into the SIF at build time.

The batch is `configs/cc_batch/` (33 experiments: `exp1..exp32` for the AT/CC families +
`exp33_RR` for the running example; see `configs/cc_batch/PROVENANCE.csv`). Every config
carries the fixed feature profile — **ACTIVE** cache + two-tier timeout + `cv_pr`
stopping + time quantization, engine `worker`, `parallel_workers=1`; **INACTIVE** interval
inference + adaptive range. `parallel_workers=1` (deterministic single-core timing) is
well within `--cpus-per-task=8`. RTAMT is NOT installed (separate study).

---

## 0. One-time setup (login node)

```bash
ssh <user>@narval.alliancecan.ca               # or beluga/graham per your allocation
git clone <your-repo-url> ~/Diagnosis          # the diagnosis-experiments repo (stable branch)
mkdir -p ~/scratch/apptainer-cache ~/scratch/apptainer-tmp ~/scratch/diagnosis-runs
mkdir -p ~/batch/logs
```

`~/Diagnosis` is the repo root: it holds `pyproject.toml`, the `diagnosis/` package,
`configs/`, `replication/`, and `hpc/`. It gets bound to `/opt/run/repo` at run time, so
manifest paths (relative to the repo root) resolve inside the container.

## 1. Build the container (login node — builds need network)

`python:3.13-slim` based to match the local venv EXACTLY (Python 3.13, z3-solver
4.15.4.0, numpy 2.4.2, scipy 1.17.1, matplotlib 3.10.8, tqdm 4.67.3 — pinned in
`hpc/requirements-lock.txt`).

```bash
cd ~/Diagnosis                                  # build from the repo root (%files are relative to it)
module load apptainer
export APPTAINER_CACHEDIR=~/scratch/apptainer-cache
export APPTAINER_TMPDIR=~/scratch/apptainer-tmp
apptainer build ~/batch/diagnosis.sif hpc/diagnosis.def
apptainer test  ~/batch/diagnosis.sif           # asserts py3.13 + z3 4.15.4.0 + java + CLI
```

Unprivileged builds work on Alliance login nodes (no sudo). If OOM-killed, build inside a
short `salloc`.

### Offline fallback (only if the login node firewalls PyPI)

The build **auto-detects** a vendored wheelhouse — no env var. Populate `hpc/wheels/` on
any x86_64 Linux with Python 3.13 + internet, copy it to the cluster, then build normally:

```bash
cd ~/Diagnosis/hpc && ./fetch_wheels.sh         # fills hpc/wheels/ with the exact pinned *.whl
cd ~/Diagnosis && apptainer build ~/batch/diagnosis.sif hpc/diagnosis.def   # installs from wheels/
```

If `hpc/wheels/` holds no `*.whl`, the build pulls the pinned wheels from pypi.org instead.

## 2. Preflight (prove the container + pipeline before any array)

```bash
cd ~/Diagnosis
# 2a. Every manifest path resolves inside the bound repo:
apptainer exec -C -B ~/Diagnosis:/opt/run/repo ~/batch/diagnosis.sif \
  bash -c 'cd /opt/run/repo && while read c; do test -f "$c" || echo "MISSING $c"; done < configs.manifest; echo ok'

# 2b. Reduced canary (~2 min, pop=10/gen=2) — asserts a J48 tree is produced:
scripts/preflight.sh ~/batch/diagnosis.sif configs/cc_batch/exp1_AT1.json ~/Diagnosis
```

`preflight.sh` PASS prints the J48 tree from the reduced run. Do not submit an array until
both pass.

## 3. Run the campaign (single node, two tiers)

Pick ONE node model and use it for **both** tiers (e.g. narval: `--constraint=milan`);
record it — timing is only comparable within one model. Each task writes
`node_info.txt`; the model reported there goes in the paper's setup section.

```bash
cd ~/batch                                       # holds diagnosis.sif, logs/
cp ~/Diagnosis/configs.manifest .
export REPO=~/Diagnosis SIF=~/batch/diagnosis.sif
wc -l configs.manifest                           # 33
```

### Tier 1 — smoke + timing (single seed → 33 tasks)

```bash
export NSEEDS=1
sbatch --constraint=<nodetype> --array=0-32%50 ~/Diagnosis/hpc/submit_batch.sh
```

Check the wall each task actually used before committing to Tier 2:

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,CPUTime
```

If the slowest experiments approach the 3 h wall, resubmit just those with a longer limit
(0-indexed manifest positions, still single seed):

| family | exp_ids | manifest indices | suggested --time |
|--------|---------|------------------|------------------|
| CC1    | 21, 22  | 20, 21           | 12:00:00 |
| CC4    | 27, 28  | 26, 27           | 12:00:00 |
| CCx    | 31, 32  | 30, 31           | 12:00:00 |

```bash
sbatch --time=12:00:00 --constraint=<nodetype> --array=20,21,26,27,30,31 ~/Diagnosis/hpc/submit_batch.sh
```

### Tier 2 — statistics (10 seeds → 330 tasks)

Only after Tier 1 fits the wall comfortably, on the **same** node model:

```bash
export NSEEDS=10                                 # 33 configs × 10 seeds = 330 tasks
sbatch --constraint=<nodetype> --array=0-329%50 ~/Diagnosis/hpc/submit_batch.sh
```

Task→(config,seed): `idx % 33 = config`, `idx / 33 = seed`; so (config `c`, seed `s`) is
array index `s*33 + c`. `%50` caps concurrency at 50 — be a good citizen.

## 4. Monitor / collect

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,CPUTime   # per-task resources → paper table
seff <jobid_taskid>                                            # efficiency per task
rsync -av --exclude '*.sqlite' narval:~/scratch/diagnosis-runs/ ./runs/   # results off-cluster
python3 scripts/aggregate_trees.py runs/                        # aggregate effectiveness
```

## 5. Gotchas

- **Always pass `--constraint=<nodetype>`** — both tiers, same model. Timing from mixed
  node types is not comparable and must not be pooled.
- **Export `REPO` and `SIF`** before `sbatch` (the script FATALs early if either is
  missing) — `REPO=~/Diagnosis`, `SIF=~/batch/diagnosis.sif`.
- **No internet on compute nodes**: any `pip install`/`wget`/registry pull inside a job
  fails. Missing a dep ⇒ rebuild the SIF on the login node.
- **`-C` (containall)** keeps host env/dotfiles out of the container; all data flows
  through the explicit `-B` binds only.
- **Wall guard**: `submit_batch.sh` derives the deadline from `scontrol` and reserves
  ~10 min so the run finalizes ARFF + J48 + summary. A task killed at the wall resumes
  cheaply from the verdict cache (SQLite in the run dir — keep it for re-runs; only
  exclude it from the final rsync).
- **Scratch is purged** (~60 days untouched) — rsync results off promptly; use `/project`
  for anything to keep.
- **Java/Weka**: baked into the SIF (`WEKA_JAR` in `%environment`); no `module load java`.
- **Account string**: `--account=def-<PI>` (set in `submit_batch.sh`); `sshare -U $USER`
  if unsure which.
- Wiki refs render fine in a browser: docs.alliancecan.ca/wiki/Apptainer and
  /wiki/Running_jobs.
