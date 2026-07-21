# Runbook — Diagnosis batches on Alliance Canada (Apptainer + SLURM)

Target: full effectiveness + efficiency campaigns (32 configs × N seeds). No Docker on Alliance
clusters; Apptainer is the supported runtime. Compute nodes have **no internet** on most
clusters — everything (pip packages, Weka jars, z3) is baked into the SIF at build time.

## 0. One-time setup (login node)

```bash
ssh <user>@narval.alliancecan.ca          # or beluga/graham per your allocation
git clone <your-repo-url> ~/Diagnosis     # the STABLE branch, not spike branches
mkdir -p ~/scratch/apptainer-cache ~/scratch/apptainer-tmp ~/scratch/diagnosis-runs
```

Copy `hpc/` (this folder) next to the clone, adjust the `%files` path in `diagnosis.def`
so it points at the clone.

## 1. Build the container (login node — builds need network)

The container is `python:3.13-slim` based to match the local venv EXACTLY (Python 3.13,
z3-solver 4.15.4.0, numpy 2.4.2, scipy 1.17.1, matplotlib 3.10.8, tqdm 4.67.3 — pinned in
`requirements-lock.txt`). RTAMT is NOT installed (separate study).

```bash
module load apptainer
export APPTAINER_CACHEDIR=~/scratch/apptainer-cache
export APPTAINER_TMPDIR=~/scratch/apptainer-tmp
apptainer build diagnosis.sif hpc/diagnosis.def
apptainer test diagnosis.sif              # %test asserts py3.13 + z3 4.15.4.0 + java + CLI
```

### Why the old `pip install z3-solver` failed, and how this def fixes it

The Alliance software stack sets `PIP_INDEX_URL` / `PIP_CONFIG_FILE` pointing at their in-house
wheelhouse, which does not carry `z3-solver==4.15.4.0`. Those variables leak into the
`apptainer build` %post step, so pip tries the wheelhouse and fails. The def now **unsets all
PIP_* redirection and forces `--index-url https://pypi.org/simple/ --only-binary=:all:`**, so the
exact z3 wheel from PyPI is installed regardless of the host environment. `pip install -e .` runs
with `--no-deps` so it can never re-resolve to different versions.

### Offline fallback (if the login node blocks PyPI entirely)

Some clusters firewall outbound PyPI even on login nodes. Then vendor the wheels first:

```bash
# On any x86_64 Linux with Python 3.13 + internet (login node w/ `module load python/3.13`,
# or your laptop), from the hpc/ dir:
./fetch_wheels.sh                          # populates hpc/wheels/ with the exact pinned wheels
# Copy hpc/ (now including wheels/) to the cluster, then:
OFFLINE=1 apptainer build diagnosis.sif hpc/diagnosis.def
```

Notes: unprivileged builds work on Alliance login nodes (no sudo). If the build is OOM-killed,
raise the login-shell memory or build in a short `salloc`.

## 2. Canary (interactive, before any array)

```bash
salloc --account=def-CHANGEME --cpus-per-task=8 --mem=16G --time=00:30:00
module load apptainer
apptainer exec -C -B ~/Diagnosis:/opt/run/repo -B ~/scratch/diagnosis-runs/canary:/opt/run/out \
  -B $SLURM_TMPDIR:/tmp --pwd /opt/run/repo diagnosis.sif \
  diagnosis run --config configs/at_batch_all_on/AT1_exp1_all_on_cv_pr.json
```

Verify: run completes, tree produced, outputs land on scratch, `report.json` sane.

## 3. Submit the campaigns

```bash
cd ~/batch && ls
#  diagnosis.sif  submit_batch.sh  configs.manifest  logs/
wc -l configs.manifest                    # 32 configs
# Effectiveness / statistics campaign (10 seeds => 320 tasks):
export NSEEDS=10
sbatch --array=0-319%50 submit_batch.sh   # %50 = max 50 concurrent, be a good citizen
# CC-heavy subset with a longer limit:
sbatch --time=12:00:00 --array=<cc-indices> submit_batch.sh
```

Task→(config,seed) mapping: `idx % n_configs` = config, `idx / n_configs` = seed.

**Efficiency (timing) campaign — separate submission:** timing numbers must come from ONE
node model. Pin it and disable seeds:
```bash
export NSEEDS=1
sbatch --constraint=<nodetype> --array=0-31 submit_batch.sh   # e.g. narval: --constraint=milan
```
Record the node model (each run writes `node_info.txt`); report it in the paper's setup
section. Do NOT mix timing from different node types or from the statistics campaign.

## 4. Monitor / collect

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,CPUTime   # per-task resources → paper table
seff <jobid_taskid>                                            # efficiency per task
# Collect to your machine:
rsync -av --exclude '*.sqlite' narval:~/scratch/diagnosis-runs/ ./runs/
# Aggregate (reuse the tree-statistics script from the experiments repo):
python3 scripts/aggregate_trees.py runs/
```

## 5. Gotchas

- **No internet on compute nodes**: any `pip install`, `wget`, or registry pull inside a job
  fails. If a dependency is missing, rebuild the SIF on the login node.
- **`-C` (containall)** keeps host env/dotfiles out of the container — reproducibility. All
  data flows through the explicit `-B` binds only.
- **Scratch is purged** (typically files untouched ~60 days) — rsync results off promptly;
  `/project` for anything to keep.
- **Wall guard**: submit_batch.sh reserves ~10 min before SLURM's limit so the run finalizes
  ARFF + J48 + summary; a killed task resumes cheaply via the verdict cache (SQLite in the
  run dir — do not exclude it from re-runs, only from the final rsync).
- **Java/Weka**: baked into the SIF (`WEKA_JAR` set in `%environment`); no `module load java`.
- **Parallel workers**: the tool reads `SLURM_CPUS_PER_TASK`-sized allocation; keep
  `parallel_workers <= cpus-per-task` in the configs (batch profile uses 8).
- **Account string**: `--account=def-<PI>`; check `sshare -U $USER` if unsure which.
- If the wiki pages block scripted access, they render fine in a browser:
  docs.alliancecan.ca/wiki/Apptainer and /wiki/Running_jobs.
