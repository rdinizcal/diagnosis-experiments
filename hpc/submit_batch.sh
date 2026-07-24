#!/bin/bash
# SLURM array job: one task = one (config, seed) pair, run inside Apptainer.
# JOINED campaign: every run produces BOTH the effectiveness artifacts (tree, ARFF,
# report.json) AND the timing/efficiency provenance (node_info.txt + sacct/seff). To
# keep timing comparable, ALWAYS pin a single node model with --constraint so all tasks
# land on identical hardware.
#
# Two tiers (see RUNBOOK_alliance.md §3):
#   Tier 1 — smoke/timing : export NSEEDS=1 ; one run per config, check it works + wall.
#   Tier 2 — statistics   : export NSEEDS=10; only after Tier 1 fits the wall comfortably.
#
# Usage (from the dir holding diagnosis.sif + configs.manifest, with REPO exported):
#   export REPO=~/Diagnosis NSEEDS=1
#   sbatch --constraint=<nodetype> \
#          --array=0-$(( $(wc -l < configs.manifest) * NSEEDS - 1 ))%50 hpc/submit_batch.sh
#
# configs.manifest: one config path per line, relative to the bound repo root, e.g.
#   configs/cc_batch/exp1_AT1.json
# Task->(config,seed): idx % n_configs = config, idx / n_configs = seed.

#SBATCH --account=def-CHANGEME          # your PI's allocation (sshare -U $USER if unsure)
#SBATCH --job-name=diagnosis-batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00                 # AT family fits; use --time=12:00:00 for CC subsets
#SBATCH --output=logs/%x_%A_%a.out
# NOTE: pass --constraint=<nodetype> on the sbatch command line (single node model).

set -Eeuo pipefail
module load apptainer

: "${NSEEDS:=1}"                        # export NSEEDS=10 for the statistics tier
: "${SIF:=$PWD/diagnosis.sif}"
: "${REPO:=$HOME/Diagnosis}"           # the diagnosis-experiments clone (bound read-mostly)
: "${OUTROOT:=$SCRATCH/diagnosis-runs}" # all outputs on scratch

[ -f "$SIF" ]  || { echo "FATAL: SIF not found: $SIF" >&2; exit 1; }
[ -d "$REPO" ] || { echo "FATAL: REPO not found: $REPO (export REPO=~/Diagnosis)" >&2; exit 1; }
[ -f configs.manifest ] || { echo "FATAL: configs.manifest not in $PWD" >&2; exit 1; }

mapfile -t CONFIGS < configs.manifest
NCFG=${#CONFIGS[@]}
CFG_IDX=$(( SLURM_ARRAY_TASK_ID % NCFG ))
SEED=$(( SLURM_ARRAY_TASK_ID / NCFG ))
CONFIG=${CONFIGS[$CFG_IDX]}
NAME=$(basename "$CONFIG" .json)_s${SEED}

RUNDIR="$OUTROOT/$NAME"
mkdir -p "$RUNDIR" logs

# Wall guard: finalize ARFF + J48 + summary ~10 min before SLURM kills the task. Derive
# the real deadline from scontrol (SLURM does NOT export a job-end-time env var); fall
# back to the requested TimeLimit, then to a safe default. A killed task resumes cheaply
# from the verdict cache on resubmit.
END_EPOCH=0
if command -v scontrol >/dev/null 2>&1; then
  END_STR=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)
  [ -n "${END_STR:-}" ] && END_EPOCH=$(date -d "$END_STR" +%s 2>/dev/null || echo 0)
fi
if [ "$END_EPOCH" -gt 0 ]; then
  GUARD=$(( END_EPOCH - $(date +%s) - 600 ))
else
  GUARD=$(( 3 * 3600 - 600 ))          # fallback: default 3 h wall minus 10 min
fi
[ "$GUARD" -lt 300 ] && GUARD=300      # never hand timeout a tiny/negative value
echo "[submit] task=$SLURM_ARRAY_TASK_ID config=$CONFIG seed=$SEED wall-guard=${GUARD}s node=$(hostname)"

# Patch seed + output dir into a per-task copy of the config (python inside the SIF,
# so the host stays dependency-free).
apptainer exec -C \
  -B "$REPO":/opt/run/repo \
  -B "$RUNDIR":/opt/run/out \
  -B "$SLURM_TMPDIR":/tmp \
  "$SIF" python3 - "$CONFIG" "$SEED" <<'PY'
import json, sys
cfg_path, seed = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(f"/opt/run/repo/{cfg_path}"))
cfg.setdefault("ga", {})["seed"] = seed
cfg["input"]["output_dir"] = "/opt/run/out"
json.dump(cfg, open("/opt/run/out/config.json", "w"), indent=1)
PY

# Record node/resource provenance FIRST so timing attribution survives a wall kill.
{ echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo; \
  echo "constraint: ${SLURM_JOB_CONSTRAINTS:-<none>}"; \
  echo "cpus: ${SLURM_CPUS_PER_TASK:-?}  mem: ${SLURM_MEM_PER_NODE:-?}"; \
  echo "started: $(date -Is)"; } > "$RUNDIR/node_info.txt"

# Run: workers sized to the allocation; timeout enforces the wall guard.
apptainer exec -C \
  -B "$REPO":/opt/run/repo \
  -B "$RUNDIR":/opt/run/out \
  -B "$SLURM_TMPDIR":/tmp \
  --pwd /opt/run/repo \
  "$SIF" timeout "$GUARD" \
    diagnosis run --config /opt/run/out/config.json \
  || echo "RUN ${NAME} exited $? (timeout or error) — state in $RUNDIR is resumable via cache"

echo "finished: $(date -Is)" >> "$RUNDIR/node_info.txt"
