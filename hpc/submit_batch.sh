#!/bin/bash
# SLURM array job: one task = one (config, seed) pair, run inside Apptainer.
# Usage (from the directory containing diagnosis.sif and configs.manifest):
#   sbatch --array=0-$(( $(wc -l < configs.manifest) * ${NSEEDS:-1} - 1 )) submit_batch.sh
#
# configs.manifest: one config path per line (relative to the bound repo root), e.g.
#   configs/at_batch_all_on/AT1_exp1_all_on_cv_pr.json
#   ...
# Efficiency runs: submit a SECOND, dedicated batch with --constraint to pin one node
# type and NSEEDS=1 — timing numbers must come from a single, described node model.

#SBATCH --account=def-CHANGEME          # your PI's allocation
#SBATCH --job-name=diagnosis-batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00                 # AT family fits easily; use 12:00:00 for CC subsets
#SBATCH --output=logs/%x_%A_%a.out

set -Eeuo pipefail
module load apptainer

: "${NSEEDS:=1}"                        # export NSEEDS=10 for the statistics campaign
: "${SIF:=$PWD/diagnosis.sif}"
: "${REPO:=$PWD/Diagnosis}"             # working copy with configs/ (bound read-mostly)
: "${OUTROOT:=$SCRATCH/diagnosis-runs}" # all outputs on scratch

mapfile -t CONFIGS < configs.manifest
NCFG=${#CONFIGS[@]}
CFG_IDX=$(( SLURM_ARRAY_TASK_ID % NCFG ))
SEED=$(( SLURM_ARRAY_TASK_ID / NCFG ))
CONFIG=${CONFIGS[$CFG_IDX]}
NAME=$(basename "$CONFIG" .json)_s${SEED}

RUNDIR="$OUTROOT/$NAME"
mkdir -p "$RUNDIR" logs

# Patch seed + output dir into a per-task copy of the config (jq ships in the SIF via python;
# use python to stay dependency-free on the host)
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

# Run: workers sized to the allocation; wall guard inside SLURM's limit so the run
# finalizes datasets + J48 + summary before SLURM kills the task.
apptainer exec -C \
  -B "$REPO":/opt/run/repo \
  -B "$RUNDIR":/opt/run/out \
  -B "$SLURM_TMPDIR":/tmp \
  --pwd /opt/run/repo \
  "$SIF" timeout $(( (SLURM_JOB_END_TIME - $(date +%s)) - 600 )) \
    diagnosis run --config /opt/run/out/config.json \
  || echo "RUN ${NAME} exited $? (timeout or error) — state in $RUNDIR is resumable via cache"

# Record provenance for the paper's resource table
{ echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo; \
  echo "cpus: $SLURM_CPUS_PER_TASK  mem: $SLURM_MEM_PER_NODE"; } > "$RUNDIR/node_info.txt"
