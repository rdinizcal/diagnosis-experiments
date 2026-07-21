#!/bin/bash
# Preflight canary: given the SIF and ONE cc_batch config, run a ~2-minute reduced
# job through Apptainer and assert a J48 tree is produced. Scripts step 2 of the
# runbook so the container + pipeline are proven before submitting an array.
#
# Usage:
#   scripts/preflight.sh <diagnosis.sif> <config-path-relative-to-repo> [repo_dir]
# e.g.
#   scripts/preflight.sh diagnosis.sif configs/cc_batch/exp1_AT1.json ~/Diagnosis
set -Eeuo pipefail

SIF=${1:?usage: preflight.sh <sif> <config> [repo]}
CONFIG=${2:?usage: preflight.sh <sif> <config> [repo]}
REPO=${3:-$PWD}
OUT=$(mktemp -d "${TMPDIR:-/tmp}/preflight.XXXXXX")
TMP=$(mktemp -d "${TMPDIR:-/tmp}/preflight-tmp.XXXXXX")
trap 'rm -rf "$OUT" "$TMP"' EXIT

command -v apptainer >/dev/null 2>&1 || module load apptainer 2>/dev/null || true

# Reduced copy of the config: tiny population/generations so a full run (incl. J48)
# finishes inside ~2 minutes without changing engine/heuristic wiring.
apptainer exec -C \
  -B "$REPO":/opt/run/repo \
  -B "$OUT":/opt/run/out \
  -B "$TMP":/tmp \
  "$SIF" python3 - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(f"/opt/run/repo/{sys.argv[1]}"))
ga = cfg.setdefault("ga", {})
ga["population_size"] = 10
ga["generations"] = 2
ga.setdefault("stopping", {}).update({"min_samples": 1, "max_samples": 50})
ga["seed"] = 0
cfg["input"]["output_dir"] = "/opt/run/out"
json.dump(cfg, open("/opt/run/out/preflight_config.json", "w"), indent=1)
print("reduced config written")
PY

echo "[preflight] running reduced canary (<=120s) for $CONFIG"
apptainer exec -C \
  -B "$REPO":/opt/run/repo \
  -B "$OUT":/opt/run/out \
  -B "$TMP":/tmp \
  --pwd /opt/run/repo \
  "$SIF" timeout 120 diagnosis run --config /opt/run/out/preflight_config.json \
  || echo "[preflight] run exited $? (timeout/error) — checking artifacts anyway"

# Success = at least one J48 tree file with a real split (root node) was produced.
if grep -Rslq "J48 pruned tree" "$OUT"/*.out 2>/dev/null; then
  echo "[preflight] PASS — J48 tree produced:"
  grep -RA6 "J48 pruned tree" "$OUT"/*.out 2>/dev/null | head -10
  exit 0
fi
echo "[preflight] FAIL — no J48 tree found under $OUT" >&2
ls -R "$OUT" >&2 || true
exit 1
