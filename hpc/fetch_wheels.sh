#!/bin/bash
# Pre-download the exact pinned wheels into hpc/wheels/ for a fully OFFLINE Apptainer build.
# Use this ONLY if the login-node pip route in diagnosis.def is blocked by the Alliance
# environment. Run on ANY x86_64 Linux with internet and Python 3.13 (e.g. a login node
# with `module load python/3.13`, or your laptop in a py3.13 venv). The wheels are
# platform+python specific — match the target: x86_64 (Narval/Beluga/Graham are all x86_64),
# CPython 3.13.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p wheels

# Force the public index; ignore any Alliance wheelhouse redirection.
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE PIP_FIND_LINKS || true

python3 -m pip download \
  --only-binary=:all: \
  --python-version 313 \
  --implementation cp \
  --platform manylinux2014_x86_64 \
  --platform manylinux_2_28_x86_64 \
  --index-url https://pypi.org/simple/ \
  -r requirements-lock.txt \
  -d wheels

echo "Vendored wheels:"
ls -1 wheels
echo
echo "wheels/ now populated — the build auto-detects them and installs fully offline:"
echo "  apptainer build diagnosis.sif hpc/diagnosis.def   (run from the repo root)"
