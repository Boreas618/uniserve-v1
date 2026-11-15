#!/usr/bin/env bash
# Runs the GPU end-to-end tests (`tests/python/e2e`) from one build of the tree.
#
# Usage, from any directory:
#
#   tools/gpu_e2e.sh [--memcheck] [pytest arguments...]
#
# The tests run the `uniserve` executable and the `uniserve-manager` the environment of `PYTHON`
# installs with the root project (`uv sync`, a release build), which is a test deployment's daemon
# and its host services. A deployment's main
# processes run the engine package (`crates/engine`), which the run's Build Server
# installs and builds with cargo from the installed snapshot of the workspace: that build finds
# FFmpeg 8.1 through pkg-config (`crates/media/README.md`) and reads `LIBCLANG_PATH` from this
# environment, and cargo keeps what it compiled between runs. Each Build Server compiles kernels
# with its toolchain environment, whose program `uniserve-build-server` must be on the search
# path. The serving processes open the driver, NCCL and CUPTI from the host's default loader path
# and refuse a release older than the one they require. The script runs pytest with one worker
# per GPU (pytest-xdist) and the caller's arguments, whose marker expression (`-m "e2e and gpu"`)
# selects the GPU cases; each case leases the GPUs it serves on
# (`tests/python/e2e/conftest.py`) and serves the released checkpoints it names by repository
# and revision from the Hugging Face cache. pytest's temporary directory, which holds every
# server's log, and the JUnit report go to `artifacts/e2e/<UTC time>/` unless
# `UNISERVE_E2E_ARTIFACTS` names another directory.
#
# `--memcheck` selects the engine package `memcheck` (`tests/python/sanitizer`), which runs every
# main process under compute-sanitizer's memcheck once a declared engine launcher replaces it.
#
# `UNISERVE_E2E_DEVICES` restricts the GPUs the worker count is taken from (pass the same set as
# `--e2e-devices`), and `UNISERVE_E2E_WORKERS` states another worker count. `PYTHON` names the
# interpreter command, as in the justfile.
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
read -r -a python <<< "${PYTHON:-uv run python}"

memcheck=0
if [ "${1:-}" = --memcheck ]; then
    memcheck=1
    shift
fi

if ! command -v uniserve-build-server > /dev/null; then
    echo "no uniserve-build-server on the search path: install the Build Server's environment" \
        "(uv sync --locked --project crates/build-server/python) with its program on the path" >&2
    exit 1
fi

artifacts=${UNISERVE_E2E_ARTIFACTS:-$root/artifacts/e2e/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$artifacts"

# The deployments' engine package and the options every deployment adds, as the tests' plugin
# takes them (`tests/support/src/uniserve_testing/plugin.py`).
deployment=()
if [ "$memcheck" = 1 ]; then
    # A main process under memcheck takes more of the device and runs slower. The `memcheck`
    # package awaits a declared engine launcher (specs/ad-hoc.md section 9, TE-202/203).
    deployment=(--engine memcheck --time-scale 20
        --serve-set mem_fraction_static=0.6 --serve-set worker_config.watchdog_timeout=2h)
fi

if [ -n "${UNISERVE_E2E_DEVICES:-}" ]; then
    gpus=$(tr ',' '\n' <<< "$UNISERVE_E2E_DEVICES" | grep -c .)
else
    gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
fi
workers=${UNISERVE_E2E_WORKERS:-$gpus}

exec "${python[@]}" -m pytest -n "$workers" --dist loadgroup \
    --basetemp "$artifacts/pytest" --junitxml "$artifacts/junit.xml" "${deployment[@]}" "$@"
