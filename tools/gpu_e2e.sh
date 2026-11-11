#!/usr/bin/env bash
# Runs the GPU end-to-end tests (`tests/python/e2e`) from one build of the tree.
#
# Usage, from any directory:
#
#   tools/gpu_e2e.sh [--memcheck] [pytest arguments...]
#
# The script builds the release `uniserve` executable, which serves every process of a test
# deployment (the daemon and its workers alike), the scheduling policy precompiler and the
# `uniserve-manager` the admin API's tests attach beside the executable, and
# exports them as `UNISERVE_BINARY`, `UNISERVE_WORKER_EXECUTABLE` and
# `UNISERVE_SCHEDULING_PRECOMPILE`; when the environment names both `UNISERVE_BINARY` and
# `UNISERVE_SCHEDULING_PRECOMPILE`, as a host sharing another host's build does, it uses them
# and builds nothing. It puts the NCCL this project pins on the loader's path
# (`tools/check_nccl.sh`), sets the gate `UNISERVE_RUN_GPU_E2E=1`, and runs pytest with one
# worker per GPU (pytest-xdist); each case leases the GPUs it serves on
# (`tests/python/e2e/conftest.py`). pytest's temporary directory, which holds every server's log,
# and the JUnit report go to `artifacts/e2e/<UTC time>/` unless `UNISERVE_E2E_ARTIFACTS` names
# another directory.
#
# `--memcheck` runs every main process under compute-sanitizer's memcheck
# (`tests/python/sanitizer/memcheck-worker`) and writes its logs to `$SANITIZER_LOGS`
# (`<artifacts>/memcheck` by default).
#
# Host facts come from the environment: the checkpoints and inputs the tests name
# (docs/testing.md lists them; a case whose asset is unset is skipped, naming the variable),
# `UNISERVE_FFMPEG_DIR` for the build, `UNISERVE_E2E_DEVICES` to restrict the GPUs, and
# `UNISERVE_E2E_WORKERS` for another worker count. `PYTHON` and `CARGO` name the interpreter
# command and cargo, as in the justfile; `MAX_JOBS` caps the Build Server's kernel compilations
# (16 when unset).
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
read -r -a python <<< "${PYTHON:-uv run python}"
cargo=${CARGO:-cargo}

memcheck=0
if [ "${1:-}" = --memcheck ]; then
    memcheck=1
    shift
fi

# A run on a host that shares another host's build names that build's executables and builds
# nothing; otherwise the tree is built here.
if [ -z "${UNISERVE_BINARY:-}" ] || [ -z "${UNISERVE_SCHEDULING_PRECOMPILE:-}" ]; then
    # The manager runs beside the executable: the admin API's tests attach it to a deployment.
    "$cargo" build --release -p uniserve --bin uniserve -p scheduling --bin scheduling-precompile \
        -p manager --bin uniserve-manager
    release=${CARGO_TARGET_DIR:-$root/target}/release
    export UNISERVE_BINARY=$release/uniserve
    export UNISERVE_SCHEDULING_PRECOMPILE=$release/scheduling-precompile
fi
export UNISERVE_RUN_GPU_E2E=1
export MAX_JOBS=${MAX_JOBS:-16}

artifacts=${UNISERVE_E2E_ARTIFACTS:-$root/artifacts/e2e/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$artifacts"

if [ "$memcheck" = 1 ]; then
    export SANITIZER_LOGS=${SANITIZER_LOGS:-$artifacts/memcheck}
    mkdir -p "$SANITIZER_LOGS"
    export UNISERVE_WORKER_EXECUTABLE=$root/tests/python/sanitizer/memcheck-worker
    export UNISERVE_MEMCHECK_TARGET=$UNISERVE_BINARY
    # A main process under memcheck takes more of the device and runs slower.
    export UNISERVE_E2E_SET=${UNISERVE_E2E_SET:-"mem_fraction_static=0.6 worker_config.watchdog_timeout=2h"}
    export UNISERVE_E2E_TIME_SCALE=${UNISERVE_E2E_TIME_SCALE:-20}
else
    export UNISERVE_WORKER_EXECUTABLE=$UNISERVE_BINARY
fi

# The environment the tests run in: its NCCL, checked against the pin, is the one a serving
# process opens by name, and a main process takes CUPTI from the environment `VIRTUAL_ENV`
# states (docs/profiling.md).
prefix=$("${python[@]}" -c 'import sys; print(sys.prefix)')
export VIRTUAL_ENV=${VIRTUAL_ENV:-$prefix}
nccl=$(PATH=$prefix/bin:$PATH tools/check_nccl.sh)
export LD_LIBRARY_PATH=$nccl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

if [ -n "${UNISERVE_E2E_DEVICES:-}" ]; then
    gpus=$(tr ',' '\n' <<< "$UNISERVE_E2E_DEVICES" | grep -c .)
else
    gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
fi
workers=${UNISERVE_E2E_WORKERS:-$gpus}

exec "${python[@]}" -m pytest -n "$workers" --dist loadgroup \
    --basetemp "$artifacts/pytest" --junitxml "$artifacts/junit.xml" "$@"
