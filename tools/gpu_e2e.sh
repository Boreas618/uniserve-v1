#!/usr/bin/env bash
# Runs the GPU end-to-end tests (`tests/python/e2e`) from one build of the tree.
#
# Usage, from any directory:
#
#   tools/gpu_e2e.sh [--memcheck] [pytest arguments...]
#
# The script builds the release `uniserve` executable, which is a test deployment's daemon and
# its host services, and the `uniserve-manager` the admin API's tests attach beside the
# executable, and exports the executable as `UNISERVE_BINARY`; when the environment names it, as a
# host sharing another host's build does, it uses it and builds nothing. A deployment's main
# processes run the engine package (`crates/engine`), which each test module's Build Server
# installs and builds with cargo from the installed snapshot of the workspace: that build finds
# FFmpeg 8.1 through pkg-config (`crates/media/README.md`) and reads `LIBCLANG_PATH` from this
# environment, and cargo keeps what it compiled between runs. Each Build Server compiles kernels
# with its toolchain environment, whose program `uniserve-build-server` must be on the search
# path. The serving processes open the driver, NCCL and CUPTI from the host's default loader path
# and refuse a release older than the one they require. The script sets the gate
# `UNISERVE_RUN_GPU_E2E=1` and runs pytest with one worker per GPU (pytest-xdist); each case
# leases the GPUs it serves on
# (`tests/python/e2e/conftest.py`). pytest's temporary directory, which holds every server's log,
# and the JUnit report go to `artifacts/e2e/<UTC time>/` unless `UNISERVE_E2E_ARTIFACTS` names
# another directory.
#
# `--memcheck` runs every main process under compute-sanitizer's memcheck: the deployments
# select the engine package `memcheck` (`tests/python/sanitizer`), whose executable runs the
# release `uniserve-engine` this script builds under the sanitizer, and its logs go to
# `$SANITIZER_LOGS` (`<artifacts>/memcheck` by default).
#
# Host facts come from the environment: the checkpoints and inputs the tests name
# (docs/testing.md lists them; a case whose asset is unset is skipped, naming the variable),
# `UNISERVE_E2E_DEVICES` to restrict the GPUs, and `UNISERVE_E2E_WORKERS` for another worker
# count. `PYTHON` and `CARGO` name the interpreter
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

if ! command -v uniserve-build-server > /dev/null; then
    echo "no uniserve-build-server on the search path: install the Build Server's environment" \
        "(uv sync --locked --project crates/build-server/python) with its program on the path" >&2
    exit 1
fi

# A run on a host that shares another host's build names that build's executable and builds
# nothing; otherwise the tree is built here.
if [ -z "${UNISERVE_BINARY:-}" ]; then
    # The manager runs beside the executable: the admin API's tests attach it to a deployment.
    "$cargo" build --release -p uniserve --bin uniserve -p manager --bin uniserve-manager
    export UNISERVE_BINARY=${CARGO_TARGET_DIR:-$root/target}/release/uniserve
fi
export UNISERVE_RUN_GPU_E2E=1
export MAX_JOBS=${MAX_JOBS:-16}

artifacts=${UNISERVE_E2E_ARTIFACTS:-$root/artifacts/e2e/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$artifacts"

if [ "$memcheck" = 1 ]; then
    export SANITIZER_LOGS=${SANITIZER_LOGS:-$artifacts/memcheck}
    mkdir -p "$SANITIZER_LOGS"
    # The engine the sanitizer runs: the tree's own, built here unless the environment names one.
    if [ -z "${UNISERVE_MEMCHECK_TARGET:-}" ]; then
        "$cargo" build --release -p engine --bin uniserve-engine
        export UNISERVE_MEMCHECK_TARGET=${CARGO_TARGET_DIR:-$root/target}/release/uniserve-engine
    fi
    export UNISERVE_ENGINE=memcheck
    export UNISERVE_ENGINE_PACKAGE=$root/tests/python/sanitizer
    # A main process under memcheck takes more of the device and runs slower.
    export UNISERVE_E2E_SET=${UNISERVE_E2E_SET:-"mem_fraction_static=0.6 worker_config.watchdog_timeout=2h"}
    export UNISERVE_E2E_TIME_SCALE=${UNISERVE_E2E_TIME_SCALE:-20}
fi

if [ -n "${UNISERVE_E2E_DEVICES:-}" ]; then
    gpus=$(tr ',' '\n' <<< "$UNISERVE_E2E_DEVICES" | grep -c .)
else
    gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
fi
workers=${UNISERVE_E2E_WORKERS:-$gpus}

exec "${python[@]}" -m pytest -n "$workers" --dist loadgroup \
    --basetemp "$artifacts/pytest" --junitxml "$artifacts/junit.xml" "$@"
