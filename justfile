# Shortcuts for local checks and tests. Build is `uv sync` / `cargo`;
# serving is `uniserve serve <model> --served-model-name <name>`.

python := env("PYTHON", ".venv/bin/python")
cargo := env("CARGO", "cargo")

# Format check (Rust)
fmt:
    {{cargo}} fmt --all -- --check

# Clippy lint (Rust)
clippy:
    {{cargo}} clippy --workspace --all-targets -- -D warnings

# Run Rust tests
test-rust:
    {{cargo}} test --workspace

# Build benchmarks without running
bench-build:
    {{cargo}} bench --workspace --no-run

# Lint: fmt + clippy + ruff + mypy
lint: fmt clippy
    {{python}} -m ruff check uniserve uniserve_eval uniserve_kernels/src tests
    {{python}} -m ruff format --check uniserve uniserve_eval uniserve_kernels/src tests
    {{python}} -m mypy

# Fast Python tests (CPU only)
test-python-fast:
    {{python}} -m pytest -m "unit and not gpu"

# Integration Python tests (CPU only)
test-python-integration:
    {{python}} -m pytest -m "integration and not gpu"

# Unit and integration Python tests that require a CUDA device
test-python-cuda:
    {{python}} -m pytest -m "(unit or integration) and gpu"

# Build the debug binary if missing
[private]
build-debug:
    {{cargo}} build -p uniserve

# End-to-end Python tests (no GPU)
test-python-e2e: build-debug
    {{python}} -m pytest -m "e2e and not gpu"

# The missing-state tests (GPU) with every worker under compute-sanitizer's
# memcheck; its logs go to $SANITIZER_LOGS
test-python-sanitizer: build-debug
    UNISERVE_RUN_GPU_E2E=1 \
    UNISERVE_WORKER_EXECUTABLE={{justfile_directory()}}/tests/python/sanitizer/memcheck-worker \
    UNISERVE_MEMCHECK_TARGET=$({{python}} -c 'import os, sys; print(os.path.join(os.path.dirname(sys.executable), "uniserve"))') \
    UNISERVE_E2E_SET="mem_fraction_static=0.6 worker_config.watchdog_timeout=2h" \
    UNISERVE_E2E_TIME_SCALE=20 \
    {{python}} -m pytest -m "e2e and gpu and missing_state"

# End-to-end Python tests (GPU, sensenova)
test-python-gpu: build-debug
    UNISERVE_RUN_GPU_E2E=1 {{python}} -m pytest -m "e2e and gpu and sensenova"

# Run all checks
test-all: fmt clippy test-rust bench-build test-python-fast test-python-integration test-python-e2e

# Resolve the decode-runtime suite without starting a server
eval-plan:
    {{python}} -m uniserve_eval plan decode-runtime
