# Shortcuts for local checks and tests. Build is `uv sync` / `cargo`;
# serving is `uniserve serve <model> --served-model-name <name>`.

python := env("PYTHON", "uv run python")
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
    {{python}} -m ruff check uniserve_eval tests
    {{python}} -m ruff format --check uniserve_eval tests
    {{python}} -m mypy

# Fast Python tests (CPU only); the kernel checks run on Triton's interpreter. A kernel check that
# launches what a package states for a descriptor key asks the environment's `uniserve` (`uniserve
# build-server launch`, which runs the Build Server's environment program `uniserve-build-server`
# from the search path), which `uv sync` installs with the root project.
test-python-fast:
    {{python}} -m pytest -m "unit and not gpu"

# The missing-state tests (GPU) with every main process under compute-sanitizer's
# memcheck; the logs go to $SANITIZER_LOGS (the run's artifacts by default).
# docs/testing.md lists the variables the tests read.
test-python-sanitizer:
    PYTHON="{{python}}" tools/gpu_e2e.sh --memcheck -m "e2e and gpu and missing_state"

# End-to-end Python tests (GPU), one case per free GPU, from one release
# build. docs/testing.md lists the variables the tests read.
test-python-gpu:
    PYTHON="{{python}}" tools/gpu_e2e.sh -m "e2e and gpu"

# Run all checks
test-all: fmt clippy test-rust bench-build test-python-fast

# Resolve the decode-runtime suite without starting a server
eval-plan:
    {{python}} -m uniserve_eval plan decode-runtime
