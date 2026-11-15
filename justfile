# The repository's checks and test suites. cargo is rustup's, at the release
# `rust-toolchain.toml` pins; every Python command runs in the repository's
# environment, which installs the `uniserve` executable the tests run. Serving
# is `uniserve serve <model> --served-model-name <name>`.

# A recipe's arguments reach its command as the shell's positional arguments,
# so `"$@"` passes each one as it was given, a marker expression's spaces
# included.
set positional-arguments

# A Python command, in the repository's environment with the extras of the
# development tools, the tests and the evaluator (docs/testing.md), which uv
# first brings up to date with the lockfile.
uv_run := "uv run --locked --extra dev --extra test --extra bench"

# Where a GPU suite writes what a run leaves: pytest's temporary directory,
# which holds every server's log, and the JUnit report. `just e2e=DIR <recipe>`
# names another directory.
e2e := "artifacts/e2e/" + datetime_utc("%Y%m%dT%H%M%SZ")

[doc("Format check (Rust)")]
fmt:
    cargo fmt --all -- --check

[doc("Clippy lint (Rust); arguments go to cargo, such as `--features media/vendored`")]
clippy *cargo:
    cargo clippy --workspace --all-targets "$@" -- -D warnings

[doc("Run Rust tests; arguments go to cargo")]
test-rust *cargo:
    cargo test --workspace "$@"

[doc("Build benchmarks without running; arguments go to cargo")]
bench-build *cargo:
    cargo bench --workspace --no-run "$@"

[doc("Lint: fmt + clippy + ruff + mypy")]
lint: fmt clippy
    {{ uv_run }} ruff check uniserve_eval tests
    {{ uv_run }} ruff format --check uniserve_eval tests
    {{ uv_run }} mypy

# The kernel checks run on Triton's interpreter.
[doc("Fast Python tests (CPU only); arguments go to pytest")]
test-python-fast *pytest:
    {{ uv_run }} pytest -m "unit and not gpu" "$@"

# One pytest-xdist worker per GPU the run may lease, each case on the GPUs it
# leases (docs/testing.md). pytest makes its temporary directory, but not the
# directories above it.
[doc("End-to-end Python tests (GPU); arguments go to pytest, such as a module")]
test-python-gpu *pytest:
    mkdir -p "{{ e2e }}"
    {{ uv_run }} pytest -n auto --dist loadgroup --basetemp "{{ e2e }}/pytest" \
        --junitxml "{{ e2e }}/junit.xml" -m "e2e and gpu" "$@"

# Every deployment runs the engine package `memcheck` (tests/python/sanitizer),
# with more of each device left free and every wait stretched; the package
# awaits a declared engine launcher (docs/testing.md).
[doc("The missing-state tests (GPU) under compute-sanitizer's memcheck; arguments go to pytest")]
test-python-sanitizer *pytest:
    {{ just_executable() }} e2e="{{ e2e }}" test-python-gpu --engine memcheck --time-scale 20 \
        --serve-set mem_fraction_static=0.6 --serve-set worker_config.watchdog_timeout=2h \
        -m "e2e and gpu and missing_state" "$@"

[doc("Run all checks")]
test-all: fmt clippy test-rust bench-build test-python-fast
