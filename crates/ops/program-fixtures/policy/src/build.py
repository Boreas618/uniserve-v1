"""Build the policy fixture's wasm32 module from the crate that holds it.

The package carries no module and no copy of the policy: the policy is
`crates/scheduling/policies/fcfs`, whose body is the guest crate's shared
arrival-order implementation. This script asks cargo for that crate's wasm32
build, with the target directory inside the build's own workspace so nothing is
written into the tree, and leaves the raw module for the wasm toolchain, which
precompiles it for this host's CPU and stores `module.wasm`.

It runs when the package is installed, from the directory the package was
published from (design 6.2), so the crate is reachable through its workspace. A
host that only serves the installed revision runs no build at all.

The Program Server runs this with `UNISERVE_PACKAGE`, `UNISERVE_OUT`,
`UNISERVE_TARGET`, `UNISERVE_DESCRIPTOR`, `UNISERVE_OP`, `UNISERVE_BACKEND`
and `MAX_JOBS` in the environment.
"""

import os
import pathlib
import shutil
import subprocess
import sys

#: The crate that holds the policy this fixture installs: sched's `fcfs`.
POLICY = "scheduling-policy-fcfs"

#: The target a policy is built for: the store's segment for a policy as well.
TARGET = "wasm32-unknown-unknown"


def repository(package):
    """The workspace root above the package: where the policy crate is a member."""
    for candidate in [package, *package.parents]:
        manifest = candidate / "Cargo.toml"
        if manifest.is_file() and "[workspace]" in manifest.read_text():
            return candidate
    raise SystemExit(f"no workspace root above {package}")


def main():
    """Build the policy for wasm32 and leave the raw module in the output.

    The compile is cached in the installed package's own `.cargo-target`, so the
    second build of one revision — the module the main process prepares — reuses
    the first one's artifacts instead of recompiling the guest crate cold.
    """
    package = pathlib.Path(os.environ["UNISERVE_PACKAGE"]).resolve()
    out = pathlib.Path(os.environ["UNISERVE_OUT"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cargo = os.environ.get("CARGO") or shutil.which("cargo")
    if not cargo:
        raise SystemExit(
            "no cargo on this host: the policy fixture builds its module with the "
            "Rust toolchain"
        )
    root = repository(package)
    # A build directory that outlives one build. One delivery run builds this
    # package more than once — the daemon resolves the epoch's policy, then the
    # main process prepares the modules it installs — and a cold wasm32 build of
    # the guest crate is minutes, longer than the bound the main process holds a
    # prepare under. It sits inside the installed package, which is the store's
    # own copy, so the tree the package was published from is never written to,
    # and cargo reuses everything the first compile produced.
    target_directory = package / ".cargo-target"
    command = [
        cargo,
        "build",
        "--release",
        "--package",
        POLICY,
        "--target",
        TARGET,
        "--target-dir",
        str(target_directory),
        "--offline",
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if built.returncode != 0:
        raise SystemExit(
            f"cargo failed with {built.returncode}:\n{built.stderr[-4000:]}"
        )
    write = target_directory / TARGET / "release"
    modules = sorted(write.glob("*.wasm"))
    if len(modules) != 1:
        raise SystemExit(f"expected one .wasm in {write}, found {modules}")
    shutil.copy(modules[0], out / "policy.wasm")
    print(f"{modules[0].name}: {modules[0].stat().st_size} bytes")


if __name__ == "__main__":
    main()
