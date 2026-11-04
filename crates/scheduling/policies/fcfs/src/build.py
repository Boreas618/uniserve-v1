#!/usr/bin/env python3
"""Build one scheduling policy's wasm32 module.

The Program Server runs this in the package root as the package's `[source]
script` (design §5.2, §6.2): the script writes the raw module, and the server
then precompiles it with `scheduling-precompile` into the artifact the main
process loads.

Contract with the server:

- reads `UNISERVE_OUT`, the directory it must write into;
- writes `$UNISERVE_OUT/policy.wasm`, the raw wasm32 module;
- exits non-zero with the build's output on stderr when the crate does not
  build.

The crate is compiled to the `wasm32-unknown-unknown` target in a target
directory inside `UNISERVE_OUT`, so the package's own tree stays clean and two
policies never share a build directory.
"""

import os
import pathlib
import shutil
import subprocess
import sys

BACKEND = "fcfs"


def main() -> int:
    out = os.environ.get("UNISERVE_OUT")
    if not out:
        print("UNISERVE_OUT is not set: the server names the output directory", file=sys.stderr)
        return 2
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    package = pathlib.Path(__file__).resolve().parent.parent
    # The package name is the crate the manifest states; the artifact keeps
    # cargo's underscore spelling of it.
    crate = f"scheduling_policy_{BACKEND}"
    target_dir = out / "cargo"
    build = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "--target-dir",
            str(target_dir),
        ],
        cwd=package,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stdout)
        sys.stderr.write(build.stderr)
        return build.returncode
    module = target_dir / "wasm32-unknown-unknown" / "release" / f"{crate}.wasm"
    if not module.is_file():
        print(f"the build wrote no {module}", file=sys.stderr)
        return 1
    shutil.copyfile(module, out / "policy.wasm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
