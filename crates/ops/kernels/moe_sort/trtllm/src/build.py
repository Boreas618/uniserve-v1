"""The moe_sort package's compile script: one descriptor -> one module.

The module is this package's routing pipeline (TensorRT-LLM's block path, see
`src/routing.cu`) compiled with nvcc into a shared object exporting
`uniserve_moe_sort`. A descriptor is the expert count, the routes per token,
the local window, the tile and the route-weight type; the kernels read all of
them at launch, so every descriptor of this package compiles the same object.

The Program Server runs this script with its own Python environment, its working
directory the package root, and `UNISERVE_OP`, `UNISERVE_BACKEND`,
`UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`, `UNISERVE_PACKAGE`, `UNISERVE_OUT` and
`MAX_JOBS` in the environment; `--package`, `--descriptor`, `--target` and
`--out` name the same values on the command line.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

#: The translation units of the module: the entry and the routing kernels.
SOURCES = ["src/entry.cpp", "src/routing.cu"]

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_moe_sort"


def arguments():
    """The package, descriptor, target and output directory of one build."""
    reader = argparse.ArgumentParser(description=__doc__)
    reader.add_argument("--package")
    reader.add_argument("--descriptor")
    reader.add_argument("--target")
    reader.add_argument("--out")
    known = reader.parse_args()
    values = {
        "package": known.package or os.environ.get("UNISERVE_PACKAGE"),
        "descriptor": known.descriptor or os.environ.get("UNISERVE_DESCRIPTOR"),
        "target": known.target or os.environ.get("UNISERVE_TARGET"),
        "out": known.out or os.environ.get("UNISERVE_OUT"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}")
    return (
        pathlib.Path(values["package"]).resolve(),
        values["descriptor"],
        values["target"],
        pathlib.Path(values["out"]),
    )


def main():
    """Compile the routing kernels for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    out.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-I{package / 'include'}",
        *[str(package / source) for source in SOURCES],
        "-lcudart",
        "-o",
        str(out / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(
            f"nvcc failed with {built.returncode}:\n{built.stderr[-4000:]}"
        )

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(out / "module.so")],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {
        line.split()[-1]
        for line in exported.stdout.splitlines()
        if line.split()
    }
    if ENTRY not in symbols:
        raise SystemExit(f"module.so does not export {ENTRY}")
    print(
        json.dumps({"descriptor": descriptor, "target": target, "entry": ENTRY})
    )


if __name__ == "__main__":
    main()
