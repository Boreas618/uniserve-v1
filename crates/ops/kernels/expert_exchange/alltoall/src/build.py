"""The expert_exchange package's compile script: one descriptor -> one module.

The module is this package's exchange: the transport's lifecycle over the CUDA
virtual-memory API and the two directions of the exchange for one expert group,
exporting the five lifecycle entries and the two the operator calls. A
descriptor is the direction, the hidden width, the routes per token, the expert
count, the transport's capacity, the group size and the activation type; the
kernels read them at launch, so every descriptor of this package compiles the
same object. A group of several ranks is a descriptor this module refuses at
launch, not a different compilation.

The Program Server runs this script with its own Python environment, its working
directory the package root, and `UNISERVE_OP`, `UNISERVE_BACKEND`,
`UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`, `UNISERVE_PACKAGE`, `UNISERVE_OUT` and
`MAX_JOBS` in the environment.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

#: The translation unit of the module: every entry and both kernels.
SOURCES = ["src/exchange.cu"]

#: The entries the module exports; the package's `[module.entries]` names the
#: launch role, the harness asks the transport's views through `info`.
ENTRIES = [
    "uniserve_expert_exchange_dispatch",
    "uniserve_expert_exchange_combine",
    "uniserve_expert_exchange_create",
    "uniserve_expert_exchange_suspend",
    "uniserve_expert_exchange_restore",
    "uniserve_expert_exchange_release",
    "uniserve_expert_exchange_info",
]


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
    """Compile the exchange for one descriptor into ``module.so``."""
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
        "-lcuda",
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
    missing = [entry for entry in ENTRIES if entry not in symbols]
    if missing:
        raise SystemExit(f"module.so does not export {', '.join(missing)}")
    print(
        json.dumps(
            {"descriptor": descriptor, "target": target, "entries": ENTRIES}
        )
    )


if __name__ == "__main__":
    main()
