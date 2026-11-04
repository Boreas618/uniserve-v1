"""The sparse_attention package's compile script: one descriptor -> one native module.

The module is this package's kernel compiled with nvcc into a shared object exporting
`uniserve_sparse_attention`. The kernel reads the geometry, the strides and the selection
from the call structure, so the descriptor decides only what a compilation cannot vary:

* the sparse block size, which is a compile-time constant of the kernel (the vendor keeps
  the 64-row and 128-row builds in separate namespaces for exactly this reason), taken
  from the descriptor's `block`;
* the tensor-map layout, which this package always takes as strided rows and heads
  (`VSA_BHSD`), because the operator's call addresses rows and heads through strides.

The Program Server runs this script with its own Python environment, its working directory the
package root, and `UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`, `UNISERVE_PACKAGE` and
`UNISERVE_OUT` in the environment; `--package`, `--descriptor`, `--target` and `--out` name
the same values on the command line.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

#: The translation unit of the module.
SOURCE = "src/attention.cu"

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_sparse_attention"

#: The block sizes the kernel knows; the descriptor's `block` chooses one.
BLOCKS = (64, 128)


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


def block_of(descriptor):
    """The sparse block size a descriptor names, from its `block` field."""
    named = re.search(r"(?:^|_)block_(\d+)(?:_|$)", descriptor)
    if named is None:
        raise SystemExit(f"the descriptor {descriptor!r} names no block size")
    block = int(named.group(1))
    if block not in BLOCKS:
        raise SystemExit(f"the kernel knows block sizes {BLOCKS}, not {block}")
    return block


def main():
    """Compile the kernel for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    out.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    block = block_of(descriptor)

    # The kernel's own two choices: which block size, and that the maps take strided rows
    # and heads. Both must precede the header that reads them.
    defines = [
        "-DVSA_BHSD=1",
        f"-DVSA_BLK128={1 if block == 128 else 0}",
    ]
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        *defines,
        f"-I{package / 'src'}",
        f"-I{package / 'include'}",
        str(package / SOURCE),
        "-lcuda",
        "-lcudart",
        "-o",
        str(out / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}:\n{built.stderr[-4000:]}")

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(out / "module.so")],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {line.split()[-1] for line in exported.stdout.splitlines() if line.split()}
    if ENTRY not in symbols:
        raise SystemExit(f"module.so does not export {ENTRY}")
    print(json.dumps({"descriptor": descriptor, "target": target, "entry": ENTRY, "block": block}))


if __name__ == "__main__":
    main()
