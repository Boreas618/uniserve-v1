"""The sparse_attention sm90 package's compile script: one descriptor -> one native module.

The module is this package's kernel compiled with nvcc into a shared object exporting
`uniserve_sparse_attention`. The kernel reads the geometry, the strides and the selection
from the call structure, so every descriptor of this package compiles the same object: the
SM90a tile is fixed at 128 rows in both directions, which is the only block it serves.

The kernel is CuTe and CUTLASS source, so nvcc needs their headers. They are found in the
order a build host most plausibly has them — `CUTLASS_PATH`, then the CUTLASS that ships
with the Program Server's own environment (flashinfer's copy), then the system's
`/usr/include/cutlass` — and the header directory that was used is printed with the build's
result, so a revision says what it was compiled against.

The Program Server runs this script with its own Python environment, its working directory the
package root, and `UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`, `UNISERVE_PACKAGE` and
`UNISERVE_OUT` in the environment; `--package`, `--descriptor`, `--target` and `--out` name
the same values on the command line.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig

#: The translation unit of the module.
SOURCE = "src/attention.cu"

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_sparse_attention"


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


def cutlass_include():
    """The CUTLASS header directory this build uses, with `cute/` beside `cutlass/`."""
    candidates = []
    if os.environ.get("CUTLASS_PATH"):
        candidates.append(pathlib.Path(os.environ["CUTLASS_PATH"]) / "include")
    # The Program Server's own environment: flashinfer ships the CUTLASS its kernels use.
    for key in ("purelib", "platlib"):
        site = sysconfig.get_paths().get(key)
        if site:
            candidates.append(pathlib.Path(site) / "flashinfer" / "data" / "cutlass" / "include")
    candidates.append(pathlib.Path("/usr/include"))
    for candidate in candidates:
        if (candidate / "cute" / "tensor.hpp").is_file() and (
            candidate / "cutlass" / "version.h"
        ).is_file():
            return candidate
    raise SystemExit(
        "no CUTLASS headers found: set CUTLASS_PATH to a CUTLASS checkout, or install the "
        "flashinfer package whose data directory carries them"
    )


def main():
    """Compile the kernel for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    out.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cutlass = cutlass_include()

    # CuTe's host-side helpers call constexpr functions from `__host__ __device__` context,
    # which nvcc allows only under this flag; the vendor's own extension build sets it too.
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--expt-relaxed-constexpr",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-I{package / 'src'}",
        f"-I{package / 'include'}",
        f"-I{cutlass}",
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
    print(
        json.dumps(
            {
                "descriptor": descriptor,
                "target": target,
                "entry": ENTRY,
                "cutlass": str(cutlass),
            }
        )
    )


if __name__ == "__main__":
    main()
