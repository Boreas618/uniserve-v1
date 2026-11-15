"""The frame_norm_pad package's build module: one descriptor -> one module.

The module is this package's kernel compiled with nvcc into a shared object
exporting `uniserve_frame_norm_pad`. The kernel reads every extent, stride and
branch from the call structure, so every descriptor of this package compiles
the same object; the descriptor name is what the store keys the revision by.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys

from uniserve_build_server.request import Request

#: The translation unit of the module.
SOURCE = "src/frame.cu"

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_frame_norm_pad"


def compile(request: Request) -> None:
    """Compile the kernel for the request's descriptor into ``module.so``.

    The module is written into the request's output directory; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    # The nvcc the Build Server checked against this package's declared version.
    compiler = request.toolchain.require_cuda().nvcc

    # No fast math: the arithmetic is PyTorch's, and its expressions must round
    # as PyTorch's do.
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-Xcompiler",
        "-ffp-contract=off",
        f"-I{package / 'include'}",
        str(package / SOURCE),
        "-lcudart",
        "-o",
        str(request.output / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}")

    print(
        json.dumps(
            {
                "descriptor": request.descriptor,
                "target": str(request.target),
                "entry": ENTRY,
            }
        )
    )
