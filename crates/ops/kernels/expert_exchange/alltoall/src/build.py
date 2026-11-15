"""The expert_exchange package's build module: one descriptor -> one module.

The module is this package's exchange: the transport's lifecycle over the CUDA
virtual-memory API and both directions of the exchange for one expert group,
exporting the six lifecycle entries and the one the operator calls. A
descriptor is the direction, the hidden width, the routes per token, the expert
count, the transport's capacity, the group size and the activation type; the
kernels read them at launch, so every descriptor of this package compiles the
same object.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import subprocess
import sys

from uniserve_build_server.request import Request

#: The translation unit of the module: every entry and the kernels.
SOURCES = ["src/exchange.cu"]


def compile(request: Request) -> None:
    """Compile the exchange for the request's descriptor into ``module.so``.

    The module is written into the request's output directory; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    # The nvcc the Build Server checked against this package's declared
    # version, and its driver API stub: the transport calls the driver's
    # virtual-memory API directly.
    cuda = request.toolchain.require_cuda()

    command = [
        cuda.nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-I{package / 'include'}",
        *[str(package / source) for source in SOURCES],
        "-lcudart",
        f"-L{cuda.stubs}",
        "-lcuda",
        "-o",
        str(request.output / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}")
