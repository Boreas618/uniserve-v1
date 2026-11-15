"""The sparse_attention package's build module: one descriptor -> one module.

The module is this package's kernel compiled with nvcc into a shared object
exporting `uniserve_sparse_attention`. The kernel reads the geometry, the
strides and the selection from the call structure, so the descriptor decides
only what a compilation cannot vary:

* the sparse block size, which is a compile-time constant of the kernel (the
  vendor keeps the 64-row and 128-row builds in separate namespaces for exactly
  this reason), taken from the descriptor's `block`;
* the tensor-map layout, which this package always takes as strided rows and
  heads (`VSA_BHSD`), because the operator's call addresses rows and heads
  through strides.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys

from uniserve_build_server.request import Request

#: The translation unit of the module.
SOURCE = "src/attention.cu"

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_sparse_attention"

#: The block sizes the kernel knows; the descriptor's `block` chooses one.
BLOCKS = (64, 128)


def block_of(request):
    """The sparse block size the request's descriptor names in `block`."""
    block = int(request.field("block"))
    if block not in BLOCKS:
        raise SystemExit(f"the kernel knows block sizes {BLOCKS}, not {block}")
    return block


def compile(request: Request) -> None:
    """Compile the kernel for the request's descriptor into ``module.so``.

    The module is written into the request's output directory; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    # The nvcc the Build Server checked against this package's declared
    # version, and its driver API stub: the kernel's tensor maps are encoded
    # through the driver.
    cuda = request.toolchain.require_cuda()
    block = block_of(request)

    # The kernel's own two choices: which block size, and that the maps take
    # strided rows and heads. Both must precede the header that reads them.
    defines = [
        "-DVSA_BHSD=1",
        f"-DVSA_BLK128={1 if block == 128 else 0}",
    ]
    command = [
        cuda.nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        *defines,
        f"-I{package / 'src'}",
        f"-I{package / 'include'}",
        str(package / SOURCE),
        f"-L{cuda.stubs}",
        "-lcuda",
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
                "block": block,
            }
        )
    )
