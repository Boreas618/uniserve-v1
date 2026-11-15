"""The sparse_attention sm90 package's build module.

One descriptor -> one native module.

The module is this package's kernel compiled with nvcc into a shared object
exporting `uniserve_sparse_attention`. The kernel reads the geometry, the
strides and the selection from the call structure, so every descriptor of this
package compiles the same object: the SM90a tile is fixed at 128 rows in both
directions, which is the only block it serves.

The kernel is CuTe and CUTLASS source, so nvcc needs their headers. They come
from the CUTLASS that ships with the FlashInfer distribution of the Build
Server's environment; the header directory that was used is printed with the
build's result, so a revision says what it was compiled against.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys

from uniserve_build_server import trtllm
from uniserve_build_server.request import Request

#: The translation unit of the module.
SOURCE = "src/attention.cu"

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_sparse_attention"


def cutlass_include():
    """The CUTLASS header directory this build uses.

    It holds `cute/` beside `cutlass/`: the CUTLASS FlashInfer ships for its
    own kernels, in the locked distribution's data tree.
    """
    include = trtllm.data_tree() / "cutlass" / "include"
    if not (
        (include / "cute" / "tensor.hpp").is_file()
        and (include / "cutlass" / "version.h").is_file()
    ):
        raise SystemExit(
            f"no CUTLASS headers at {include}: the FlashInfer distribution of "
            "the Build Server's environment carries them"
        )
    return include


def compile(request: Request) -> None:
    """Compile the kernel for the request's descriptor into ``module.so``.

    The module is written into the request's output directory; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    # The nvcc the Build Server checked against this package's declared
    # version, and its driver API stub: CuTe encodes the kernel's tensor maps
    # through the driver.
    cuda = request.toolchain.require_cuda()
    cutlass = cutlass_include()

    # CuTe's host-side helpers call constexpr functions from
    # `__host__ __device__` context, which nvcc allows only under this flag;
    # the vendor's own extension build sets it too.
    command = [
        cuda.nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
        "--expt-relaxed-constexpr",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-I{package / 'src'}",
        f"-I{package / 'include'}",
        f"-I{cutlass}",
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
                "cutlass": str(cutlass),
            }
        )
    )
