"""The sparse_attention wgmma package's build module.

One descriptor -> one module: this package's kernel and entry,
`src/attention.cu`, linked into a shared object exporting
`uniserve_sparse_attention`. The kernel reads the geometry, the strides and the
selection from the call structure, so every descriptor of this package compiles
the same object: the SM90a tile is fixed at 128 rows in both directions, which
is the only block it serves.

The kernel is CuTe and CUTLASS source, compiled against the CUTLASS headers the
locked FlashInfer distribution installs (the manifest names the distribution),
where the distribution installs them.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's unit: the kernel's entry.
SOURCE = "src/attention.cu"

#: Where the FlashInfer distribution installs CUTLASS's headers, `cute/` beside
#: `cutlass/`.
CUTLASS = ("flashinfer-python", "flashinfer/data/cutlass/include")


def compile(request: Request) -> None:
    """Compile the kernel for the request's descriptor into ``module.so``.

    A distribution that installs no CUTLASS headers, or a compiler failure,
    refuses the build, with the compiler's output in the build log.
    """
    try:
        cutlass = fetch.installed(*CUTLASS)
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error

    # CuTe's host-side helpers call constexpr functions from
    # `__host__ __device__` context, which nvcc allows only with relaxed
    # constexpr; CUTLASS's own build compiles every unit so.
    unit = cuda.Unit(
        source=request.package / SOURCE,
        standard="c++17",
        includes=[cutlass],
        relaxed_constexpr=True,
    )
    # CuTe encodes the kernel's tensor maps through the driver API.
    cuda.native(request, [unit], libraries=["cuda"])
