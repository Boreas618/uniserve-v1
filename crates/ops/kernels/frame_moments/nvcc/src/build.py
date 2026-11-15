"""The frame_moments package's build module: one descriptor -> one module.

The module is this package's unit, `src/frame.cu`, linked into a shared object
exporting `uniserve_frame_moments`. The kernel reads every extent, stride and
branch from the call structure, so every descriptor compiles the same object.

The unit reuses ATen's Welford operator and block reduction, compiled from the
headers the locked torch distribution installs (the manifest names the
distribution), where the distribution installs them.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's unit.
SOURCE = "src/frame.cu"

#: Where the torch distribution installs its C++ headers, ATen's among them.
ATEN = ("torch", "torch/include")


def compile(request: Request) -> None:
    """Compile the kernel for the request's descriptor into ``module.so``.

    A distribution that installs no headers, or a compiler failure, refuses the
    build, with the compiler's output in the build log.
    """
    try:
        aten = fetch.installed(*ATEN)
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error

    # No contraction in host code: the arithmetic is PyTorch's, and its
    # expressions round as PyTorch's do.
    unit = cuda.Unit(
        source=request.package / SOURCE,
        standard="c++17",
        includes=[aten],
        host_contraction=False,
    )
    cuda.native(request, [unit])
