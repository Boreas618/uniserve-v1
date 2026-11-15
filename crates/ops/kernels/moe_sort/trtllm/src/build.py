"""The moe_sort package's build module: one descriptor -> one module.

The module is this package's routing pipeline (TensorRT-LLM's block path, see
`src/routing.cu`) compiled with nvcc into a shared object exporting
`uniserve_moe_sort`. A descriptor is the expert count, the routes per token,
the local window, the tile and the route-weight type; the kernels read all of
them at launch, so every descriptor of this package compiles the same object.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys

from uniserve_build_server.request import Request

#: The translation units of the module: the entry and the routing kernels.
SOURCES = ["src/entry.cpp", "src/routing.cu"]

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_moe_sort"


def compile(request: Request) -> None:
    """Compile the routing kernels for the request's descriptor into a module.

    ``module.so`` is written into the request's output directory; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    # The nvcc the Build Server checked against this package's declared version.
    compiler = request.toolchain.require_cuda().nvcc

    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-I{package / 'include'}",
        *[str(package / source) for source in SOURCES],
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
