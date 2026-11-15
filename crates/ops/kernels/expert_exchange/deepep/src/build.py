"""The deepep package's build module: one descriptor -> one module.

The descriptor fixes every template argument of DeepEP's kernels, so this
compiles them here, and a serving host compiles nothing. The Build Server
compiles a descriptor by calling :func:`compile` with the compile request.

DeepEP's kernels and the NCCL device API they are built against are pinned
sources (`uniserve_build_server.fetch`'s `deepep` and `nccl` pins): the
compile includes them from their verified cache directories.
"""

import json
import pathlib
import subprocess
import sys
import sysconfig

from uniserve_build_server import fetch
from uniserve_build_server.request import Request

SOURCES = ["src/exchange.cu"]

ENTRIES = [
    "uniserve_expert_exchange_launch",
    "uniserve_expert_exchange_create",
    "uniserve_expert_exchange_step",
    "uniserve_expert_exchange_suspend",
    "uniserve_expert_exchange_restore",
    "uniserve_expert_exchange_release",
    "uniserve_expert_exchange_info",
]


def constants(request):
    """The compile-time values a descriptor states, by the macro each defines.

    The direction and the element type are the manifest's single choices, so
    only the extents reach the compile.
    """
    return {
        "UNISERVE_HIDDEN": request.field("hidden"),
        "UNISERVE_TOP": request.field("top"),
        "UNISERVE_EXPERTS": request.field("experts"),
        "UNISERVE_MAX_TOKENS": request.field("max_tokens"),
        "UNISERVE_RANKS": request.field("ranks"),
        "UNISERVE_ATTENTION_RANKS": request.field("attention"),
    }


def nccl_library():
    """The directory of a `libnccl.so.2` to name in the module's needed list.

    The module takes the serving process's NCCL at run time, which
    `tools/check_nccl.sh` holds to the project's version; this one only states
    the library's name to the linker.
    """
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    directory = site / "nvidia" / "nccl" / "lib"
    if not (directory / "libnccl.so.2").exists():
        raise SystemExit(f"no libnccl.so.2 in {directory}")
    return directory


def compile(request: Request) -> None:
    """Compile the exchange for the request's descriptor into ``module.so``.

    The pinned sources are fetched into the Build Server's cache of pins when
    it does not hold them yet; a compiler failure refuses the build, with the
    compiler's output in the build log.
    """
    root = request.package
    try:
        deepep = fetch.fetch(
            fetch.pin("deepep"), request.cache, jobs=request.jobs
        )
        nccl = fetch.fetch(fetch.pin("nccl"), request.cache, jobs=request.jobs)
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error
    # The nvcc the Build Server checked against this package's declared
    # version, and its driver API stub: the kernels call the driver directly.
    cuda = request.toolchain.require_cuda()
    defines = [
        f"-D{name}={value}" for name, value in constants(request).items()
    ]
    command = [
        cuda.nvcc,
        "-O3",
        "-std=c++20",
        f"-arch={request.target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "--diag-suppress=39,128,161,174,177,186,940,2417,3012",
        "--ptxas-options=--register-usage-level=10",
        "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
        *defines,
        f"-I{root / 'include'}",
        f"-I{deepep / 'deep_ep' / 'include'}",
        f"-I{nccl / 'nvidia' / 'nccl' / 'include'}",
        *[str(root / source) for source in SOURCES],
        "-lcudart",
        f"-L{cuda.stubs}",
        "-lcuda",
        f"-L{nccl_library()}",
        "-Xlinker",
        "--no-as-needed",
        "-l:libnccl.so.2",
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
                "entries": ENTRIES,
            }
        )
    )
