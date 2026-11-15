"""The range_marker package's build module: one descriptor -> one cubin.

The descriptor names which of the two kernels of ``range_marker.cu`` its
module holds: ``stamp`` (a range marker) or ``clock`` (a clock read for a
calibration). Both are single-thread kernels, launched with one block of one
warp, so the module declares one warp and no shared memory.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys

from uniserve_build_server.request import Request

#: Each descriptor's kernel, by the descriptor's ``function`` value: its
#: symbol and the parameters it takes at launch, in order and with their types
#: as a module's signature spells them, so ``ops`` compares the engine's packed
#: words against them.
FUNCTIONS = {
    "stamp": (
        "uniserve_range_marker",
        {
            "switches": "*u32",
            "klass": "u32",
            "marker": "u32",
            "cell": "*u32",
            "mark": "u32",
            "step": "u32",
            "counter": "*u64",
            "written": "*u32",
            "stamps": "*u64",
            "filled": "*u64",
            "shift": "u32",
        },
    ),
    "clock": ("uniserve_clock", {"out": "*u64"}),
}


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.cubin`` and its metadata.

    The cubin and ``metadata.json`` are written into the request's output
    directory; a function the package does not hold or a compiler failure
    refuses the build, with the compiler's output in the build log.
    """
    function = request.field("function")
    if function not in FUNCTIONS:
        raise SystemExit(f"function {function!r} is no kernel of this package")
    name, signature = FUNCTIONS[function]
    out = request.output

    # The nvcc the Build Server checked against this package's declared version.
    compiler = request.toolchain.require_cuda().nvcc
    command = [
        compiler,
        "-cubin",
        f"-arch={request.target}",
        "-O3",
        "-std=c++20",
        "--threads=1",
        "-o",
        str(out / "module.cubin"),
        str(request.package / "src" / "range_marker.cu"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}")

    metadata = {
        "name": name,
        "num_warps": 1,
        "num_ctas": 1,
        "shared": 0,
        "launch_pdl": False,
        "launch_cooperative_grid": False,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "signature": signature,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "descriptor": request.descriptor,
                "target": str(request.target),
                "symbol": name,
            }
        )
    )
