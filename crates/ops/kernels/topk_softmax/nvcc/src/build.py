"""The topk_softmax package's build module: one descriptor -> one cubin.

The descriptor bakes the two values the kernel needs at compile time: the
scores a lane walks (the expert count rounded up to whole warps) and the
butterfly width of the renormalization (the top-k width rounded up to a power
of two). One warp routes one token, so the module declares four warps and no
dynamic shared memory.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys

from uniserve_build_server.request import Request

#: The warps one block holds, which the operator's grid counts tokens by.
WARPS_PER_BLOCK = 4

#: The storage kinds the kernel reads, keyed by descriptor value.
KINDS = {
    "f32": 0,
    "bf16": 1,
    "f16": 2,
    "fp32": 0,
    "fp16": 2,
    "none": -1,
}

#: The element type each storage kind reads, spelled as a module's signature
#: spells one.
POINTERS = {
    "f32": "fp32",
    "fp32": "fp32",
    "bf16": "bf16",
    "f16": "fp16",
    "fp16": "fp16",
}


def compile(request: Request) -> None:
    """Compile the request's descriptor into `module.cubin` and its metadata.

    This kernel is specialized on every dimension its manifest declares, so a
    key that leaves one out is refused by name. The cubin, `metadata.json` and
    `baked.json` are written into the request's output directory; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    out = request.output
    experts = int(request.field("experts"))
    top = int(request.field("top"))
    renormalize = request.field("renormalize")
    scores = request.field("scores")
    scale = request.field("scale")
    iterations = max(1, -(-experts // 32))
    butterfly = 1 << max(0, (top - 1).bit_length())
    if renormalize not in ("true", "false"):
        raise SystemExit(f"renormalize {renormalize!r} is not a flag")
    if scores not in KINDS:
        raise SystemExit(f"scores {scores!r} has no storage kind")
    if scale not in KINDS:
        raise SystemExit(f"scale {scale!r} has no storage kind")

    # The nvcc the Build Server checked against this package's declared version.
    compiler = request.toolchain.require_cuda().nvcc
    command = [
        compiler,
        "-cubin",
        f"-arch={request.target}",
        "-O3",
        "-std=c++20",
        "--threads=1",
        f"-DUNISERVE_ITERATIONS={iterations}",
        f"-DUNISERVE_TOP={butterfly}",
        # The descriptor's own numbers, so the kernel can refuse a call that
        # arrives with a different expert count or a wider top-k: `experts` and
        # `top` are launch arguments, and a cubin has no error channel, so the
        # kernel traps instead of reducing over part of the vocabulary.
        f"-DUNISERVE_EXPERTS={experts}",
        f"-DUNISERVE_MAX_TOP={top}",
        "-o",
        str(out / "module.cubin"),
        str(package / "src" / "topk_softmax.cu"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}")

    # The kernel routes one token per warp of a four-warp block in registers, so
    # its metadata is the fixed shape the operator's launch checks.
    metadata = {
        "name": "uniserve_topk_softmax",
        "num_warps": WARPS_PER_BLOCK,
        "num_ctas": 1,
        "shared": 0,
        "launch_pdl": False,
        "launch_cooperative_grid": False,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        # The parameters the kernel takes at launch, in order and with their
        # types, so `ops` can compare the operator's packed words against them:
        # this cubin's list is hand-written on both sides and one word of it is
        # optional, which is the kind that drifts. The last two are the scale,
        # a pointer even when the call has none (the operator passes a null
        # one), and its storage kind.
        "signature": {
            "scores": "*" + POINTERS[scores],
            "score_kind": "i32",
            "row_stride": "i64",
            "rows": "i32",
            "experts": "i32",
            "top": "i32",
            "renormalize": "i32",
            # A pointer even when the call has no scale: the operator passes a
            # null one, and a pointer slot filled with a scalar is exactly what
            # `ops` refuses by name.
            "scale": "*" + POINTERS.get(scale, "fp32"),
            "scale_kind": "i32",
            "ids": "*i32",
            "weights": "*fp32",
            # The logical-to-physical expert map, a pointer the operator passes
            # null without one.
            "map": "*i32",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # The constants this cubin was built with, beside it: a check compares them
    # with its own flags and refuses a module built for another descriptor
    # before it allocates anything, which is the only way to observe the guard
    # in the kernel without a device. Kept out of `metadata.json`, whose fields
    # the loader reads as the launch's own.
    (out / "baked.json").write_text(
        json.dumps(
            {
                "experts": experts,
                "top": top,
                "iterations": iterations,
                "butterfly": butterfly,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "descriptor": request.descriptor,
                "target": str(request.target),
                "symbol": metadata["name"],
                "iterations": iterations,
                "butterfly": butterfly,
                "experts": experts,
                "max_top": top,
            }
        )
    )
