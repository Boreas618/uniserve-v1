"""The patch package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The direction is baked in as a constexpr: the two read and write the two views the other way
round, and the launch's grid names the packed rows either way, so the modes are two programs.
Programmatic dependent launch is a compilation option of the module, so a descriptor that asks
for it compiles one that declares it in its metadata.
"""

import _compile
import patch as kernels

#: The element types the latent and the rows may have; the move is exact in each of them.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of columns needs; the operator's block is 1024 elements."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one patch descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._patch_kernel,
        "signature": {
            "x_ptr": dtype,
            "out_ptr": dtype,
            "frames": "i32",
            "rows": "i32",
            "channels": "i32",
            "height": "i32",
            "width": "i32",
            "block": "i32",
            "pdl": "i32",
            "PATCHIFY": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "PATCHIFY": fields["mode"] == "patchify",
            "BLOCK": block,
        },
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
