"""The rgb_bytes package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor dimensions give, and the compilation options.

The element type is compiled in, because it is the load's type and the descriptor
names it. The channels and the block are compiled in as constexprs. Nothing about
the raster is: the frame count, the sides and the strides are launch values, so one
module serves every clip. The output is always ``u8`` and is never in the key.
"""

import _compile
import rgb_bytes as kernels

#: Triton's element type for each element type the operator accepts, as the
#: signature spells a pointer to it.
POINTERS = {"f16": "*fp16", "bf16": "*bf16", "f32": "*fp32"}


def warps_of(block):
    """The warp count one block of bytes needs: a load, three constants and a store."""
    return max(1, min(8, block // 512))


def spec(fields, parameters, target):
    """The compilation of one rgb_bytes descriptor."""
    block = int(fields["block"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._rgb_bytes_kernel,
        "signature": {
            "input": POINTERS[fields["dtype"]],
            "mean": "*fp32",
            "std": "*fp32",
            "out": "*u8",
            "frames": "i64",
            "rows": "i64",
            "columns": "i64",
            "channels": "i32",
            "in0": "i64",
            "in1": "i64",
            "in2": "i64",
            "out0": "i64",
            "out1": "i64",
            "out2": "i64",
            "pdl": "i32",
            "CHANNELS": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "CHANNELS": int(fields["channels"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
