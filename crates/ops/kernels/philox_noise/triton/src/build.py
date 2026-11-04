"""The philox_noise package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The draw's kind is baked in: a normal draw takes one counter per element pair and stores two values,
a uniform draw takes one counter per element, so the two read different words from the same counter
and are separate programs. So is where the key comes from: per row of the buffer, or one launch word
for the whole flattened buffer, which addresses the same draw differently and is a second program.
The row count, the element count, the stage, the draw index and the scalar key are launch words.
"""

import _compile
import philox_noise as kernels

#: The dtypes the drawn noise may have: the draw is computed in FP32 and stored.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one philox_noise descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._philox_noise_kernel,
        "signature": {
            "out_ptr": dtype,
            "keys_ptr": "*i64",
            "elements": "i64",
            "stage": "i32",
            "draw": "i32",
            "key": "i64",
            "BLOCK": "constexpr",
            "NORMAL": "constexpr",
            "ROWS": "constexpr",
        },
        "constexprs": {
            "BLOCK": block,
            "NORMAL": fields["kind"] == "normal",
            # The descriptor names the scalar form; the per-row form is the default and prints
            # nothing, which is what the key's other program always looked like.
            "ROWS": fields.get("source", "rows") == "rows",
        },
        "options": {"num_warps": warps_of(block)},
    }


if __name__ == "__main__":
    _compile.main(spec)
