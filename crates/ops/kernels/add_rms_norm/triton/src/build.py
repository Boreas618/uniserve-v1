"""The add_rms_norm package's compile script: one descriptor -> one module.

Both outputs derive from the unrounded FP32 residual sum, so a descriptor is
the row's dtype and width only. The row block is the next power of two at
least the width, and rows at least 2048 wide use 8 warps, as the launch of the
kernel this package replaces computed.
"""

import _compile
import add_rms_norm as kernels

#: The floating dtypes of the two inputs, the weight and both outputs.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def block_of(width):
    """The row block and warp count of a row ``width`` wide."""
    block = 1 << (width - 1).bit_length()
    return block, 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one add_rms_norm descriptor."""
    width = int(fields["width"])
    block, warps = block_of(width)
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = _compile.capability(target) >= 100
    return {
        "kernel": kernels._add_rms_norm_kernel,
        "signature": {
            "x_ptr": dtype,
            "r_ptr": dtype,
            "w_ptr": dtype,
            "y_ptr": dtype,
            "c_ptr": dtype,
            "n_cols": "constexpr",
            "eps": "constexpr",
            "block": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "n_cols": width,
            "eps": float(parameters["eps"]),
            "block": block,
            "PDL": pdl,
        },
        "options": {"num_warps": warps, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
