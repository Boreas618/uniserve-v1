"""The layer_norm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The row width, the lane tile that covers it, whether the layer has a bias and whether the module
issues the dependent launch are baked in; the row count, the epsilon and the flags stay launch
words.
"""

import _compile
import layer_norm as kernels

#: The floating dtypes a row, its affine parameters and its output may have.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(tile):
    """The warp count one row tile needs."""
    return 8 if tile >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one layer_norm descriptor."""
    tile = int(fields["tile"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._layer_norm_kernel,
        "signature": {
            "x_ptr": dtype,
            "weight_ptr": dtype,
            "bias_ptr": dtype,
            "out_ptr": dtype,
            "columns": "i32",
            "eps": "fp32",
            "has_bias": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "HAS_BIAS": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "BLOCK": tile,
            "HAS_BIAS": fields["bias"] == "true",
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(tile), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
