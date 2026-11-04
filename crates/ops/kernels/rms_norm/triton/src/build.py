"""The rms_norm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature, the compile-time values the
descriptor dimension and the package parameters give, and the compilation
options. The Program Server's Triton source drives ``_compile`` with it.
"""

import _compile
import rms_norm as kernels

#: The floating dtypes a row, its weight and its output may have, as Triton
#: spells them in an argument signature.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def block_of(width):
    """The row block and warp count of a row ``width`` wide."""
    block = 1 << (width - 1).bit_length()
    return block, 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one rms_norm descriptor."""
    width = int(fields["width"])
    block, warps = block_of(width)
    if fields["layout"] != "contiguous":
        raise SystemExit(f"layout {fields['layout']} has no kernel in this package")
    dtype = "*" + DTYPES[fields["dtype"]]
    # Contiguous rows merge their leading axes into one row axis whose stride
    # is the width, so the kernel's row geometry is the empty tuple of trailing
    # extents and the width stride, exactly as its launch computed it.
    strides = (width,)
    pdl = _compile.capability(target) >= 100
    return {
        "kernel": kernels._rms_norm_kernel,
        "signature": {
            "x_ptr": dtype,
            "w_ptr": dtype,
            "y_ptr": dtype,
            "x_shape": "constexpr",
            "x_strides": "constexpr",
            "y_shape": "constexpr",
            "y_strides": "constexpr",
            "n_cols": "constexpr",
            "eps": "constexpr",
            "block": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "x_shape": (),
            "x_strides": strides,
            "y_shape": (),
            "y_strides": strides,
            "n_cols": width,
            "eps": float(parameters["eps"]),
            "block": block,
            "PDL": pdl,
        },
        "options": {"num_warps": warps, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
