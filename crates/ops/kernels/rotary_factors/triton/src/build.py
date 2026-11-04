"""The rope_factors package's compile script: one descriptor -> one module.

A descriptor is the position dtype (the kernel reads positions through their
own pointer type), the factor dtype, the number of frequencies per row and the
positions' layout. Factors are FP32 phases rounded once into their output
dtype, so the phase domain does not depend on the output dtype.
"""

import _compile
import rope_factors as kernels

#: Position dtypes. Rotary positions are token offsets, which the caller
#: passes as int32 or int64.
POSITIONS = {"i32": "i32", "i64": "i64"}

#: Factor dtypes: FP32 keeps the FP32 phase, BF16 halves the table.
FACTORS = {"fp32": "fp32", "bf16": "bf16"}

#: Positions are one flat sequence: every stride is the element stride, so the
#: kernel's trailing extents are the empty tuple.
LAYOUTS = {"sequence": ((), (1,))}


def spec(fields, parameters, target):
    """The compilation of one rope_factors descriptor."""
    shape, strides = LAYOUTS[fields["layout"]]
    half_dim = int(fields["half_dim"])
    pdl = _compile.capability(target) >= 100
    return {
        "kernel": kernels._rotary_factors_kernel,
        "signature": {
            "positions": "*" + POSITIONS[fields["positions"]],
            "frequencies": "*fp32",
            "cosine": "*" + FACTORS[fields["factors"]],
            "sine": "*" + FACTORS[fields["factors"]],
            "total": "i64",
            "width": "constexpr",
            "shape": "constexpr",
            "strides": "constexpr",
            "frequency_stride": "constexpr",
            "scale": "constexpr",
            "block": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "width": half_dim,
            "shape": shape,
            "strides": strides,
            "frequency_stride": 1,
            "scale": float(parameters["scale"]),
            "block": int(parameters["block"]),
            "PDL": pdl,
        },
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
