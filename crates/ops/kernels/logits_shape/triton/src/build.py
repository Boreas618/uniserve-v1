"""The sampler_shape package's compile script: one descriptor -> one module.

A descriptor is the logits dtype, the number of extra tokens a row counts
besides its slot's history (a verification row's draft tokens; none in
ordinary decoding) and whether the rows are divided by their temperatures. The
vocabulary and the block width are runtime values, so a new vocabulary or row
count reuses the module.
"""

import _compile
import sampler_shape as kernels

#: The logits dtypes a row may hold; every row is read and penalized in FP32.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one sampler_shape descriptor."""
    return {
        "kernel": kernels._shape_kernel,
        "signature": {
            "logits_ptr": "*" + DTYPES[fields["dtype"]],
            "counts_ptr": "*i32",
            "slots_ptr": "*i64",
            "params_ptr": "*fp32",
            "extra_ptr": "*i64",
            "out_ptr": "*fp32",
            "vocab": "i32",
            "logits_stride": "i64",
            "counts_stride": "i64",
            "params_stride": "i64",
            "extra_stride": "i64",
            "out_stride": "i64",
            "block": "constexpr",
            "extras": "constexpr",
            "divide": "constexpr",
        },
        "constexprs": {
            "block": int(parameters["block"]),
            "extras": int(fields["extras"]),
            "divide": fields["divide"] == "true",
        },
        "options": {"num_warps": int(parameters["warps"])},
    }


if __name__ == "__main__":
    _compile.main(spec)
