"""The sampler_logprobs package's compile script: one descriptor -> one module.

A descriptor is the logits dtype. The vocabulary and every stride are runtime
values, so one module serves every row count and vocabulary.
"""

import _compile
import sampler_logprobs as kernels

#: The logits dtypes the kernel reads; it scores in FP32 in every case.
DTYPES = {"fp32": "fp32", "bf16": "bf16", "fp16": "fp16"}


def spec(fields, parameters, target):
    """The compilation of one sampler_logprobs descriptor."""
    return {
        "kernel": kernels._logprobs_kernel,
        "signature": {
            "logits_ptr": "*" + DTYPES[fields["dtype"]],
            "rows_ptr": "*i32",
            "tokens_ptr": "*i64",
            "scores_ptr": "*fp32",
            "ranks_ptr": "*i32",
            "vocab": "i32",
            "logits_stride": "i64",
            "rows_stride": "i64",
            "tokens_stride": "i64",
            "block": "constexpr",
        },
        "constexprs": {"block": int(parameters["block"])},
        "options": {"num_warps": int(parameters["warps"])},
    }


if __name__ == "__main__":
    _compile.main(spec)
