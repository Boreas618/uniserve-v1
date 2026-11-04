"""The vocab_unpad package's compile script: one descriptor -> one module.

A descriptor is the gathered logits' dtype, the vocabulary, its padded width, the
program's block and the dependent-launch flag. The row count stays a runtime
value.
"""

import _compile
import vocab_unpad as kernels

#: The gathered logits' dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one vocab_unpad descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    # The output's element type is its own field: the head's product is in the model's dtype and the
    # sampler's candidate buffer is FP32, so this program is one of the places the two meet.
    output = "*" + DTYPES[fields["output"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._vocab_unpad_kernel,
        "signature": {
            "gathered_ptr": dtype,
            "out_ptr": output,
            "rows": "i64",
            "padded": "i32",
            "vocabulary": "i32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": int(fields["block"])},
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
