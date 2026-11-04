"""The gather_logits package's compile script: one descriptor -> one module.

A descriptor is the hidden state's dtype and width and the row index dtype. One
program copies one block of one row, so the token and request counts stay
runtime values.
"""

import _compile
import gather_logits as kernels

#: The hidden state's dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}

#: The row index dtype of a request's last position.
ROWS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one gather_logits descriptor."""
    return {
        "kernel": kernels._gather_logits_kernel,
        "signature": {
            "hidden_ptr": "*" + DTYPES[fields["dtype"]],
            "rows_ptr": "*" + ROWS[fields["rows"]],
            "out_ptr": "*" + DTYPES[fields["dtype"]],
            "width": "constexpr",
            "block": "constexpr",
        },
        "constexprs": {
            "width": int(fields["width"]),
            "block": int(parameters["block"]),
        },
        "options": {"num_warps": int(parameters["warps"])},
    }


if __name__ == "__main__":
    _compile.main(spec)
