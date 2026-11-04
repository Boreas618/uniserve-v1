"""The candidate_order package's compile script: one descriptor -> one module.

A descriptor is the candidate value type, the id type and the dependent-launch flag. Neither the row
count nor the candidate count is a descriptor value: the grid names the rows, and the launch hands the
width to the program, which walks the row in tiles of the package's block.
"""

import _compile
import candidate_order as kernels

#: The candidate value: probabilities or shaped logits, permuted unchanged.
VALUES = {"f32": "fp32"}

#: The candidate id, which the permutation compares and moves unchanged.
IDS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one candidate_order descriptor."""
    block = int(parameters["block"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    ids = "*" + IDS[fields["ids"]]
    return {
        "kernel": kernels._candidate_order_kernel,
        "signature": {
            "values_ptr": "*" + VALUES[fields["values"]],
            "ids_ptr": ids,
            "ordered_values_ptr": "*" + VALUES[fields["values"]],
            "ordered_ids_ptr": ids,
            "rows": "i64",
            "width": "i32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": block},
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
