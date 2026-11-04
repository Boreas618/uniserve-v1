"""The expert_gather package's compile script: one descriptor -> one module.

A descriptor is the expert count the permutation covers and the top-k width.
The token count stays a runtime value through ``entries``, which the launch
computes as ``tokens · top``.
"""

import _compile
import expert_gather as kernels


def spec(fields, parameters, target):
    """The compilation of one expert_gather descriptor."""
    return {
        "kernel": kernels._expert_gather_kernel,
        "signature": {
            "ids_ptr": "*i32",
            "map_ptr": "*i32",
            "out_ptr": "*i32",
            "entries": "i64",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": int(parameters["block"])},
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
