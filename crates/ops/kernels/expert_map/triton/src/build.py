"""The expert_map package's compile script: one descriptor -> one module.

A descriptor is the expert count the permutation covers and the top-k width.
One program maps one token's picks, so the token count stays a runtime value.
"""

import _compile
import expert_map as kernels


def spec(fields, parameters, target):
    """The compilation of one expert_map descriptor."""
    top = int(fields["top"])
    return {
        "kernel": kernels._expert_map_kernel,
        "signature": {
            "ids_ptr": "*i32",
            "map_ptr": "*i32",
            "out_ptr": "*i32",
            "top": "constexpr",
            "block": "constexpr",
        },
        "constexprs": {
            "top": top,
            # One lane per pick, rounded up to a warp-wide tile.
            "block": 1 << max(5, (top - 1).bit_length()),
        },
        "options": {"num_warps": 1},
    }


if __name__ == "__main__":
    _compile.main(spec)
