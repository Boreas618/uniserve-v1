"""The sigmoid_gated_add package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor gives, and the compilation options. The
Build Server's Triton source drives ``uniserve_build_server.triton`` with it.
"""

import sigmoid_gated_add as kernels

#: The floating dtypes of the target, the update and the gate, as Triton spells them in an
#: argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one sigmoid_gated_add descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        # The target and the update are arena rows on sixteen-byte bounds; the gate column is not.
        "aligned": ["target_ptr", "update_ptr"]
        + (["columns"] if int(fields["columns"]) % 16 == 0 else []),
        "kernel": kernels._sigmoid_gated_add_kernel,
        "signature": {
            "target_ptr": dtype,
            "update_ptr": dtype,
            "gate_ptr": dtype,
            "columns": "i32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": int(fields["block"])},
        "options": {"num_warps": 4},
    }
