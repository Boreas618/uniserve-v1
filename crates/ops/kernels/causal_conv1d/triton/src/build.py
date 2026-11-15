"""The causal_conv1d package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor gives, and the compilation options. The
Build Server's Triton source drives ``uniserve_build_server.triton`` with it.

The channel count, the tap count and the channel block are the key's; the token
tile one program walks at a time is this package's.
"""

import causal_conv1d as kernels

#: The floating dtypes of the inputs, the taps, the output and the state banks, as Triton spells
#: them in an argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}

#: Tokens one program convolves at a time: a tile of rows for every tap's loads.
TOKENS = 16


def spec(fields, parameters, target):
    """The compilation of one causal_conv1d descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    width = int(fields["width"])
    if width < 2:
        raise SystemExit(f"a convolution of {width} taps keeps no state")
    return {
        "aligned": ["x_ptr", "out_ptr", "state_ptr", "checkpoint_ptr", "trail_ptr"],
        "kernel": kernels._causal_conv1d_kernel,
        "signature": {
            "x_ptr": dtype,
            "weight_ptr": dtype,
            "out_ptr": dtype,
            "offsets_ptr": "*i32",
            "seq_lens_ptr": "*i32",
            "slots_ptr": "*i32",
            "checkpoints_ptr": "*i32",
            "state_ptr": dtype,
            "checkpoint_ptr": dtype,
            "trail_ptr": dtype,
            "x_row_stride": "i32",
            "trailing": "i32",
            "CHANNELS": "constexpr",
            "WIDTH": "constexpr",
            "KEPT": "constexpr",
            "BLOCK": "constexpr",
            "TOKENS": "constexpr",
        },
        "constexprs": {
            "CHANNELS": int(fields["channels"]),
            "WIDTH": width,
            "KEPT": 1 << (width - 2).bit_length(),
            "BLOCK": int(fields["block"]),
            "TOKENS": TOKENS,
        },
        "options": {"num_warps": 4},
    }
