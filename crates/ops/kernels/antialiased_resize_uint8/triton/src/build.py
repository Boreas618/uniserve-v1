"""The antialiased_resize_uint8 package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The interpolation and the channel count are baked in: the mode selects the filter's support and its
formula, and the channel count is the plane's depth. The extents, the two scales and the flag are
launch words, which is why the module derives its taps and weights on device.
"""

import _compile
import antialiased_resize_uint8 as kernels


def warps_of(block):
    """The warp count one block of output rows needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one antialiased_resize_uint8 descriptor."""
    block = int(fields["block"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._antialiased_resize_uint8_kernel,
        "signature": {
            "x_ptr": "*u8",
            "out_ptr": "*u8",
            "rows": "i64",
            "channels": "i32",
            "height": "i32",
            "width": "i32",
            "out_height": "i32",
            "out_width": "i32",
            "scale_h": "fp32",
            "scale_w": "fp32",
            "pdl": "i32",
            "CHANNELS": "constexpr",
            "BICUBIC": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "CHANNELS": int(fields["channels"]),
            "BICUBIC": fields["mode"] == "bicubic",
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
