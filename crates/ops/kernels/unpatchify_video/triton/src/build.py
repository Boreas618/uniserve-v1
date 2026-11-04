"""The unpatchify_video package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

Every field of the descriptor is baked in — the operator's key names the grid, the patch and the
affine flag, because a model's raster is its own geometry — and only the launch's block width and
its PDL flag travel as words. The affine form's launch carries two more operands, so its signature
has two more pointers; a non-affine module's does not, and a launch that hands them anyway is
refused by the operator before it reaches the kernel.
"""

import _compile
import unpatchify_video as kernels

#: The element types the tokens, the scale, the bias and the raster may have.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of voxels needs."""
    return 8 if block >= 1024 else 4


def patch_shape(fields):
    """The patch a descriptor's `patch` field names, such as ``"1x2x2"``."""
    frames, height, width = (int(part) for part in fields["patch"].split("x"))
    return frames, height, width


def spec(fields, parameters, target):
    """The compilation of one unpatchify_video descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    affine = fields["affine"] == "true"
    signature = {"tokens_ptr": dtype}
    if affine:
        signature["scale_ptr"] = dtype
        signature["bias_ptr"] = dtype
    signature["out_ptr"] = dtype
    signature.update(
        {
            "channels": "i32",
            "frames": "i32",
            "height": "i32",
            "width": "i32",
            "patch_frames": "i32",
            "patch_height": "i32",
            "patch_width": "i32",
            "block": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
        }
    )
    return {
        # The affine form is its own kernel because its launch carries two more operands; the
        # operator's descriptor says which form a compilation is.
        "kernel": (
            kernels._unpatchify_video_affine_kernel if affine else kernels._unpatchify_video_kernel
        ),
        "signature": signature,
        "constexprs": {"BLOCK": block},
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
