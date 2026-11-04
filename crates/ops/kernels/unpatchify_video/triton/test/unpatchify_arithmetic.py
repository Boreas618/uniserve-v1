#!/usr/bin/env python3
"""Execute the unpatchify_video kernel on the CPU and check it against the operator's formulas.

Triton's interpreter runs the kernel body with numpy-backed tensors, one program at a time, so the
mapping can be checked here: it is index arithmetic plus, in the affine form, one multiply-add per
voxel.

    token (t, y, x), element c·pf·ph·pw + pt·ph·pw + ph_i·pw + pw_i
      -> out[c, t·pf + pt, y·ph + ph_i, x·pw + pw_i]
    with affine:  out = value · scale[c] + bias[c]

The reference is that mapping written out in torch, once per variant. The check covers both forms
and two patch shapes, and a block narrower than a raster row so the lane slices are exercised.

    TRITON_INTERPRET=1 python unpatchify_arithmetic.py
"""

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import unpatchify_video as kernels  # noqa: E402


def reference(tokens, channels, frames, height, width, patch, affine_scale=None, affine_bias=None):
    """The operator's mapping, in torch, from the formulas in `crates/ops/src/unpatchify.rs`."""
    patch_frames, patch_height, patch_width = patch
    raster = torch.zeros(
        channels,
        frames * patch_frames,
        height * patch_height,
        width * patch_width,
        dtype=torch.float32,
    )
    for frame in range(frames):
        for row in range(height):
            for column in range(width):
                token = (frame * height + row) * width + column
                for channel in range(channels):
                    for inner_frame in range(patch_frames):
                        for inner_row in range(patch_height):
                            for inner_column in range(patch_width):
                                value = tokens[
                                    token,
                                    ((channel * patch_frames + inner_frame) * patch_height
                                     + inner_row) * patch_width + inner_column,
                                ]
                                if affine_scale is not None:
                                    value = value * affine_scale[channel] + affine_bias[channel]
                                raster[
                                    channel,
                                    frame * patch_frames + inner_frame,
                                    row * patch_height + inner_row,
                                    column * patch_width + inner_column,
                                ] = value
    return raster


def one(channels, frames, height, width, patch, affine, block):
    """Check one geometry, in the form the flag names."""
    patch_frames, patch_height, patch_width = patch
    tokens_count = frames * height * width
    token_width = channels * patch_frames * patch_height * patch_width
    tokens = torch.rand(tokens_count, token_width, dtype=torch.float32)
    raster = torch.full(
        (
            channels,
            frames * patch_frames,
            height * patch_height,
            width * patch_width,
        ),
        float("nan"),
    )
    grid = ((raster.numel() + block - 1) // block, 1, 1)
    if affine:
        scale = torch.rand(channels, dtype=torch.float32) + 0.5
        bias = torch.rand(channels, dtype=torch.float32) - 0.5
        kernels._unpatchify_video_affine_kernel[grid](
            tokens,
            scale,
            bias,
            raster,
            channels,
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            0,
            0,
            BLOCK=block,
        )
        expected = reference(tokens, channels, frames, height, width, patch, scale, bias)
    else:
        kernels._unpatchify_video_kernel[grid](
            tokens,
            raster,
            channels,
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            0,
            0,
            BLOCK=block,
        )
        expected = reference(tokens, channels, frames, height, width, patch)

    error = (raster - expected).abs().max().item()
    tolerance = 4.0 * torch.finfo(torch.float32).eps * max(1.0, expected.abs().max().item())
    print(
        f"channels={channels} frames={frames} {height}×{width} patch={patch} "
        f"affine={affine} block={block}: max |error| = {error:.3e}"
    )
    return not torch.isnan(raster).any() and error <= tolerance


def main() -> int:
    cases = [
        (2, 2, 2, 3, (1, 2, 2), False, 8),
        (2, 2, 2, 3, (1, 2, 2), True, 8),
        (1, 1, 2, 2, (2, 2, 2), False, 16),
        (3, 2, 2, 2, (1, 2, 2), True, 1024),
    ]
    return 0 if all(one(*case) for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
