"""The unpatchify_video programs against the operator's unpacking of tokens.

The mapping is index arithmetic plus, in the affine form, one multiply-add per
voxel (``crates/ops/src/unpatchify.rs``):

```text
token (t, y, x), element c·pf·ph·pw + pt·ph·pw + ph_i·pw + pw_i
  -> out[c, t·pf + pt, y·ph + ph_i, x·pw + pw_i]
with affine:  out = value · scale[c] + bias[c]
```

The reference is that mapping written out in torch. The cases cover both forms
and two patch shapes, a block narrower than a raster row so the lane slices
run, and a raster placed inside a larger clip, ``margin`` positions from its
edges as a decoder's tile sits inside its clip: every position outside the
raster stays NaN, so a write through a wrong stride reads as an error, and no
position inside it may stay NaN. The values are compared within four FP32
units of the larger of one and the raster's largest magnitude.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: (channels, frames, height, width, patch, affine, block, margin).
CASES = [
    pytest.param(2, 2, 2, 3, (1, 2, 2), False, 8, 0, id="plain, block 8"),
    pytest.param(2, 2, 2, 3, (1, 2, 2), True, 8, 0, id="affine, block 8"),
    pytest.param(1, 1, 2, 2, (2, 2, 2), False, 16, 0, id="a 2x2x2 patch"),
    pytest.param(3, 2, 2, 2, (1, 2, 2), True, 1024, 0, id="affine, one block"),
    pytest.param(3, 2, 2, 3, (1, 2, 2), False, 8, 3, id="plain, in a clip"),
    pytest.param(3, 2, 2, 3, (1, 2, 2), True, 16, 5, id="affine, in a clip"),
]

#: The seed every case draws its tokens and coefficients from.
SEED = 41


def launch(kernels, tokens, raster, grid_shape, patch, block, affine=None):
    """One launch writing every voxel of `raster` through its three outer
    strides; `affine` is the per-channel `(scale, bias)` of the affine form.
    """  # noqa: D205
    channels, frames, height, width = grid_shape
    words = (
        channels,
        frames,
        height,
        width,
        *patch,
        *raster.stride()[:3],
        block,
    )
    grid = (-(-raster.numel() // block), 1, 1)
    num_warps = 8 if block >= 1024 else 4
    if affine is None:
        kernels._unpatchify_video_kernel[grid](
            tokens, raster, *words, BLOCK=block, num_warps=num_warps
        )
    else:
        kernels._unpatchify_video_affine_kernel[grid](
            tokens, *affine, raster, *words, BLOCK=block, num_warps=num_warps
        )


def reference(tokens, grid_shape, patch, affine=None):
    """The operator's mapping, in torch, from the formulas in
    ``crates/ops/src/unpatchify.rs``.
    """  # noqa: D205
    channels, frames, height, width = grid_shape
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
                                element = (
                                    (channel * patch_frames + inner_frame)
                                    * patch_height
                                    + inner_row
                                ) * patch_width + inner_column
                                value = tokens[token, element]
                                if affine is not None:
                                    scale, bias = affine
                                    value = (
                                        value * scale[channel] + bias[channel]
                                    )
                                raster[
                                    channel,
                                    frame * patch_frames + inner_frame,
                                    row * patch_height + inner_row,
                                    column * patch_width + inner_column,
                                ] = value
    return raster


@pytest.mark.parametrize(
    (
        "channels",
        "frames",
        "height",
        "width",
        "patch",
        "affine",
        "block",
        "margin",
    ),
    CASES,
)
def test_the_raster_is_the_operators_unpacking(
    channels,
    frames,
    height,
    width,
    patch,
    affine,
    block,
    margin,
    torch_device,
    kernel_source,
):
    torch.manual_seed(SEED)
    patch_frames, patch_height, patch_width = patch
    grid_shape = (channels, frames, height, width)
    token_width = channels * patch_frames * patch_height * patch_width
    tokens = torch.rand(frames * height * width, token_width)
    coefficients = (
        (torch.rand(channels) + 0.5, torch.rand(channels) - 0.5)
        if affine
        else None
    )
    rows, columns = height * patch_height, width * patch_width
    clip = torch.full(
        (
            channels,
            frames * patch_frames,
            rows + 2 * margin,
            columns + 2 * margin,
        ),
        float("nan"),
        device=torch_device,
    )
    raster = clip[:, :, margin : margin + rows, margin : margin + columns]

    launch(
        kernel_source("unpatchify_video"),
        tokens.to(torch_device),
        raster,
        grid_shape,
        patch,
        block,
        None
        if coefficients is None
        else tuple(part.to(torch_device) for part in coefficients),
    )

    expected = reference(tokens, grid_shape, patch, coefficients)
    got = raster.cpu()
    label = (
        f"channels={channels} frames={frames} {height}x{width} patch={patch} "
        f"affine={affine} block={block} margin={margin}"
    )
    unwritten = int(torch.isnan(got).sum())
    assert unwritten == 0, f"{label}: {unwritten} raster positions unwritten"
    written = torch.zeros(clip.shape, dtype=torch.bool)
    written[:, :, margin : margin + rows, margin : margin + columns] = True
    outside = clip.cpu()[~written]
    stray = int((~torch.isnan(outside)).sum())
    assert stray == 0, f"{label}: {stray} positions outside the raster written"
    error = (got - expected).abs().max().item()
    tolerance = (
        4.0
        * torch.finfo(torch.float32).eps
        * max(1.0, expected.abs().max().item())
    )
    assert error <= tolerance, (
        f"{label}: max |error| = {error:.3e}, tolerance {tolerance:.3e}"
    )
