"""The patch program against the operator's packing of a latent into rows.

The program is a permutation, compared with the operator's own definition
(``crates/ops/src/patch.rs``) written out in torch:

```text
x [channels, frames, height, width] -> r [rows, channels · 4]
row    = ((f · height/2) + i) · (width/2) + j,  column = ((c · 2) + a) · 2 + b
r[row, column] = x[c, f, 2·i + a, 2·j + b]
```

One direction, because the operator has one: the inverse raster is
``unpatchify_video``, whose own check covers it. The geometries are small (one
and two frames, a single patch and a 4x6 latent of three channels) and the
block narrower than a row, so the column slices run: it is the grid, the two
decodings and the store under test. A permutation writes every destination
element, so a sentinel left in the output is a hole in the mapping.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: The latent's channels and the columns one program writes: narrower than a
#: packed row, so the column slices run.
CHANNELS = 3
BLOCK = 4

#: (frames, height, width) of every geometry.
GEOMETRIES = [
    (frames, height, width)
    for frames in (1, 2)
    for height, width in ((2, 2), (4, 6))
]

#: What the output holds before the launch: no latent value equals it.
SENTINEL = -1.0


def launch(kernels, x, out, frames, rows, channels, height, width, block):
    """One launch over every packed row and column slice of `out`."""
    kernels._patch_kernel[(rows, -(-(channels * 4) // block), 1)](
        x,
        out,
        frames,
        rows,
        channels,
        height,
        width,
        block,
        0,
        BLOCK=block,
    )


def reference(latent, rows, channels, height, width):
    """The operator's mapping, in torch, from the formulas in
    ``crates/ops/src/patch.rs``.
    """  # noqa: D205
    stacked = torch.zeros(rows, channels * 4, dtype=latent.dtype)
    per_frame = (height // 2) * (width // 2)
    for row in range(rows):
        frame, within = divmod(row, per_frame)
        patch_row, patch_column = divmod(within, width // 2)
        for column in range(channels * 4):
            channel, inner = divmod(column, 4)
            inner_row, inner_column = divmod(inner, 2)
            stacked[row, column] = latent[
                channel,
                frame,
                2 * patch_row + inner_row,
                2 * patch_column + inner_column,
            ]
    return stacked


@pytest.mark.parametrize(
    ("frames", "height", "width"),
    GEOMETRIES,
    ids=[f"{f}x{h}x{w}" for f, h, w in GEOMETRIES],
)
def test_the_packing_is_the_operators_permutation(
    frames, height, width, torch_device, kernel_source
):
    rows = frames * (height // 2) * (width // 2)
    latent = torch.arange(
        CHANNELS * frames * height * width, dtype=torch.float32
    ).reshape(CHANNELS, frames, height, width)
    out = torch.full((rows, CHANNELS * 4), SENTINEL, device=torch_device)

    launch(
        kernel_source("patch"),
        latent.to(torch_device),
        out,
        frames,
        rows,
        CHANNELS,
        height,
        width,
        BLOCK,
    )

    got = out.cpu()
    expected = reference(latent, rows, CHANNELS, height, width)
    where = f"{CHANNELS}x{frames}x{height}x{width}, block {BLOCK}"
    unwritten = int((got == SENTINEL).sum())
    assert unwritten == 0, f"{where}: {unwritten} elements left unwritten"
    differing = (got != expected).nonzero()
    assert differing.numel() == 0, (
        f"{where}: {differing.size(0)} of {got.numel()} elements differ; "
        f"first at {tuple(differing[0].tolist())}"
    )
