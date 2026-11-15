"""The vsa_zero_padding program against the operator's masking.

The operator's whole claim is an absence with a boundary: the rows a tile has
past its valid size read as zeros, and the rows inside it are not touched.
Both halves are checked, because a kernel that zeroed the whole tile passes
any check that only looks at the padding, and one that zeroed the wrong block
passes any check that only looks at a tile's first rows. The reference masks a
``[tiles, tile, heads, width]`` view of the rows by the same comparison
(:func:`uniserve_testing.kernels.vsa.operands.zero_padding`), and
the rows are non-zero everywhere, so a row left unzeroed cannot pass as
already zero and a row written by mistake cannot pass as untouched.

The cases are the ones a plausible-but-wrong kernel passes:

- **a tile whose valid size is zero**, so every row block's mask is full;
- **a partial block**, where the valid size ends inside one ``block_rows``
  block, the only place a store holds both kinds of row;
- **a strided row layout**, the row stride wider than ``heads · width``, so an
  address that ignored the stride writes inside the tile and a head that
  ignored the head stride lands on its neighbour;
- **several heads**, which the kernel walks with one mask;
- **``f32`` and ``f16`` next to ``bf16``**, because the zeros are stored in the
  values' own element type.
"""

import pytest
import torch
from uniserve_testing.kernels.vsa import operands

pytestmark = pytest.mark.unit

#: (dtype, valid sizes, heads, width, row stride or None for packed rows),
#: over 64-row tiles in blocks of 16.
CASES = {
    "a partial tile and a full one": ("bf16", [64, 5], 4, 64, None),
    "an empty tile": ("bf16", [0, 40, 64], 2, 64, None),
    "a partial block in f32": ("f32", [17], 3, 32, None),
    "strided rows in f16": ("f16", [33, 64], 8, 128, 1200),
    "one valid row and one padded row": ("bf16", [1, 63], 1, 16, None),
}
TILE = 64
BLOCK = 16

#: The seed every case draws its rows from.
SEED = 13


def launch(kernels, rows, sizes, tile, block):
    """One launch over every row block of every tile of `rows`."""
    heads, width = rows.shape[1:]
    kernels._zero_tile_padding_kernel[(sizes.numel(), tile // block)](
        rows,
        sizes,
        rows.stride(0),
        rows.stride(1),
        heads=heads,
        tile_rows=tile,
        width=width,
        block_rows=block,
    )


@pytest.mark.parametrize("case", CASES)
def test_the_padding_is_zeroed_and_the_valid_rows_untouched(
    case, torch_device, kernel_source
):
    dtype, sizes, heads, width, row_stride = CASES[case]
    element = operands.DTYPES[dtype]
    stride = heads * width if row_stride is None else row_stride
    tiles = len(sizes)
    count = tiles * TILE
    torch.manual_seed(SEED)
    flat = (torch.randn(count * stride) * 2.0 + 3.0).to(element)
    view = torch.as_strided(
        flat.to(torch_device), (count, heads, width), (stride, width, 1)
    )
    before = view.clone().cpu()
    table = torch.tensor(sizes, dtype=torch.int32)

    launch(
        kernel_source("vsa_zero_padding"),
        view,
        table.to(torch_device),
        TILE,
        BLOCK,
    )

    got = view.cpu()
    expected = before.clone()
    operands.zero_padding(expected, table, TILE)
    label = f"{dtype} sizes {sizes} heads {heads} width {width} stride {stride}"
    differing = got != expected
    if differing.any():
        difference = (got.double() - expected.double()).abs()
        position = int(difference.reshape(-1).argmax())
        pytest.fail(
            f"{label}: {int(differing.sum())} of {got.numel()} elements "
            f"differ; worst at {position}: {float(got.reshape(-1)[position])} "
            f"vs {float(expected.reshape(-1)[position])}"
        )
    # The rows inside the valid size are bit for bit what they were, stated
    # apart so a failure names the half that broke.
    live = torch.arange(TILE) < table[:, None]
    inside = before.reshape(tiles, TILE, heads, width)[live]
    after = got.reshape(tiles, TILE, heads, width)[live]
    assert torch.equal(inside, after), f"{label}: a live row was written"
