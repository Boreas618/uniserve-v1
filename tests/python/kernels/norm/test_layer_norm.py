"""The layer_norm program against torch's own arithmetic, every row.

The operator normalizes each row by its own mean and variance and then scales
and shifts it per channel. The row width is a **runtime** extent of this
program and the width a module is built for is its tile, so the cases are
chosen for what a masked row can get wrong: a width far below one tile (the
mask and the reduction over it), widths that straddle a tile with shapes a
deployment uses (1152 and 5120 of 2048 and 8192), and a row whose values are
all equal, where the variance is exactly zero and ``eps`` alone decides the
inverse.

The reference is torch in FP64 over the row, mean, biased variance,
``(x - mean) rsqrt(var + eps) weight + bias``, and the bound the operator's
(`LayerNorm::tolerance`), ``rounding(2 columns, F32, out)``: two sums over the
row in FP32 and one rounding to the output type.

A declared bound with no absolute term is hard for a normalized output to meet
elementwise: an element whose exact value is near zero must agree relatively,
and a row of unit normal values holds a few such elements. A failure states
the count under the absolute floor the operand scale implies
(``u_out max(|weight| + |bias|)``) beside the declared count; the declared one
decides, since the tolerance is the operator's to change.

A hand-computed case holds the program to literals, which a mistake the kernel
and the reference share (a row read as a column, a dropped ``eps``) cannot
pass.
"""

import pytest
import torch

from tests.python.kernels.norm.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    gamma,
    interpreter_store,
    largest_relative,
    outside,
)

pytestmark = pytest.mark.unit

#: `LayerNorm::tolerance`'s reductions over the row: the mean and the
#: variance.
REDUCTIONS = 2

EPS = 1e-5

#: The cases, cheapest and most diverse first: the rows, the row width, the
#: program's tile, whether the layer has a bias, whether the first row is one
#: repeated value, and the seed.
CASES = {
    "the fixture's width, an audio encoder's row of 1152 in a 2048 tile": (
        4,
        1152,
        2048,
        True,
        True,
        1,
    ),
    "the fixture's width without a bias": (4, 1152, 2048, False, True, 1),
    "a row far shorter than the tile, almost every lane masked": (
        3,
        100,
        2048,
        True,
        False,
        2,
    ),
    "a width that straddles its tile, 5000 of 8192": (
        2,
        5000,
        8192,
        True,
        False,
        3,
    ),
    "a deployment's 5120 of 8192, without a bias": (
        4,
        5120,
        8192,
        False,
        True,
        1,
    ),
}

#: The value a flat row repeats: its variance is zero.
FLAT = 2.5

#: Two rows of four with fixed operands and their values computed by hand from
#: `(x - mean) rsqrt(var + eps) weight + bias` (means 2.5 and 0.375, variances
#: 1.25 and 1.171875).
HAND = {
    "x": [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]],
    "weight": [1.0, 0.5, -1.0, 2.0],
    "bias": [0.0, 1.0, 0.0, -1.0],
    "expect": [
        [
            -1.3416354199689269,
            0.7763940966718454,
            -0.447211806656309,
            1.6832708399378538,
        ],
        [
            0.11546956116884853,
            0.3649174135713331,
            -1.5011042951950309,
            -1.6928173670130913,
        ],
    ],
}

#: The tile the hand-computed rows are normalized in.
HAND_TILE = 2048


def bound(columns, dtype):
    """`LayerNorm::tolerance`: `rounding(2 columns, f32, dtype)`'s relative
    factor.
    """  # noqa: D205
    return 2.0 * gamma(REDUCTIONS * columns) + UNIT_ROUNDOFF[dtype]


def launch(kernels, x, weight, bias, out, tile):
    """One launch, one program per row; a layer without a bias passes the
    weight's address for it.
    """  # noqa: D205
    rows, columns = x.shape
    kernels._layer_norm_kernel[(rows,)](
        x,
        weight,
        bias if bias is not None else weight,
        out,
        columns,
        EPS,
        1 if bias is not None else 0,
        0,
        BLOCK=tile,
        HAS_BIAS=bias is not None,
        PDL=False,
    )


def test_the_hand_computed_rows_match_their_literals(
    torch_device, kernel_source
):
    x = torch.tensor(HAND["x"], device=torch_device)
    weight = torch.tensor(HAND["weight"], device=torch_device)
    bias = torch.tensor(HAND["bias"], device=torch_device)
    out = torch.full_like(x, float("nan"))

    launch(kernel_source("layer_norm"), x, weight, bias, out, HAND_TILE)

    expect = torch.tensor(HAND["expect"]).double()
    difference = (out.cpu().double() - expect).abs()
    relative = bound(x.shape[1], "f32")
    missed = outside(difference, relative * expect.abs())
    assert missed == 0, (
        f"hand-computed 2 x 4 of tile {HAND_TILE}, f32: {missed} outside the "
        f"bound, max absolute {difference.max().item():.3e}, rtol "
        f"{relative:.3e}"
    )


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("case", CASES)
def test_every_row_matches_fp64_within_the_operators_bound(
    case, dtype, torch_device, interpreted, kernel_source
):
    rows, columns, tile, has_bias, flat_row, seed = CASES[case]
    element = TORCH_TYPES[dtype]
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, columns, generator=generator)
    if flat_row:
        x[0, :] = FLAT
    x = x.to(element)
    weight = torch.randn(columns, generator=generator).to(element)
    bias = (
        torch.randn(columns, generator=generator).to(element)
        if has_bias
        else None
    )
    out = torch.full(
        (rows, columns), float("nan"), dtype=element, device=torch_device
    )

    launch(
        kernel_source("layer_norm"),
        x.to(torch_device),
        weight.to(torch_device),
        bias.to(torch_device) if bias is not None else None,
        out,
        tile,
    )

    exact = x.double()
    mean = exact.mean(dim=-1, keepdim=True)
    variance = exact.var(dim=-1, unbiased=False, keepdim=True)
    expected = (exact - mean) / torch.sqrt(variance + EPS) * weight.double()
    scale = weight.double().abs()
    if bias is not None:
        expected = expected + bias.double()
        scale = scale + bias.double().abs()
    expected = interpreter_store(expected, dtype, interpreted)
    difference = (out.cpu().double() - expected).abs()
    relative = bound(columns, dtype)
    floor = UNIT_ROUNDOFF[dtype] * scale.max().item()
    declared = outside(difference, relative * expected.abs())
    floored = outside(difference, relative * expected.abs() + floor)
    assert declared == 0, (
        f"{case}, {rows} x {columns} of tile {tile}, {dtype}: {declared} "
        f"elements outside the declared bound ({floored} with the "
        f"operands' absolute floor {floor:.3e}), max absolute "
        f"{difference.nan_to_num(nan=float('inf')).max().item():.3e}, max "
        f"relative {largest_relative(difference, expected):.3e}, rtol "
        f"{relative:.3e}"
    )
