"""The position_interpolate program against its four-tap sum, and launched
dependently.

The operator resamples a learned position table to another grid, but it
computes **no geometry**: the caller hands four tap indices (i32) and four
FP32 weights per output point, and the program gathers four table rows and
writes their weighted sum::

    out[j, c] = sum_{t < 4} weights[j, t] table[indices[j, t], c]

so the check tests exactly that and nothing about a resampling convention. The
reference is torch doing the same sum in FP64, rounded once into the output
type, and the bound ``Tolerance::dot(4, f32, dtype)`` (`PositionInterpolate::
tolerance`): four weighted products accumulated in FP32 with the output's own
rounding, plus the reference's own rounding, since two correct
implementations may sit one rounding apart.

The table is **monotone and smooth**, which is what a learned position table
is and what the bound assumes: a blend of neighbouring points with positive
weights does not cancel, so the bound's ``sqrt(4)`` correction is the right
size, and the gradient is steep enough per point that a wrong tap lands on a
distinguishably wrong value. Each case is aimed at a way the kernel could be
wrong while looking right:

- 1152 channels in a 1024-lane block, the fixture's geometry, where the last
  program covers 128 lanes and the rest is masked;
- one table point tapped by two output rows at different weights, so a kernel
  that read the wrong row's taps lands on a plausible value;
- a seven-point grid with one row whose taps are in descending order, which a
  gather that sorted them would fail;
- a clamped corner, all four taps one point with weights summing to one,
  which must equal that row;
- weights that do not sum to one: the caller normalizes, so a kernel that
  divided or averaged is wrong here and right everywhere else.

Triton's interpreter rounds an FP32 value into BF16 away from a device's
nearest-even at a tie (1.005859375 becomes 1.0, 0.666015625 becomes
0.66796875), so a BF16 run there allows a few units of the table's magnitude
per element.

A module compiled for a dependent launch reads only what the grid before it
wrote: the operator launched behind a producer still writing its inputs
(:mod:`tests.python.kernels.chain`) must give its launch after a
synchronization, bit for bit. That launch needs a device.
"""  # noqa: D205

import pytest
import torch

from tests.python.kernels import chain
from tests.python.kernels.rope.rounding import TORCH_TYPES, gamma

pytestmark = pytest.mark.unit

#: Taps one output point reads (`TAPS` in `crates/ops/src/position.rs`).
TAPS = 4

#: The unit roundoff the check takes for each output type. BF16's is 2^-9,
#: half of `ops::tolerance::unit_roundoff`'s 2^-8, which holds a BF16 output
#: tighter than the operator's bound.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-9, "f32": 2.0**-24}

#: The interpreter's BF16 allowance, in units of `UNIT_ROUNDOFF["bf16"]` of
#: the table's largest magnitude.
INTERPRETER_UNITS = 8.0

#: The seed of the dependent launch's values.
SEED = 0


def declared_bound(dtype):
    """`Tolerance::dot(TAPS, f32, dtype)`'s relative bound."""
    return 2.0 * gamma(TAPS) * TAPS**0.5 + UNIT_ROUNDOFF[dtype]


def taps_for(points, side, shared=False):
    """A bilinear tap set over a `side x side` table: four neighbouring
    points, clamped at the table's edges, a quarter each. With `shared`, the
    second point's first two taps are the first point's last.
    """  # noqa: D205
    indices = torch.zeros(points, TAPS, dtype=torch.int32)
    for point in range(points):
        row, column = (point * 7) % side, (point * 13) % side
        below, right = min(row + 1, side - 1), min(column + 1, side - 1)
        indices[point] = torch.tensor(
            [
                row * side + column,
                row * side + right,
                below * side + column,
                below * side + right,
            ]
        )
    if shared:
        indices[1, 0] = indices[0, 3]
        indices[1, 1] = indices[0, 3]
    return indices, torch.full((points, TAPS), 0.25)


def fixture_geometry():
    """Different weights per point, normalized, so the weights are not a
    constant the kernel could ignore.
    """  # noqa: D205
    indices, weights = taps_for(6, 7)
    weights = weights * torch.arange(1, 3).repeat_interleave(2).repeat(6, 1)
    return indices, weights / weights.sum(dim=1, keepdim=True)


def shared_point():
    """One table point reached by two output rows through different taps."""
    indices, weights = taps_for(5, 6, shared=True)
    weights[1] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    return indices, weights


def descending_taps():
    """The grid's own tail, and a row whose taps are in descending order."""
    indices, weights = taps_for(7, 5)
    indices[3] = indices[3, [3, 2, 0, 1]]
    weights[3] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    return indices, weights


def corner_indices():
    """Two points whose four taps are one table point each: 0 and 35."""
    return torch.tensor([[0] * TAPS, [35] * TAPS], dtype=torch.int32)


#: The cases: the output type, the channels, the table's side, the program's
#: block, and the taps with their weights.
CASES = {
    "1152 channels in a 1024-lane block, a masked tail": (
        "bf16",
        1152,
        7,
        1024,
        fixture_geometry,
    ),
    "one table point tapped by two rows at different weights": (
        "f16",
        64,
        6,
        64,
        shared_point,
    ),
    "a seven-point grid with a row of descending taps": (
        "f32",
        48,
        5,
        32,
        descending_taps,
    ),
    "a clamped corner, weights summing to one": (
        "f32",
        32,
        6,
        32,
        lambda: (corner_indices(), torch.full((2, TAPS), 0.25)),
    ),
    "weights that do not sum to one": (
        "f32",
        32,
        6,
        32,
        lambda: (corner_indices()[:1], torch.full((1, TAPS), 0.5)),
    ),
}


def launch(kernels, table, indices, weights, out, block, pdl=False):
    """One launch, one program per output point and block of channels,
    compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    points, channels = out.shape
    kernels._position_interpolate_kernel[
        (points, (channels + block - 1) // block, 1)
    ](
        table,
        indices,
        weights,
        out,
        points,
        channels,
        block,
        int(pdl),
        CHANNELS=channels,
        BLOCK=block,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("case", CASES)
def test_every_channel_matches_the_four_tap_sum(
    case, torch_device, interpreted, kernel_source
):
    dtype, channels, side, block, taps = CASES[case]
    indices, weights = taps()
    element = TORCH_TYPES[dtype]
    points_axis = torch.arange(side * side, dtype=torch.float64).reshape(-1, 1)
    channels_axis = torch.arange(channels, dtype=torch.float64).reshape(1, -1)
    table = (0.5 + 0.01 * points_axis + 0.0001 * channels_axis).to(element)
    out = torch.full(
        (indices.shape[0], channels),
        float("nan"),
        dtype=element,
        device=torch_device,
    )

    launch(
        kernel_source("position_interpolate"),
        table.to(torch_device),
        indices.to(torch_device),
        weights.to(torch_device),
        out,
        block,
    )

    products = table.double()[indices.long()] * weights.double()[:, :, None]
    expected = products.sum(dim=1).to(element).double()
    bound = declared_bound(dtype)
    allowance = (
        INTERPRETER_UNITS * UNIT_ROUNDOFF["bf16"] * float(table.abs().max())
        if dtype == "bf16" and interpreted
        else 0.0
    )
    allowed = (
        (bound + UNIT_ROUNDOFF[dtype]) * expected.abs() + allowance + 1e-30
    )
    got = out.cpu().double()
    missed = ~((got - expected).abs() <= allowed)
    if missed.any():
        position = int(missed.reshape(-1).nonzero()[0])
        cancellation = float(
            (products.abs().sum(dim=1) / expected.abs().clamp_min(1e-30)).max()
        )
        pytest.fail(
            f"{case}, {dtype} channels {channels} side {side} block {block}: "
            f"{int(missed.sum())} of {got.numel()} elements outside the "
            f"bound ({int(got.isnan().sum())} not finite); first at "
            f"{position}: {float(got.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}, rtol {bound:.3e}, "
            f"worst sum|ab|/|out| {cancellation:.1f}"
        )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(SEED)
    kernels = kernel_source("position_interpolate")
    side, points, channels, block = 16, 64, 64, 64

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["table"],
            inputs["indices"],
            inputs["weights"],
            outputs["out"],
            block,
            pdl,
        )
        return [outputs["out"]]

    case = chain.Case(
        "position_interpolate",
        {
            "table": torch.randn(side * side, channels).to(torch.bfloat16),
            "indices": torch.randint(
                0, side * side, (points, TAPS), dtype=torch.int32
            ),
            "weights": torch.rand(points, TAPS),
        },
        {"out": torch.zeros(points, channels, dtype=torch.bfloat16)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"position_interpolate launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a "
        "synchronization"
    )
