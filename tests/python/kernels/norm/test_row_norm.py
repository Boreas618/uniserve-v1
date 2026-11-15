"""The row-normalization programs, rms_norm and add_rms_norm, against FP64.

One module covers both operators because they are one row reduction with one
difference: ``add_rms_norm`` sums ``x + residual`` first, writes that sum back
out as the new residual, and normalizes the sum. ``rms_norm`` also reads a
zero-centred weight (``1 + w``) and a gain of several rows (row ``i`` reads
gain row ``i mod groups``); ``add_rms_norm`` reads the centred weight.

The cases are what a masked row reduction can get wrong: the shapes the
fixture's keys name, a width far below the tile where almost every lane is
masked, a row whose values are all equal (and, for the fused add, a residual
that cancels the input exactly, so the sum and the normalized row are exactly
zero), and a row with a large common offset.

The reference is torch in FP64 and the bound ``gamma_n + u_out``: the FP32
reduction of the row, of ``n = columns`` terms for ``rms_norm`` and
``columns + 1`` for ``add_rms_norm``, and one rounding into the output type,
within the operators' declared ``rounding(n, F32, out)`` (`RmsNorm::tolerance`,
`AddRmsNorm::tolerance`, ``2 gamma_n + u_out``). The updated residual is one
rounded sum, held to ``u_out`` of itself.

The residual's bound has no absolute term, so a sum that cancels must agree
relatively; a failure states the count under the floor of its operands'
magnitudes (``u_out (|x| + |residual|)``) beside the declared one, and the
declared one decides, since the tolerance is the operator's to change.

An F16 output is checked on a device: its store on Triton's interpreter
leaves a few elements a unit in the last place outside the bound.

A hand-computed case holds ``rms_norm`` to literals, which a mistake the
kernel and the reference share (a row read as a column, a dropped ``eps``)
cannot pass.
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

#: Each operator's roundings beyond the row's reduction: the residual's add.
EXTRA = {"rms_norm": 0, "add_rms_norm": 1}

EPS = 1e-6

#: The forms checked: the operator, whether its weight is zero-centred, and
#: the rows of `rms_norm`'s gain.
FORMS = {
    "rms_norm": ("rms_norm", False, 1),
    "rms_norm, zero-centred": ("rms_norm", True, 1),
    "rms_norm, zero-centred, a gain of 4 rows": ("rms_norm", True, 4),
    "add_rms_norm": ("add_rms_norm", False, 1),
    "add_rms_norm, zero-centred": ("add_rms_norm", True, 1),
}

#: The cases, cheapest and most diverse first: the rows (per row of the gain),
#: the row width, the program's tile, the seed, and the special rows: a flat
#: first row, a large common offset on the last, and a residual cancelling the
#: first.
CASES = {
    "the fixture's shape, 2048 of a 2048 tile": {
        "rows": 4,
        "columns": 2048,
        "tile": 2048,
        "seed": 1,
        "flat_row": True,
        "offset_row": True,
        "cancelling_row": True,
    },
    "a deployment's 5120 of an 8192 tile": {
        "rows": 4,
        "columns": 5120,
        "tile": 8192,
        "seed": 1,
        "flat_row": True,
        "offset_row": True,
        "cancelling_row": True,
    },
    "a row far shorter than the tile, almost every lane masked": {
        "rows": 3,
        "columns": 100,
        "tile": 2048,
        "seed": 2,
    },
    "a width that is a whole number of tiles": {
        "rows": 2,
        "columns": 512,
        "tile": 512,
        "seed": 3,
        "offset_row": True,
    },
}

#: The value a flat row repeats, and the offset of the offset row.
FLAT, OFFSET = 3.0, 1000.0

#: Two rows of four with fixed operands and their values computed by hand from
#: `x rsqrt(mean(x^2) + eps) weight` (mean squares 7.5 and 1.875).
HAND = {
    "x": [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.5]],
    "weight": [1.0, 0.5, -1.0, 2.0],
    "expect": [
        [
            0.3651483473268884,
            0.3651483473268884,
            -1.0954450419806652,
            2.921186778615107,
        ],
        [
            0.3651482742972506,
            -0.3651482742972506,
            -1.4605930971890024,
            2.1908896457835034,
        ],
    ],
}

#: The tile the hand-computed rows are normalized in.
HAND_TILE = 2048


def bound(columns, dtype, operator):
    """The relative bound over a row of `columns` terms: `gamma_n + u_out`."""
    return gamma(columns + EXTRA[operator]) + UNIT_ROUNDOFF[dtype]


def launch_rms_norm(kernels, x, weight, out, tile, centred, groups):
    """One `rms_norm` launch, one program per row."""
    rows, columns = x.shape
    kernels._rms_norm_kernel[(rows,)](
        x,
        weight,
        out,
        columns,
        EPS,
        0,
        BLOCK=tile,
        PDL=False,
        CENTRED=centred,
        GROUPS=groups,
    )


def launch_add_rms_norm(
    kernels, x, residual, weight, out, summed, tile, centred
):
    """One `add_rms_norm` launch, one program per row; `summed` receives the
    updated residual.
    """  # noqa: D205
    rows, columns = x.shape
    kernels._add_rms_norm_kernel[(rows,)](
        x,
        residual,
        weight,
        out,
        summed,
        columns,
        EPS,
        0,
        BLOCK=tile,
        PDL=False,
        CENTRED=centred,
    )


def failure(name, difference, expected, declared, relative, floored=None):
    """The message of an output outside its bound, with the count under the
    operands' floor where the output has one.
    """  # noqa: D205
    floor = "" if floored is None else f" ({floored} with the operands' floor)"
    return (
        f"{name}: {declared} of {expected.numel()} outside the bound{floor}, "
        "max absolute "
        f"{difference.nan_to_num(nan=float('inf')).max().item():.3e}, max "
        f"relative {largest_relative(difference, expected):.3e}, rtol "
        f"{relative:.3e}"
    )


def test_the_hand_computed_rows_match_their_literals(
    torch_device, kernel_source
):
    x = torch.tensor(HAND["x"], device=torch_device)
    weight = torch.tensor(HAND["weight"], device=torch_device)
    out = torch.full_like(x, float("nan"))

    launch_rms_norm(
        kernel_source("rms_norm"), x, weight, out, HAND_TILE, False, 1
    )

    expect = torch.tensor(HAND["expect"], dtype=torch.float64)
    difference = (out.cpu().double() - expect).abs()
    relative = bound(x.shape[1], "f32", "rms_norm")
    missed = outside(difference, relative * expect.abs())
    assert missed == 0, (
        f"hand-computed 2 x 4 of tile {HAND_TILE}, f32: {missed} outside the "
        f"bound, max absolute {difference.max().item():.3e}, rtol "
        f"{relative:.3e}"
    )


@pytest.mark.parametrize(
    "dtype", ["bf16", "f32", pytest.param("f16", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("form", FORMS)
def test_every_output_matches_fp64_within_the_operators_bound(
    form, case, dtype, torch_device, interpreted, kernel_source
):
    operator, centred, groups = FORMS[form]
    spec = CASES[case]
    columns, tile = spec["columns"], spec["tile"]
    # The rows are whole groups of the gain.
    rows = spec["rows"] * groups
    element = TORCH_TYPES[dtype]
    generator = torch.Generator().manual_seed(spec["seed"])
    x = torch.randn(rows, columns, generator=generator)
    if spec.get("flat_row"):
        x[0, :] = FLAT
    if spec.get("offset_row"):
        x[-1, :] += OFFSET
    x = x.to(element)
    weight = torch.randn(groups * columns, generator=generator).to(element)
    out = torch.full(
        (rows, columns), float("nan"), dtype=element, device=torch_device
    )
    kernels = kernel_source(operator)

    exact_x = x.double()
    if operator == "rms_norm":
        launch_rms_norm(
            kernels,
            x.to(torch_device),
            weight.to(torch_device),
            out,
            tile,
            centred,
            groups,
        )
        row = exact_x
        gain = weight.double().view(groups, columns)[
            torch.arange(rows) % groups
        ]
    else:
        residual = torch.randn(rows, columns, generator=generator).to(element)
        if spec.get("cancelling_row"):
            # `x + residual` is exactly zero: the sum and its normalization
            # both vanish.
            residual[0, :] = -x[0, :]
        summed = torch.full_like(out, float("nan"))
        launch_add_rms_norm(
            kernels,
            x.to(torch_device),
            residual.to(torch_device),
            weight.to(torch_device),
            out,
            summed,
            tile,
            centred,
        )
        exact_r = residual.double()
        row = exact_x + exact_r
        gain = weight.double()

        expected = interpreter_store(row, dtype, interpreted)
        difference = (summed.cpu().double() - expected).abs()
        relative = UNIT_ROUNDOFF[dtype]
        floor = UNIT_ROUNDOFF[dtype] * (exact_x.abs() + exact_r.abs())
        declared = outside(difference, relative * expected.abs())
        floored = outside(difference, relative * expected.abs() + floor)
        assert declared == 0, failure(
            f"{form}, {case}, {dtype}, the residual",
            difference,
            expected,
            declared,
            relative,
            floored,
        )

    inverse = torch.rsqrt((row * row).sum(dim=-1, keepdim=True) / columns + EPS)
    expected = interpreter_store(
        row * inverse * (gain + float(centred)), dtype, interpreted
    )
    difference = (out.cpu().double() - expected).abs()
    relative = bound(columns, dtype, operator)
    declared = outside(difference, relative * expected.abs())
    assert declared == 0, failure(
        f"{form}, {case}, {dtype}, the normalized row",
        difference,
        expected,
        declared,
        relative,
    )
