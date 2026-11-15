"""The fused residual-and-normalization programs against FP64, every row.

One module covers scaled_residual_rms_norm and scaled_residual_layer_norm
because they are one row reduction behind one difference: both fold a
channel-scaled, optionally biased update into the residual, then normalize,
the RMS form by the row's root mean square and the layer form by its mean and
variance, adding the learned bias afterwards. The RMS form also writes the
updated residual back into ``hidden``; the layer form leaves the residual
unchanged.

The cases: the deployed geometry, the ``update_bias`` and ``absmax`` flags on,
a residual that cancels the update exactly (the RMS of zero, and a layer
output that is nothing but the bias, where the normalized value is exactly
zero and a relative bound has nothing to divide), a row of one repeated value,
one scale for the whole row, and, for the **RMS form only**, a row with a
large common offset. A layer norm over a row of values near a thousand with
unit-scale variation is ill-conditioned in FP32 by construction: the mean's
own rounding (``gamma(columns) 1024 ~ 0.1``) shifts every centered value by
more than the variance it divides by, so no implementation meets an exact
reference there; the layer form's second case is a row of ordinary values.

The bound is ``gamma_n + 3 u_out``: the reduction over the row in FP32 plus
three roundings of the gated update in the output type, within the
operators' declared ``rounding(n, F32, out)`` plus those roundings
(`ScaledResidualRmsNorm::tolerance`, `ScaledResidualLayerNorm::tolerance`).
``atol`` is zero, so a cancelled row must agree relatively against nothing.
The residual's own rounding is the compiler's to choose:
``hidden + update * scale`` is one fused multiply-add on a device and two
roundings on Triton's interpreter, so a reference that mirrors either one is
wrong on the other. Each output therefore also has a floor of the roundings
its operands imply; in BF16 the declared bound absorbs the residual's
rounding and decides, in FP32 it does not and the bound with the floor
decides. A failure states both counts.

A BF16 store on Triton's interpreter truncates, so there the reference of the
updated residual and of the normalized row is truncated the same way.
"""

import pytest
import torch
from uniserve_testing.kernels.norm.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    gamma,
    interpreter_store,
    largest_relative,
    outside,
)

pytestmark = pytest.mark.unit

#: Each operator's programs, without and with the magnitude record.
OPERATORS = {
    "scaled_residual_rms_norm": (
        "_scaled_residual_rms_norm_kernel",
        "_scaled_residual_rms_norm_absmax_kernel",
    ),
    "scaled_residual_layer_norm": (
        "_scaled_residual_layer_norm_kernel",
        "_scaled_residual_layer_norm_absmax_kernel",
    ),
}

EPS = 1e-6

#: Whether the bound with the operands' floor decides an element type: the
#: declared bound absorbs the residual's rounding in BF16 and not in FP32.
FLOORED = {"bf16": False, "f32": True}

#: The deployed geometry: the row width and the tile covering it.
COLUMNS, TILE = 1536, 2048

#: The cases, cheapest and most diverse first: the rows, the row width and
#: tile, the flags, the special rows (a residual cancelling the update on the
#: first row, a flat last row, a large common offset on the last row of the
#: RMS form) and the seed.
CASES = {
    "the deployed geometry": {
        "rows": 3,
        "cancelling_row": True,
        "flat_row": True,
        "seed": 1,
    },
    "the deployed geometry, with the magnitude record": {
        "rows": 3,
        "absmax": True,
        "cancelling_row": True,
        "flat_row": True,
        "seed": 1,
    },
    "a row with a large common offset in the RMS form, ordinary values in "
    "the layer form": {"rows": 2, "offset_row": True, "seed": 2},
    "one row, with an update bias and the magnitude record": {
        "rows": 1,
        "columns": 1024,
        "tile": 1024,
        "update_bias": True,
        "absmax": True,
        "seed": 3,
    },
    "one scale for the whole row": {
        "rows": 2,
        "scalar_scale": True,
        "cancelling_row": True,
        "seed": 4,
    },
}

#: The value of the flat row, and the common offset of the offset row.
FLAT, OFFSET = 2.0, 1000.0


def launch(kernels, operator, operands, eps, tile, absmax):
    """One launch, one program per row, of the operator's program with or
    without the magnitude record. `operands` are the program's tensors in
    order, the layer form's bias among them, `partials` last when `absmax`.
    """  # noqa: D205
    kernel = getattr(kernels, OPERATORS[operator][absmax])
    hidden, update, update_bias, scale = operands[:4]
    kernel[(hidden.shape[0],)](
        *operands,
        eps,
        scale.numel(),
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=hidden.shape[1],
        BLOCK=tile,
    )


def compare(name, got, expected, relative, floor, floored):
    """The message of an output outside its deciding bound, or None."""
    difference = (got - expected).abs()
    declared = outside(difference, relative * expected.abs())
    with_floor = outside(difference, relative * expected.abs() + floor)
    deciding = with_floor if floored else declared
    if deciding == 0:
        return None
    return (
        f"{name}: {declared} of {expected.numel()} outside the declared "
        f"bound, {with_floor} with the operands' floor ("
        f"{'deciding' if floored else 'reported'}), max absolute "
        f"{difference.nan_to_num(nan=float('inf')).max().item():.3e}, max "
        f"relative {largest_relative(difference, expected):.3e}, rtol "
        f"{relative:.3e}"
    )


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("operator", OPERATORS)
def test_the_residual_and_the_row_match_fp64_within_the_bound(
    operator, case, dtype, torch_device, interpreted, kernel_source
):
    spec = CASES[case]
    rows = spec["rows"]
    columns, tile = spec.get("columns", COLUMNS), spec.get("tile", TILE)
    absmax = spec.get("absmax", False)
    rms = operator == "scaled_residual_rms_norm"
    element = TORCH_TYPES[dtype]
    generator = torch.Generator().manual_seed(spec["seed"])
    hidden = torch.randn(rows, columns, generator=generator).to(element)
    update = torch.randn(rows, columns, generator=generator).to(element)
    # One scale per channel, or one for the whole row: the program reads lane
    # 0 for every column when the launch states a scale of width 1.
    scale = torch.randn(
        1 if spec.get("scalar_scale") else columns, generator=generator
    ).to(element)
    weight = torch.randn(columns, generator=generator).to(element)
    update_bias = (
        torch.randn(columns, generator=generator).to(element)
        if spec.get("update_bias")
        else None
    )
    bias = torch.randn(columns, generator=generator).to(element)
    if spec.get("cancelling_row"):
        hidden[0, :] = (-update[0].double() * scale.double()).to(element)
    if spec.get("flat_row"):
        hidden[-1, :] = FLAT
    if spec.get("offset_row") and rms:
        hidden[-1, :] = 0.0
        update[-1, :] = (OFFSET / scale.double()).to(element)
    # The residual as the program reads it: the RMS form overwrites `hidden`.
    hidden_before = hidden.double()

    on_device = {
        name: tensor.to(torch_device) if tensor is not None else None
        for name, tensor in (
            ("hidden", hidden),
            ("update", update),
            ("update_bias", update_bias),
            ("scale", scale),
            ("weight", weight),
            ("bias", bias),
        )
    }
    out = torch.full(
        (rows, columns), float("nan"), dtype=element, device=torch_device
    )
    partials = torch.full((rows,), float("nan"), device=torch_device)
    names = ["hidden", "update", "update_bias", "scale", "weight"]
    if not rms:
        names.append("bias")
    operands = [on_device[name] for name in names] + [out]
    if absmax:
        operands.append(partials)

    launch(kernel_source(operator), operator, operands, EPS, tile, absmax)

    exact_update = update.double()
    if update_bias is not None:
        exact_update = exact_update + update_bias.double()
    product = exact_update * scale.double()
    residual = hidden_before + product
    if rms:
        inverse = torch.rsqrt(
            (residual * residual).sum(dim=-1, keepdim=True) / columns + EPS
        )
        expected_out = residual * inverse * weight.double()
    else:
        centered = residual - residual.mean(dim=-1, keepdim=True)
        inverse = torch.rsqrt(
            (centered * centered).sum(dim=-1, keepdim=True) / columns + EPS
        )
        expected_out = centered * inverse * weight.double() + bias.double()

    unit = UNIT_ROUNDOFF[dtype]
    # One FP32 rounding of the residual's two operands: the difference between
    # a fused multiply-add and two roundings, the compiler's choice.
    residual_floor = UNIT_ROUNDOFF["f32"] * (
        hidden_before.abs() + product.abs()
    )
    # The normalized value is a product scaled by the row's inverse deviation
    # and, in the layer form, shifted by the learned bias, whose magnitude is
    # what a near-zero output is compared against. The residual's rounding
    # reaches it through that scale.
    out_floor = unit * (
        expected_out.abs() + (0.0 if rms else bias.double().abs())
    ) + residual_floor * (weight.double().abs() * inverse)
    floored = FLOORED[dtype]
    where = f"{operator}, {case}, {rows} x {columns} of tile {tile}, {dtype}"
    failures = []
    if rms:
        failures.append(
            compare(
                f"{where}, the residual",
                on_device["hidden"].cpu().double(),
                interpreter_store(residual, dtype, interpreted),
                unit,
                unit * residual.abs() + residual_floor,
                floored,
            )
        )
    expected_out = interpreter_store(expected_out, dtype, interpreted)
    failures.append(
        compare(
            f"{where}, the normalized row",
            out.cpu().double(),
            expected_out,
            gamma(columns) + 3.0 * unit,
            out_floor,
            floored,
        )
    )
    failures = [failure for failure in failures if failure is not None]
    assert not failures, "; ".join(failures)

    if absmax:
        # A maximum of values already in the output type has nothing to
        # round, so the record is compared exactly.
        record = partials.cpu().double()
        wanted = out.cpu().double().abs().max(dim=-1).values
        wrong = int((record != wanted).sum())
        assert wrong == 0, (
            f"{where}: {wrong} of {rows} magnitudes differ from the largest "
            "magnitude of the row the program stored"
        )
