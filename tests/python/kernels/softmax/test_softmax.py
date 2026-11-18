"""The row softmax and its logarithm against torch in FP64.

One module covers both operators because they are one row reduction with one
difference: ``softmax`` stores the exponentials' share of their sum in the
input's type, while ``log_softmax`` walks the row in ``STEP``-wide tiles (so a
tile larger than a program's registers costs more steps, not a bigger block)
and stores the log-probability in FP32 whatever the input's type.

The cases are what a row reduction can get wrong: the shape of the fixture's
keys; a width far below the tile, where almost every lane is masked; a width
far above it, where the row is walked in as many tiles as it has and the
running maximum must be rescaled when a later tile raises it; a softmax row
whose leading tiles hold only ``-inf``, as a sampler's candidates ordered by
token id do, where the running maximum is still ``-inf`` when the first tiles
are summed; and a row of one repeated logit, a uniform distribution whose
probabilities are ``1/columns``.

The reference is torch in FP64 and the bound is the operator's own:
``rounding(columns, F32, out)`` for ``softmax``, ``rounding(columns + 1, F32,
F32)`` for ``log_softmax``, with no tolerance of the check's own choosing. Its
absolute part is the output type's subnormal unit: a row of a few thousand
logits has probabilities below FP16's smallest normal, ``2^-14``, which are
stored on its subnormal grid, a rounding that is absolute there.

**``log_softmax`` subtracts, so it is one where an absolute term could be
needed and the declared bound has none.** ``atol`` is zero, so a
log-probability near zero (the chosen token's own, ``x - logsumexp``, which a
sampling step reads) must agree relatively. A log-probability below FP32's ulp
of the normalizer is exactly zero in both the kernel and the reference, and the
bound's ``gamma(columns)`` is a wide ``1e-4`` at a hundred thousand columns,
well above the subtraction's error. The floor the subtraction implies
(``u_out * (|x| + |logsumexp|)``) is reported beside the declared count when a
case fails, so the number stays visible.

A ``bf16`` input stored by Triton's interpreter truncates where a device
rounds, so the reference truncates the same way there; ``f16`` needs a
device.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: The output types' unit roundoff, as `ops::tolerance::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

#: The output types' subnormal unit, as `ops::tolerance::underflow` gives it.
UNDERFLOW = {"f16": 2.0**-24, "bf16": 2.0**-133, "f32": 2.0**-149}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The operators this module checks, and the extra roundings each adds to the
#: reduction.
OPERATORS = {"softmax": 0, "log_softmax": 1}

#: The widest step `log_softmax`'s program walks a row in, its kernel module's
#: `LANES`.
LANES = 4096

#: The cases, cheapest and most diverse first. The first is the fixture's
#: descriptor, `dtype_<dtype>_columns_2048_tile_2048_pdl_false` over three rows.
CASES = {
    "the fixture's shape": {
        "rows": 3,
        "columns": 2048,
        "tile": 2048,
        "dominant_row": True,
        "flat_row": True,
        "seed": 1,
    },
    "a row far shorter than the tile, almost every lane masked": {
        "rows": 2,
        "columns": 100,
        "tile": 2048,
        "dominant_row": False,
        "flat_row": False,
        "seed": 2,
    },
    # No dominant row here on purpose: a hundred-nat spread at this width makes
    # every other probability a subnormal FP32, and the relative bound cannot
    # hold there.
    "a row far wider than one tile, walked in ten of them": {
        "rows": 2,
        "columns": 20480,
        "tile": 2048,
        "dominant_row": False,
        "flat_row": False,
        "seed": 4,
    },
    "a row of one repeated logit, a uniform distribution": {
        "rows": 2,
        "columns": 512,
        "tile": 512,
        "dominant_row": False,
        "flat_row": True,
        "seed": 3,
    },
    "a row whose leading tiles are all -inf, a sampler's ordered candidates": {
        "rows": 2,
        "columns": 20480,
        "tile": 2048,
        "dominant_row": False,
        "flat_row": False,
        "masked_head": True,
        "seed": 5,
    },
}

#: The cases each operator takes: `log_softmax` scores the model's raw logits,
#: which are finite, so it takes no row of `-inf`.
OPERATOR_CASES = [
    (operator, case)
    for operator in OPERATORS
    for case in CASES
    if operator == "softmax" or not CASES[case].get("masked_head")
]


def gamma(reduction):
    """`gamma_n = n u / (1 - n u)` for `n` FP32 roundings, as the operator
    crate computes it.
    """  # noqa: D205
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def bound(columns, out_dtype, extra):
    """The operator's relative bound over a row of `columns` terms."""
    return gamma(columns + extra) + UNIT_ROUNDOFF[out_dtype]


def truncate(values):
    """The reference as Triton's interpreter stores it: truncated, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own
    mantissa.
    """
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def launch(kernel_source, operator, x, out, columns, tile):
    """One launch of `operator`'s program, one program per row of `x`."""
    rows = x.shape[0]
    if operator == "softmax":
        kernel_source("softmax")._softmax_kernel[(rows,)](
            x, out, columns, BLOCK=tile, PDL=False
        )
    else:
        kernel_source("log_softmax")._log_softmax_kernel[(rows,)](
            x, out, columns, STEP=min(tile, LANES), PDL=False
        )


@pytest.mark.parametrize(
    "dtype", ["f32", "bf16", pytest.param("f16", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize(("operator", "case"), OPERATOR_CASES)
def test_every_row_matches_torch_within_the_operators_bound(
    operator, case, dtype, torch_device, interpreted, kernel_source
):
    shape = CASES[case]
    generator = torch.Generator().manual_seed(shape["seed"])
    rows, columns, tile = shape["rows"], shape["columns"], shape["tile"]

    x = torch.randn(rows, columns, generator=generator)
    # Both of row 0's patterns apply in this order: the flat row last.
    if shape["dominant_row"]:
        # One logit a hundred nats above the rest: probabilities 1 and 0,
        # log-probabilities 0 and -100.
        x[0, :] = -100.0
        x[0, columns // 2] = 0.0
    if shape["flat_row"]:
        x[0, :] = 0.5
    if shape.get("masked_head"):
        # A sampler's candidates in token order: every dropped candidate is
        # `-inf`, so the tiles before the first kept one hold no finite value at
        # all. The kept ones sit in the last tile, and the probabilities of the
        # dropped ones are exactly zero.
        x[:, : columns - tile // 2] = float("-inf")
    x = x.to(TORCH_TYPES[dtype])

    out_dtype = x.dtype if operator == "softmax" else torch.float32
    out = torch.full(
        (rows, columns), float("nan"), dtype=out_dtype, device=torch_device
    )
    launch(kernel_source, operator, x.to(torch_device), out, columns, tile)

    exact = x.double()
    largest = exact.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(exact - largest)
    total = exponentials.sum(dim=-1, keepdim=True)
    logsumexp = largest + torch.log(total)
    if operator == "softmax":
        expected = exponentials / total
    else:
        expected = exact - logsumexp
    # The interpreter's narrow store truncates, and only a narrow output is
    # stored narrow: `log_softmax` stores FP32 whatever comes in.
    if interpreted and dtype == "bf16" and operator == "softmax":
        expected = truncate(expected)

    stored = dtype if operator == "softmax" else "f32"
    relative = bound(columns, stored, OPERATORS[operator])
    absolute = UNDERFLOW[stored]
    if operator == "softmax":
        floor = 0.0
    else:
        # The subtraction's rounding is of the logit and the normalizer, not of
        # the result.
        floor = UNIT_ROUNDOFF["f32"] * (exact.abs() + logsumexp.abs())

    got = out.cpu().double()
    difference = (got - expected).abs()
    # A difference that is not a number is never inside a bound: `<=` keeps it
    # outside, where `>` would let it pass.
    outside = ~(difference <= absolute + relative * expected.abs())
    scaled = ~(difference <= absolute + relative * expected.abs() + floor)
    largest_relative = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    assert not outside.any(), (
        f"{operator} {rows} x {columns} of tile {tile}, {dtype}: "
        f"{int(outside.sum())} of {expected.numel()} outside the declared "
        f"bound {relative:.3e}, atol {absolute:.3e} (per row "
        f"{outside.sum(dim=-1).tolist()}), "
        f"{int(scaled.sum())} outside it with the operands' floor; max "
        f"absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, "
        f"max relative {largest_relative:.3e}"
    )
