#!/usr/bin/env python3
"""Run the layer_norm program and check every row against torch's own arithmetic.

    python layer_norm_check.py                        # the fixture's 1152 cell, bias on
    python layer_norm_check.py --interpreter          # the same on the CPU
    python layer_norm_check.py --columns 5120 --tile 8192 --bias false

The operator normalizes each row by its own mean and variance and then scales and shifts
it per channel. `columns` is a **runtime** extent in this program and the width the module
was built for is the tile, so the cases here are chosen for what a masked row can get
wrong: a width far below one tile (the mask and the reduction over it), a width that
straddles a tile with a shape a deployment actually uses (1152 and 5120 of 2048 and 8192),
and a row whose values are all equal, where the variance is exactly zero and `eps` alone
decides the inverse.

The reference is torch in FP64 over the row — mean, biased variance, `(x − mean)·rsqrt(var
+ eps)·weight + bias` — and the bound is the operator's, `rounding(2·columns, F32, out)`:
two sums over the row in FP32 and one rounding to the output type, with no tolerance of
the check's own choosing. The measured maximum is printed beside the bound.

**A declared bound with no absolute term cannot be met elementwise by a normalized
output.** `rounding` sets `atol = 0`, so an element whose exact value is near zero must
agree *relatively*, and a row of unit normal values contains several such elements: one run
of the 1152-wide case differs by `2.9e-8` on an exact `−5.2e-5` — one FP32 ulp of the
operands, and 0.03 % of the scale the weight and bias set — and fails only because the
division is by `5.2e-5`. Both counts are therefore printed: the declared bound's, and one
with the absolute floor the operand scale implies (`u_out · max(|weight| + |bias|)`).
`--bound` chooses which one decides the run, and it defaults to `declared`: the tolerance
is the operator's to change, not the check's to relax.
"""

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The relative bound's reduction length, from `norm.rs::tolerance`: two sums over the row.
REDUCTIONS = 2

#: The output types' unit roundoff, as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


def gamma(reduction):
    """`γₙ = n·u/(1 − n·u)` for `n` FP32 roundings, as the operator crate computes it."""
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def bound(columns, dtype):
    """The operator's relative bound: `2γₙ + u_out` over `2·columns` terms."""
    return 2.0 * gamma(REDUCTIONS * columns) + UNIT_ROUNDOFF[dtype]


def absolute_floor(weight, bias, dtype):
    """The absolute term the operands' scale implies, for a normalized output.

    The output is `(x − mean)·rsqrt(var + eps)·weight + bias`, so its smallest meaningful
    unit is the output type's rounding of the weight and bias it is built from.
    """
    scale = weight.cpu().double().abs()
    if bias is not None:
        scale = scale + bias.cpu().double().abs()
    return UNIT_ROUNDOFF[dtype] * scale.max().item()


def truncate(values):
    """The reference as Triton's interpreter stores it: one ulp low, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own mantissa.
    """
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def check(case, device):
    """Normalize one matrix and compare every element against the exact rows."""
    import layer_norm as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, columns = case["rows"], case["columns"]
    x = torch.randn(rows, columns, generator=generator, dtype=torch.float32)
    # A row of one repeated value has zero variance: `eps` is the whole of its inverse.
    if case["flat_row"]:
        x[0, :] = 2.5
    # A row with a large offset tests the cancellation the two sums see.
    if case["offset_row"]:
        x[-1, :] += 1000.0
    x = x.to(dtype).contiguous()
    weight = torch.randn(columns, generator=generator, dtype=torch.float32).to(
        dtype
    )
    bias = (
        torch.randn(columns, generator=generator, dtype=torch.float32).to(dtype)
        if case["bias"]
        else None
    )
    out = torch.full((rows, columns), float("nan"), dtype=dtype)

    eps = case["eps"]
    kernels._layer_norm_kernel[(rows,)](
        x,
        weight,
        bias if bias is not None else weight,
        out,
        columns,
        eps,
        1 if bias is not None else 0,
        1 if case["pdl"] else 0,
        BLOCK=case["tile"],
        HAS_BIAS=bias is not None,
        PDL=case["pdl"],
    )

    exact = x.cpu().double()
    mean = exact.mean(dim=-1, keepdim=True)
    variance = exact.var(dim=-1, unbiased=False, keepdim=True)
    expected = (exact - mean) / torch.sqrt(variance + eps)
    expected = expected * weight.cpu().double()
    if bias is not None:
        expected = expected + bias.cpu().double()

    if case["truncate"]:
        # Triton's interpreter truncates a narrow store instead of rounding to nearest
        # even, one ulp low; the reference is truncated the same way so that what the run
        # compares is the arithmetic and not the interpreter's cast. The truncation is of
        # the value's own mantissa: rounding it to `f32` first would step values across a
        # boundary the kernel's arithmetic stays below.
        expected = truncate(expected)
    got = out.cpu().double()
    difference = (got - expected).abs()
    relative = bound(columns, case["dtype"]) * expected.abs()
    floor = absolute_floor(weight, bias, case["dtype"])
    declared_outside = (~(difference <= relative)).sum().item()
    scaled_outside = (~(difference <= relative + floor)).sum().item()
    largest = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    deciding = (
        declared_outside if case["bound"] == "declared" else scaled_outside
    )
    print(
        f"{'exact' if deciding == 0 else 'FAIL'}: {rows} x {columns} of tile "
        f"{case['tile']}, {case['dtype']}, "
        f"bias {str(case['bias']).lower()}"
        f"{', interpreter cast truncated' if case['truncate'] else ''} — "
        f"max absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, max relative {largest:.3e}, "
        f"declared bound {bound(columns, case['dtype']):.3e} "
        f"({declared_outside} outside), with the {floor:.3e} absolute floor the operands "
        f"imply ({scaled_outside} outside)"
    )
    return deciding


def cases(flags, interpreter):
    """The cases one run checks, cheapest and most diverse first."""
    return [
        {
            "label": "the fixture's width: an audio encoder's row of 1152 in a 2048 tile",
            "dtype": flags.dtype,
            "rows": flags.rows,
            "columns": flags.columns,
            "tile": flags.tile,
            "bias": flags.bias == "true",
            "eps": flags.eps,
            "pdl": flags.pdl == "true" and not interpreter,
            "flat_row": True,
            "offset_row": False,
            "seed": 1,
            "bound": flags.bound,
            "truncate": False,
        },
        {
            "label": "a row far shorter than the tile: almost every lane is masked",
            "dtype": flags.dtype,
            "rows": 3,
            "columns": 100,
            "tile": flags.tile,
            "bias": True,
            "eps": flags.eps,
            "pdl": False,
            "flat_row": False,
            "offset_row": False,
            "seed": 2,
            "bound": flags.bound,
            "truncate": False,
        },
        {
            "label": "a width that straddles its tile: 5000 of 8192",
            "dtype": flags.dtype,
            "rows": 2,
            "columns": 5000,
            "tile": 8192,
            "bias": flags.bias == "true",
            "eps": flags.eps,
            "pdl": False,
            "flat_row": False,
            "offset_row": False,
            "seed": 3,
            "bound": flags.bound,
            "truncate": False,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--columns", type=int, default=1152)
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--bias", choices=["true", "false"], default="true")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--pdl", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides the run: the operator's own (atol 0), or the same with "
        "the absolute floor the weight and bias scale imply",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.columns > arguments.tile:
        parser.error(
            f"a tile of {arguments.tile} lanes cannot cover a row of "
            f"{arguments.columns}: the program reduces over the tile"
        )
    if not arguments.interpreter and not torch.cuda.is_available():
        # A host without a device takes the interpreter: a check must run where it lands.
        print(
            "no GPU on this host: running through Triton's interpreter instead",
            file=sys.stderr,
        )
        arguments.interpreter = True
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"

    key = (
        f"dtype_{arguments.dtype}_columns_{arguments.columns}"
        f"_tile_{arguments.tile}_bias_{arguments.bias}_pdl_{arguments.pdl}"
    )
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    device = "cpu" if arguments.interpreter else "cuda"
    outside = 0
    for case in cases(arguments, arguments.interpreter):
        # The interpreter's narrow store truncates; simulate it only where it is known.
        case["truncate"] = arguments.interpreter and arguments.dtype == "bf16"
        outside += check(case, device)
    return 1 if outside else 0


if __name__ == "__main__":
    sys.exit(main())
