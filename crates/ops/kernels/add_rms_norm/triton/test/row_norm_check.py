#!/usr/bin/env python3
"""Run one of the row-normalization programs and check every row against torch.

    python row_norm_check.py --op rms_norm                          # 2048 of a 2048 tile
    python row_norm_check.py --op add_rms_norm --interpreter         # the same on the CPU
    python row_norm_check.py --op rms_norm --columns 5120 --tile 8192

One script covers both operators because they are one row reduction with one difference —
`add_rms_norm` sums `x + residual` first, writes that sum back out as the new residual, and
normalizes the sum. The same file is kept in each operator's package, byte for byte, and a
test in `crates/program-server/tests/it/packages.rs` holds them together.

The cases are what a masked row reduction can get wrong: the shapes the fixture's keys
name, a width far below the tile where almost every lane is masked, a row whose values are
all equal (and, for the fused add, a residual that cancels the input exactly, so the sum and
the normalized row are exactly zero), and a row with a large common offset.

The reference is torch in FP64 and the bound is the operator's own — `rounding(columns,
F32, out)` for `rms_norm`, `rounding(columns + 1, …)` for `add_rms_norm` — with no tolerance
of the check's own choosing.

**And the residual output needs an absolute term that bound does not carry.** `atol` is
zero, so the updated residual must agree *relatively* with a sum that cancels: rows whose
`x + residual` is near zero have every implementation outside the bound, while their
absolute difference is a couple of FP32 ulps of the operands. Each run prints both counts —
the declared bound's, and one taken against `u_out · (|x| + |residual|)` — and `--bound`
decides the run, defaulting to `declared`: the tolerance is the operator's to change, not
the check's to relax.

A `bf16` run under Triton's interpreter truncates its reference the way the interpreter
truncates narrow stores (one ulp, not round-to-nearest) and says so in its output; `f16` is
device-only.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The output types' unit roundoff, as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

TORCH_TYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}

#: The operators this script runs, and the extra roundings each adds to the reduction.
OPERATORS = {"rms_norm": 0, "add_rms_norm": 1}


def gamma(reduction):
    """`γₙ = n·u/(1 − n·u)` for `n` FP32 roundings, as the operator crate computes it."""
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def bound(columns, dtype, extra):
    """The operator's relative bound over a row of `columns` terms."""
    return gamma(columns + extra) + UNIT_ROUNDOFF[dtype]


def truncate(values, dtype):
    """The reference as Triton's interpreter would store it: one ulp low, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own mantissa —
    truncating a *rounded* `f32` instead would step values across a boundary that the
    kernel's own arithmetic stays below. Only `bf16` has the simulation; that is the type
    the interpreter's narrow cast is known for.
    """
    assert dtype == torch.bfloat16, "only the bf16 cast is simulated"
    # FP64 keeps 52 explicit mantissa bits and bf16 seven: clearing the low 45 truncates.
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def compare(name, got, expected, floor, relative, deciding_bound):
    """Count the elements outside each bound and give the line that reports them."""
    difference = (got - expected).abs()
    declared_outside = (~(difference <= relative * expected.abs())).sum().item()
    scaled_outside = (~(difference <= relative * expected.abs() + floor)).sum().item()
    largest = (difference / expected.abs().clamp(min=1e-30)).nan_to_num(nan=float("inf")).max().item()
    deciding = declared_outside if deciding_bound == "declared" else scaled_outside
    line = (
        f"  {name}: max absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, max relative "
        f"{largest:.3e}, declared {declared_outside} of {expected.numel()} outside, "
        f"with the operands' floor {scaled_outside} outside"
    )
    return deciding, line


def check(operator, case):
    """Run one operator over one matrix and compare every output tensor."""
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = TORCH_TYPES[case["dtype"]]
    rows, columns, tile = case["rows"], case["columns"], case["tile"]

    x = torch.randn(rows, columns, generator=generator)
    if case["flat_row"]:
        x[0, :] = 3.0
    if case["offset_row"]:
        x[-1, :] += 1000.0
    x = x.to(dtype)
    weight = torch.randn(columns, generator=generator).to(dtype)
    eps = case["eps"]

    if operator == "rms_norm":
        import rms_norm as kernels

        out = torch.full((rows, columns), float("nan"), dtype=dtype)
        kernels._rms_norm_kernel[(rows,)](
            x,
            weight,
            out,
            columns,
            eps,
            int(case["pdl"]),
            BLOCK=tile,
            PDL=case["pdl"],
        )
        exact_x = x.cpu().double()
        expected_out = exact_x * torch.rsqrt(
            (exact_x * exact_x).sum(dim=-1, keepdim=True) / columns + eps
        )
        expected_out = expected_out * weight.cpu().double()
        if case["truncate"]:
            expected_out = truncate(expected_out, dtype)
        relative = bound(columns, case["dtype"], OPERATORS[operator])
        outside, line = compare(
            "out", out.cpu().double(), expected_out, 0.0, relative, case["bound"]
        )
        lines = [line]
    else:
        import add_rms_norm as kernels

        residual = torch.randn(rows, columns, generator=generator).to(dtype)
        if case["cancelling_row"]:
            # `x + residual` is exactly zero: the sum and its normalization both vanish.
            residual[0, :] = -x[0, :]
        out = torch.full((rows, columns), float("nan"), dtype=dtype)
        residual_out = torch.full((rows, columns), float("nan"), dtype=dtype)
        kernels._add_rms_norm_kernel[(rows,)](
            x,
            residual,
            weight,
            out,
            residual_out,
            columns,
            eps,
            int(case["pdl"]),
            BLOCK=tile,
            PDL=case["pdl"],
        )
        exact_x = x.cpu().double()
        exact_r = residual.cpu().double()
        exact_c = exact_x + exact_r
        expected_out = exact_c * torch.rsqrt(
            (exact_c * exact_c).sum(dim=-1, keepdim=True) / columns + eps
        )
        expected_out = expected_out * weight.cpu().double()
        expected_residual = exact_c
        if case["truncate"]:
            expected_out = truncate(expected_out, dtype)
            expected_residual = truncate(expected_residual, dtype)
        relative = bound(columns, case["dtype"], OPERATORS[operator])
        # The residual is a sum of two operands, so its rounding is of their magnitudes.
        residual_floor = UNIT_ROUNDOFF[case["dtype"]] * (exact_x.abs() + exact_r.abs())
        outside, residual_line = compare(
            "residual",
            residual_out.cpu().double(),
            expected_residual,
            residual_floor,
            UNIT_ROUNDOFF[case["dtype"]],
            case["bound"],
        )
        out_outside, out_line = compare(
            "out", out.cpu().double(), expected_out, 0.0, relative, case["bound"]
        )
        outside += out_outside
        lines = [residual_line, out_line]

    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {operator}, {case['label']} — "
        f"{rows} x {columns} of tile {tile}, {case['dtype']}, eps {eps:g}"
        f"{', interpreter cast truncated' if case['truncate'] else ''}"
    )
    for line in lines:
        print(line)
    return outside


def cases(flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {"dtype": flags.dtype, "eps": flags.eps, "pdl": flags.pdl == "true"}
    return [
        {
            **common,
            "label": "the fixture's shape, one row per program",
            "rows": flags.rows,
            "columns": flags.columns,
            "tile": flags.tile,
            "flat_row": True,
            "offset_row": True,
            "cancelling_row": True,
            "seed": 1,
        },
        {
            **common,
            "label": "a row far shorter than the tile, almost every lane masked",
            "rows": 3,
            "columns": 100,
            "tile": flags.tile,
            "flat_row": False,
            "offset_row": False,
            "cancelling_row": False,
            "seed": 2,
        },
        {
            **common,
            "label": "a width that is a whole number of tiles",
            "rows": 2,
            "columns": 512,
            "tile": 512,
            "flat_row": False,
            "offset_row": True,
            "cancelling_row": False,
            "seed": 3,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=sorted(OPERATORS), default="rms_norm")
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--pdl", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides the residual output: the operator's own (atol 0), or "
        "the same taken against the magnitudes of the two operands it sums",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.columns > arguments.tile:
        parser.error(
            f"a tile of {arguments.tile} lanes cannot cover a row of {arguments.columns}"
        )
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"
    elif not torch.cuda.is_available():
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

    key = (
        f"dtype_{arguments.dtype}_columns_{arguments.columns}"
        f"_tile_{arguments.tile}_pdl_{arguments.pdl}"
    )
    print(f"descriptor: {key} ({arguments.op})")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments):
        # The interpreter's narrow store truncates; simulate it only where it is known.
        case["truncate"] = arguments.interpreter and arguments.dtype == "bf16"
        case["bound"] = arguments.bound
        failures += check(arguments.op, case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
