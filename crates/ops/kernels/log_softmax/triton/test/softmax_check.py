#!/usr/bin/env python3
"""Run one of the row-softmax programs and check every row against torch.

    python softmax_check.py --op softmax                       # 2048 of a 2048 tile
    python softmax_check.py --op log_softmax --interpreter      # the same on the CPU
    python softmax_check.py --op log_softmax --rows 2           # the fixture's vocab row

One script covers both because they are one row reduction with one difference: `softmax`
stores the exponentials' share of their sum in the input's type, while `log_softmax` walks
the row in `STEP`-wide tiles — so a tile larger than a program's registers costs more steps,
not a bigger block — and stores the log-probability in FP32 whatever the input's type. The
same file is kept in both packages, byte for byte, and a test in
`crates/program-server/tests/it/packages.rs` holds them together.

The cases are what a row reduction can get wrong: the shapes the fixture's keys name, a
width far below the tile where almost every lane is masked, a row of one repeated logit
(a uniform distribution, whose probabilities are `1/columns`), and a row with one dominant
logit a hundred nats above the rest, where the probabilities are `1` and `0` and the
log-probabilities are `0` and `−100` — the case a relative comparison cannot survive.

The reference is torch in FP64 and the bound is the operator's own — `rounding(columns,
F32, out)` for `softmax`, `rounding(columns + 1, F32, F32)` for `log_softmax` — with no
tolerance of the check's own choosing.

**`log_softmax` subtracts, so it is one where an absolute term could be needed and the
declared bound has none.** `atol` is zero, so a log-probability near zero — the chosen
token's own, `x − logsumexp`, which a sampling step reads — must agree *relatively*. In the
runs here it did: a log-probability below FP32's ulp of the normalizer is exactly zero in
both the kernel and the reference, and the bound's `γ(columns)` is a wide `1e-4` at a
hundred thousand columns, well above the subtraction's error. The floor the subtraction
implies (`u_out · (|x| + |logsumexp|)`) is computed and printed beside the declared count so
the number stays visible, and `--bound scale` applies it if a future case needs it.

A `bf16` input run under Triton's interpreter truncates its reference the way the interpreter
truncates narrow stores; `f16` is device-only.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The output types' unit roundoff, as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The operators this script runs, and the extra roundings each adds to the reduction.
OPERATORS = {"softmax": 0, "log_softmax": 1}

#: The widest tile `log_softmax`'s build steps a row in, from its `build.py`.
LANES = 4096


def gamma(reduction):
    """`γₙ = n·u/(1 − n·u)` for `n` FP32 roundings, as the operator crate computes it."""
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def bound(columns, out_dtype, extra):
    """The operator's relative bound over a row of `columns` terms."""
    return gamma(columns + extra) + UNIT_ROUNDOFF[out_dtype]


def truncate(values):
    """The reference as Triton's interpreter stores it: one ulp low, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own mantissa.
    """
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def compare(name, got, expected, floor, relative, deciding_bound):
    """Count the elements outside each bound and give the line that reports them."""
    difference = (got - expected).abs()
    declared_outside = (~(difference <= relative * expected.abs())).sum().item()
    scaled_outside = (
        (~(difference <= relative * expected.abs() + floor)).sum().item()
    )
    largest = (difference / expected.abs().clamp(min=1e-30)).nan_to_num(nan=float("inf")).max().item()
    deciding = (
        declared_outside if deciding_bound == "declared" else scaled_outside
    )
    line = (
        f"  {name}: max absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, max relative "
        f"{largest:.3e}, declared {declared_outside} of {expected.numel()} outside, "
        f"with the operands' floor {scaled_outside} outside"
    )
    return deciding, line


def check(operator, case):
    """Run one operator over one matrix and compare every row."""
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = TORCH_TYPES[case["dtype"]]
    rows, columns, tile = case["rows"], case["columns"], case["tile"]

    x = torch.randn(rows, columns, generator=generator)
    if case["dominant_row"]:
        # One logit a hundred nats above the rest: probabilities 1 and 0, log-probs 0 and -100.
        x[0, :] = -100.0
        x[0, columns // 2] = 0.0
    if case["flat_row"]:
        x[0, :] = 0.5
    x = x.to(dtype)

    out_dtype = dtype if operator == "softmax" else torch.float32
    out = torch.full((rows, columns), float("nan"), dtype=out_dtype)
    pdl = int(case["pdl"])
    if operator == "softmax":
        import softmax as kernels

        kernels._softmax_kernel[(rows,)](
            x, out, columns, pdl, BLOCK=tile, PDL=case["pdl"]
        )
    else:
        import log_softmax as kernels

        kernels._log_softmax_kernel[(rows,)](
            x, out, columns, pdl, STEP=min(tile, LANES), PDL=case["pdl"]
        )

    exact = x.cpu().double()
    largest = exact.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(exact - largest)
    total = exponentials.sum(dim=-1, keepdim=True)
    logsumexp = largest + torch.log(total)
    if operator == "softmax":
        expected = exponentials / total
    else:
        expected = exact - logsumexp
    if case["truncate"]:
        expected = truncate(expected)

    relative = bound(
        columns,
        case["dtype"] if operator == "softmax" else "f32",
        OPERATORS[operator],
    )
    if operator == "softmax":
        floor = 0.0
    else:
        # The subtraction's rounding is of the logit and the normalizer, not of the result.
        floor = UNIT_ROUNDOFF["f32"] * (exact.abs() + logsumexp.abs())

    outside, line = compare(
        "out", out.cpu().double(), expected, floor, relative, case["bound"]
    )
    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {operator}, {case['label']} — "
        f"{rows} x {columns} of tile {tile}, {case['dtype']} in"
        f"{', interpreter cast truncated' if case['truncate'] else ''}"
    )
    print(line)
    return outside


def cases(flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {"dtype": flags.dtype, "pdl": flags.pdl == "true"}
    return [
        {
            **common,
            "label": "the fixture's shape",
            "rows": flags.rows,
            "columns": flags.columns,
            "tile": flags.tile,
            "dominant_row": True,
            "flat_row": True,
            "seed": 1,
        },
        {
            **common,
            "label": "a row far shorter than the tile, almost every lane masked",
            "rows": 2,
            "columns": 100,
            "tile": flags.tile,
            "dominant_row": False,
            "flat_row": False,
            "seed": 2,
        },
        {
            **common,
            "label": "a row of one repeated logit, a uniform distribution",
            "rows": 2,
            "columns": 512,
            "tile": 512,
            "dominant_row": False,
            "flat_row": True,
            "seed": 3,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=sorted(OPERATORS), default="softmax")
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="f32")
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--pdl", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides the run: the operator's own (atol 0), or the same with "
        "the floor the log-probability's subtraction implies",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.op == "softmax" and arguments.columns > arguments.tile:
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
        # The interpreter's narrow store truncates; simulate it only where it is known, and
        # only where the output is itself narrow — `log_softmax` stores FP32 whatever comes in.
        case["truncate"] = (
            arguments.interpreter
            and arguments.dtype == "bf16"
            and arguments.op == "softmax"
        )
        case["bound"] = arguments.bound
        failures += check(arguments.op, case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
