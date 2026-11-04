#!/usr/bin/env python3
"""Run the scaled_residual program and check the whole stream, in place.

    python scaled_residual_check.py                     # the fixture's 1152 cell, per-channel
    python scaled_residual_check.py --interpreter        # the same on the CPU
    python scaled_residual_check.py --columns 5376 --block 1024

The operator is `hidden += (update [+ bias]) · scale`, where `scale` is a per-channel
vector or a single factor for the whole stream, and it writes **in place**: the residual
tensor it is given is the one that comes back. `columns` is a runtime extent while the
scale's width and the flags are compiled in, so the cases here are what a flat program over
a row-major stream can get wrong: a row width the block does not divide (a block that
straddles two rows, where the channel index must come from the element's own position and
not from the program's), a width that is a multiple of the block, and a single broadcast
factor rather than a vector.

The reference is torch in FP64 and the bound is the operator's: three roundings in the
output type (the biased update, the product, the sum) and no reduction, so a relative
comparison at `3·u_out` — with no tolerance of the check's own choosing.

**That bound is about the terms, not about the sum, and a residual add cancels.** `atol` is
zero, so an element whose sum is near zero has to agree *relatively*: 149 of 4608 elements
of the 1152-wide `f32` case differ by up to `6.8e-5` relative and pass in no implementation,
because `hidden` and `update · scale` are each larger than their sum. The check prints both
counts — the declared bound's, and the same with the floor the three roundings actually
imply (`3·u_out·(|hidden| + |update · scale|)`) — and `--bound` decides the run, defaulting
to `declared`: the tolerance is the operator's to change, not the check's to relax.

Also, as in every narrow store here: Triton's interpreter truncates where the hardware
rounds, so a `bf16` run under the interpreter truncates its reference the same way and says
so; `f16` is device-only.
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

#: The bound's factor, from `residual.rs::tolerance`.
ROUNDINGS = 3.0


def check(case, device):
    """Add one scaled update into a residual in place and compare every element."""
    import scaled_residual as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, columns = case["rows"], case["columns"]
    hidden = torch.randn(rows, columns, generator=generator).to(dtype)
    update = torch.randn(rows, columns, generator=generator).to(dtype)
    bias = (
        torch.randn(columns, generator=generator).to(dtype)
        if case["update_bias"]
        else None
    )
    width = columns if case["per_channel"] else 1
    scale = torch.randn(width, generator=generator).to(dtype)
    before = hidden.clone()

    elements = rows * columns
    block = case["block"]
    kernels._scaled_residual_kernel[(math_ceil(elements, block),)](
        hidden,
        update,
        bias if bias is not None else update,
        scale,
        elements,
        HAS_UPDATE_BIAS=bias is not None,
        WIDTH=width,
        BLOCK=block,
    )

    exact_update = update.cpu().double()
    if bias is not None:
        exact_update = exact_update + bias.cpu().double()
    expected = before.cpu().double() + exact_update * scale.cpu().double()
    if case["truncate"]:
        # Triton's interpreter truncates a narrow store instead of rounding to nearest
        # even; the reference takes the same cast so the run compares arithmetic.
        bits = expected.float().view(torch.int32)
        expected = (bits & ~0xFFFF).view(torch.float32).to(dtype).double()
    rounding = ROUNDINGS * UNIT_ROUNDOFF[case["dtype"]]
    # The three roundings are of the terms, whose magnitudes the sum can cancel away.
    terms = (
        before.cpu().double().abs()
        + (exact_update * scale.cpu().double()).abs()
    )
    got = hidden.cpu().double()
    difference = (got - expected).abs()
    declared_outside = (difference > rounding * expected.abs()).sum().item()
    scaled_outside = (difference > rounding * terms).sum().item()
    largest = (difference / expected.abs().clamp(min=1e-30)).max().item()
    deciding = (
        declared_outside if case["bound"] == "declared" else scaled_outside
    )
    print(
        f"{'exact' if deciding == 0 else 'FAIL'}: {rows} x {columns} of block "
        f"{block}, {case['dtype']}, bias {str(case['update_bias']).lower()}, "
        f"{'per-channel' if case['per_channel'] else 'one factor'} scale"
        f"{', interpreter cast truncated' if case['truncate'] else ''} — "
        f"max absolute {difference.max().item():.3e}, max relative {largest:.3e}, "
        f"declared {declared_outside} outside, on the terms' magnitudes "
        f"{scaled_outside} outside"
    )
    return deciding


def math_ceil(total, block):
    """`ceil(total / block)` without importing math for one use."""
    return -(-total // block)


def cases(flags):
    """The cases one run checks, cheapest and most diverse first."""
    return [
        {
            "label": "the fixture's width: 1152 in blocks of 1024, so blocks straddle rows",
            "dtype": flags.dtype,
            "rows": flags.rows,
            "columns": flags.columns,
            "block": flags.block,
            "update_bias": flags.update_bias == "true",
            "per_channel": True,
            "seed": 1,
            "bound": flags.bound,
            "truncate": False,
        },
        {
            "label": "one factor for the whole stream",
            "dtype": flags.dtype,
            "rows": 3,
            "columns": flags.columns,
            "block": flags.block,
            "update_bias": False,
            "per_channel": False,
            "seed": 2,
            "bound": flags.bound,
            "truncate": False,
        },
        {
            "label": "a width that is a whole number of blocks",
            "dtype": flags.dtype,
            "rows": 2,
            "columns": 512,
            "block": 512,
            "update_bias": True,
            "per_channel": True,
            "seed": 3,
            "bound": flags.bound,
            "truncate": False,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--columns", type=int, default=1152)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument(
        "--update-bias", choices=["true", "false"], default="false"
    )
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides the run: the operator's own (atol 0, relative to the "
        "sum), or the same taken against the magnitudes of the terms it cancels",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"
    elif not torch.cuda.is_available():
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

    key = (
        f"dtype_{arguments.dtype}_columns_{arguments.columns}"
        f"_block_{arguments.block}_update_bias_{arguments.update_bias}"
    )
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    device = "cpu" if arguments.interpreter else "cuda"
    outside = 0
    for case in cases(arguments):
        # The interpreter's narrow store truncates; simulate it only where it is known.
        case["truncate"] = arguments.interpreter and arguments.dtype == "bf16"
        outside += check(case, device)
    return 1 if outside else 0


if __name__ == "__main__":
    sys.exit(main())
