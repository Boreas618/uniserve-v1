#!/usr/bin/env python3
"""Run one of the fused residual-and-normalization programs and check every row.

    python fused_row_norm_check.py --op scaled_residual_rms_norm
    python fused_row_norm_check.py --op scaled_residual_layer_norm --interpreter
    python fused_row_norm_check.py --op scaled_residual_rms_norm --absmax true

One script covers both because they are one row reduction behind one difference: both fold a
channel-scaled, optionally biased update into the residual, then normalize — the RMS form by
the row's root mean square and the layer form by its mean and variance, adding the learned
bias afterwards. The RMS form also writes the updated residual back into `hidden`; the layer
form leaves the residual unchanged, which its arguments say in as many words ("unchanged by
the launch"). The same file is kept in both packages, byte for byte, and a test in
`crates/program-server/tests/it/packages.rs` holds the copies together.

Cases: the deployed geometry, the `update_bias` and `absmax` flags on, a residual that
cancels the update exactly (the RMS of zero, and a layer output that is nothing but the bias —
where the normalized value is exactly zero and a relative bound has nothing to divide), a row
of one repeated value, and — for the **RMS form only** — a row with a large common offset.

That last exclusion is a property of the arithmetic, not a gap: a layer norm over a row of
values near a thousand with unit-scale variation is ill-conditioned in FP32 by construction,
because the mean's own rounding (`γ(columns)·1024 ≈ 0.1`) shifts every centered value by more
than the variance it divides by, so no implementation of the declared bound meets an exact
reference there. The RMS form has no mean and no such case, and it is checked with the offset.

`columns` and `tile` are both baked, and the tile must cover the row: a module whose tile is
narrower than its columns would silently normalize a prefix. That is the descriptor's
business rather than the check's — a check passes the module's own descriptor — so it is said
here and not tested.

The bound is the operator's own: the reduction over the row in FP32 plus three roundings of
the gated update in the output type. `atol` is zero, so a cancelled row must agree relatively
against nothing; each run prints both counts, the declared bound's and one with the floor the
operands imply, and `--bound` decides the run, defaulting to `declared`.

**And the residual's own rounding is the compiler's to choose**, which is why the floors
above exist rather than a mirrored reference: `hidden + update · scale` is one fused
multiply-add on a device and two roundings in Triton's interpreter, so a reference that
mirrors either one is wrong on the other. In `bf16` the declared bound absorbs it; in `f32`
it does not, and `f32` runs are compared with `--bound scale` for that reason — the number is
printed either way.

A `bf16` run under Triton's interpreter truncates narrow stores — the updated residual and the
normalized row — so the reference is truncated the same way, on the value's own mantissa.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The family's other package, so the harness finds its sibling wherever it is run from.
SIBLINGS = (
    "scaled_residual_rms_norm",
    "scaled_residual_layer_norm",
)

for _sibling in SIBLINGS:
    _source = (
        pathlib.Path(__file__).resolve().parents[3]
        / _sibling
        / "triton"
        / "src"
    )
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

#: The output types' unit roundoff, as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The operators this script runs.
OPERATORS = ("scaled_residual_rms_norm", "scaled_residual_layer_norm")


def default_op():
    """The operator this copy belongs to, from the package it is kept in."""
    package = pathlib.Path(__file__).resolve().parents[2].name
    return package if package in OPERATORS else OPERATORS[0]


def gamma(reduction):
    """`γₙ = n·u/(1 − n·u)` for `n` FP32 roundings, as the operator crate computes it."""
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def truncate(values):
    """The reference as Triton's interpreter stores it: one ulp low, not rounded."""
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def compare(name, got, expected, relative, floor, deciding_bound):
    """Count the elements outside each bound and give the line that reports them."""
    difference = (got - expected).abs()
    declared_outside = (~(difference <= relative * expected.abs())).sum().item()
    scaled_outside = (
        (~(difference <= relative * expected.abs() + floor)).sum().item()
    )
    largest = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    deciding = (
        declared_outside if deciding_bound == "declared" else scaled_outside
    )
    line = (
        f"  {name}: max absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, "
        f"max relative {largest:.3e}, declared {declared_outside} of {expected.numel()} "
        f"outside, with the operands' floor {scaled_outside} outside"
    )
    return deciding, line


def check(operator, case):
    """Fuse one update into a residual and compare the residual and the normalized row."""
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = TORCH_TYPES[case["dtype"]]
    rows, columns, tile = case["rows"], case["columns"], case["tile"]
    rms = operator == "scaled_residual_rms_norm"

    hidden = torch.randn(rows, columns, generator=generator).to(dtype)
    update = torch.randn(rows, columns, generator=generator).to(dtype)
    scale = torch.randn(columns, generator=generator).to(dtype)
    weight = torch.randn(columns, generator=generator).to(dtype)
    update_bias = (
        torch.randn(columns, generator=generator).to(dtype)
        if case["update_bias"]
        else None
    )
    bias = torch.randn(columns, generator=generator).to(dtype)
    if case["cancelling_row"]:
        # `hidden + update · scale` is exactly zero on the first row: the RMS form has an RMS
        # of zero and the layer form an output that is nothing but the bias.
        hidden[0, :] = (-update[0].double() * scale.double()).to(dtype)
    if case["flat_row"]:
        hidden[-1, :] = 2.0
    if case["offset_row"] and rms:
        hidden[-1, :] = 0.0
        update[-1, :] = (1000.0 / scale.double()).to(dtype)
    out = torch.full((rows, columns), float("nan"), dtype=dtype)
    partials = torch.full((rows,), float("nan"), dtype=torch.float32)
    eps = case["eps"]
    # The residual as the kernel sees it: the launch overwrites `hidden` in the RMS form, so
    # the operand is kept before the call rather than rebuilt afterwards.
    hidden_before = hidden.cpu().double()

    if rms:
        import scaled_residual_rms_norm as kernels

        name = (
            kernels._scaled_residual_rms_norm_absmax_kernel
            if case["absmax"]
            else kernels._scaled_residual_rms_norm_kernel
        )
        arguments = [hidden, update, update_bias, scale, weight, out]
        if case["absmax"]:
            arguments.append(partials)
        arguments.append(eps)
        name[(rows,)](
            *arguments,
            HAS_UPDATE_BIAS=update_bias is not None,
            WIDTH=columns,
            BLOCK=tile,
        )
    else:
        import scaled_residual_layer_norm as kernels

        name = (
            kernels._scaled_residual_layer_norm_absmax_kernel
            if case["absmax"]
            else kernels._scaled_residual_layer_norm_kernel
        )
        arguments = [hidden, update, update_bias, scale, weight, bias, out]
        if case["absmax"]:
            arguments.append(partials)
        arguments.append(eps)
        name[(rows,)](
            *arguments,
            HAS_UPDATE_BIAS=update_bias is not None,
            WIDTH=columns,
            BLOCK=tile,
        )

    exact_update = update.cpu().double()
    if update_bias is not None:
        exact_update = exact_update + update_bias.cpu().double()
    # The residual in exact arithmetic. The kernel computes it in FP32, and **how many
    # roundings that takes is the compiler's**: `hidden + update · scale` is one fused
    # multiply-add on a device and two roundings in Triton's interpreter, so a reference
    # that mirrors either one is wrong on the other. The difference is carried in the floors
    # below as one rounding of the accumulate type — which is also why a row whose sum
    # cancels needs the floor: there the whole value is that rounding.
    residual = hidden_before + exact_update * scale.cpu().double()
    if rms:
        inverse = torch.rsqrt(
            (residual * residual).sum(dim=-1, keepdim=True) / columns + eps
        )
        expected_out = residual * inverse * weight.cpu().double()
        expected_hidden = residual
    else:
        mean = residual.mean(dim=-1, keepdim=True)
        centered = residual - mean
        inverse = torch.rsqrt(
            (centered * centered).sum(dim=-1, keepdim=True) / columns + eps
        )
        expected_out = (
            centered * inverse * weight.cpu().double() + bias.cpu().double()
        )
        expected_hidden = None

    relative = gamma(columns) + 3.0 * UNIT_ROUNDOFF[case["dtype"]]
    floors = {}
    if case["truncate"]:
        expected_out = truncate(expected_out)
        if expected_hidden is not None:
            expected_hidden = truncate(expected_hidden)
    # One FP32 rounding of the residual's two operands: the difference between a fused
    # multiply-add and two roundings, which is the compiler's choice rather than the module's.
    residual_floor = UNIT_ROUNDOFF["f32"] * (
        hidden_before.abs() + (exact_update * scale.cpu().double()).abs()
    )
    if expected_hidden is not None:
        # The stored residual is a cast of an FP32 value the floor above already widens.
        floors["hidden"] = (
            UNIT_ROUNDOFF[case["dtype"]] * residual.abs() + residual_floor
        )
    # The normalized value is a product scaled by the row's inverse deviation and, in the layer
    # form, shifted by the learned bias — whose magnitude is what a near-zero output is
    # compared against. The residual's own rounding reaches it through that scale.
    floors["out"] = UNIT_ROUNDOFF[case["dtype"]] * (
        expected_out.abs() + (0.0 if rms else bias.cpu().double().abs())
    ) + residual_floor * (weight.cpu().double().abs() * inverse)

    outside = 0
    lines = []
    if expected_hidden is not None:
        counted, line = compare(
            "hidden",
            hidden.cpu().double(),
            expected_hidden,
            UNIT_ROUNDOFF[case["dtype"]],
            floors["hidden"],
            case["bound"],
        )
        outside += counted
        lines.append(line)
    counted, line = compare(
        "out",
        out.cpu().double(),
        expected_out,
        relative,
        floors["out"],
        case["bound"],
    )
    outside += counted
    lines.append(line)

    if case["absmax"]:
        # The record is each row's largest stored magnitude; a maximum of values already in
        # the output type has nothing to round, so it is compared exactly.
        record = partials.cpu().double()
        wanted = out.cpu().double().abs().max(dim=-1).values
        wrong = int((record != wanted).sum())
        outside += wrong
        lines.append(f"  partials: {wrong} of {rows} magnitudes differ")

    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {operator}, {case['label']} — "
        f"{rows} x {columns} of tile {tile}, {case['dtype']}, eps {eps:g}, "
        f"update_bias {str(case['update_bias']).lower()}, "
        f"absmax {str(case['absmax']).lower()}"
        f"{', interpreter cast truncated' if case['truncate'] else ''}"
    )
    for line in lines:
        print(line)
    return outside


def cases(operator, flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {
        "dtype": flags.dtype,
        "columns": flags.columns,
        "tile": flags.tile,
        "eps": flags.eps,
    }
    return [
        {
            **common,
            "label": "the deployed geometry",
            "rows": flags.rows,
            "update_bias": flags.update_bias == "true",
            "absmax": flags.absmax == "true",
            "cancelling_row": True,
            "flat_row": True,
            "offset_row": False,
            "seed": 1,
        },
        {
            **common,
            # A layer norm over a row near a thousand with unit variation is ill-conditioned in
            # FP32 by construction, so the offset belongs to the RMS form's cases alone.
            "label": (
                "a row with a large common offset"
                if operator == "scaled_residual_rms_norm"
                else "a row of ordinary values"
            ),
            "rows": 2,
            "update_bias": False,
            "absmax": False,
            "cancelling_row": False,
            "flat_row": False,
            "offset_row": True,
            "seed": 2,
        },
        {
            **common,
            "label": "one row, with the magnitude record on",
            "rows": 1,
            "columns": min(flags.columns, 1024),
            "tile": min(flags.tile, 1024),
            "update_bias": True,
            "absmax": True,
            "cancelling_row": False,
            "flat_row": False,
            "offset_row": False,
            "seed": 3,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPERATORS, default=default_op())
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--columns", type=int, default=1536)
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--update-bias", choices=["true", "false"], default="false"
    )
    parser.add_argument("--absmax", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides the run: the operator's own (atol 0), or the same with "
        "the floor the operands' magnitudes imply",
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
        f"dtype_{arguments.dtype}_columns_{arguments.columns}_tile_{arguments.tile}"
        f"_update_bias_{arguments.update_bias}_absmax_{arguments.absmax}"
    )
    print(f"descriptor: {key} ({arguments.op})")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments.op, arguments):
        # The interpreter's narrow store truncates; simulate it only where it is known.
        case["truncate"] = arguments.interpreter and arguments.dtype == "bf16"
        case["bound"] = arguments.bound
        failures += check(arguments.op, case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
