#!/usr/bin/env python3
"""Run one of the SwiGLU programs and check its output and its magnitude record.

    python swiglu_check.py                                  # this package's own operator
    python swiglu_check.py --op swiglu --value-stride 8192  # rows that are not adjacent
    python swiglu_check.py --interpreter                     # the same on the CPU

One script covers both because they are one operation over two layouts: `value · silu(gate)`,
optionally with a bias on each half and optionally recording each program's largest output
magnitude for a later quantization. `value_first_swiglu` reads a **packed projection** whose
rows start with the value half and end with the gate half, one bias of `2 · width`; `swiglu`
reads separate value and gate views, each with its own bias and its own row stride. The same
file is kept in both packages, byte for byte, and a test in
`crates/program-server/tests/it/packages.rs` holds the copies together.

Cases. Random values; gates swept to ±80, where the sigmoid has saturated and `silu` of a very
negative gate is already ~1e-35 while `exp(-gate)` is still finite in FP32; and a zeroed row,
whose output and magnitude are exactly zero. A fourth is reported and does not fail, because
what it measures is the formula's own limit — see below.

**Every stride a case builds is a different number from the width, and from the other stride.**
A stride that equals the width makes a wrong address arithmetic indistinguishable from the
right one: the mutation is invisible, and a mutation that passes has found a case rather than
proved anything. With `value_stride` twice the width and `gate_stride` equal to it, a kernel
that assumed rows are adjacent reads the wrong row's channels.

**A fourth scenario is reported and does not fail, because what it measures is the formula's
own limit.** Every FP32 implementation of `silu` as `gate / (1 + exp(-gate))` returns zero once
`-gate` passes `log(FLT_MAX) ≈ 88.7`, since `exp` overflows to infinity and the ratio
underflows: at `gate = −89.5` the exact value is `3.3e-37` and the kernel stores `0`. A
careful form (`gate · exp(gate) / (1 + exp(gate))` for negative gates) would keep it, and so
would any absolute term in the bound, but neither is this program's to choose: with
`atol = 0` the difference of `3e-37` reads as a full miss. The run prints how many lanes are
past FP32's `exp` so the number is visible and never mistaken for a passing comparison.

The bound is each operator's own, from `activation.rs::tolerance`: five roundings in the output
type for the separate form (two bias adds, the sigmoid, its product with the gate, the product
with the value) and four for the packed one, which has a single bias. The magnitudes are
compared **exactly**: a maximum of values already in the output type has nothing to round.

A `bf16` store under Triton's interpreter truncates where hardware rounds, and an `f16` store
there rounds; the reference takes the same cast as the run's store, so what is compared is the
arithmetic and not the interpreter's store model.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The pair's other package, so the harness finds its sibling wherever it is run from.
SIBLINGS = ("swiglu", "value_first_swiglu")

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

#: The operators this script runs, and the bound's factor from `activation.rs::tolerance`.
OPERATORS = ("swiglu", "value_first_swiglu")
ROUNDINGS = {"swiglu": 5.0, "value_first_swiglu": 4.0}

#: `log` of FP32's largest finite value: past it `exp(-gate)` overflows.
EXP_LIMIT = 88.72

SCENARIOS = ("random values", "saturating gates", "a zeroed row")


def default_op():
    """The operator this copy belongs to, from the package it is kept in."""
    package = pathlib.Path(__file__).resolve().parents[2].name
    return package if package in OPERATORS else OPERATORS[0]


def silu(gate):
    """`gate · sigmoid(gate)` in torch, in FP64."""
    return gate * torch.sigmoid(gate)


def truncate(values):
    """The reference as Triton's interpreter stores it: one ulp low, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own mantissa.
    """
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def store(values, dtype, truncated):
    """A value as the run's store keeps it: truncated, rounded, or already exact."""
    if dtype == "f32":
        return values
    if dtype == "bf16" and truncated:
        return truncate(values)
    return values.to(TORCH_TYPES[dtype]).double()


def magnitude_failures(partials, expected, block, programs):
    """Count the programs whose recorded magnitude is not their own lanes' largest."""
    record = partials.cpu().double()
    flat = expected.reshape(-1)
    failures = 0
    for program in range(programs):
        lanes = flat[program * block : (program + 1) * block]
        want = lanes.abs().max().item() if lanes.numel() else 0.0
        have = record[program].item()
        if have != want:
            failures += 1
            if failures == 1:
                print(
                    f"  program {program}: magnitude {have:.9e}, the lanes' largest "
                    f"{want:.9e}"
                )
    return failures


def check(operator, scenario, case):
    """Evaluate one layout and compare the output and the magnitudes."""
    dtype = TORCH_TYPES[case["dtype"]]
    truncated = case["truncate"]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, width, block = case["rows"], case["width"], case["block"]
    elements = rows * width
    programs = -(-elements // block)
    out = torch.full((rows, width), float("nan"), dtype=dtype)
    partials = torch.full((programs,), float("nan"), dtype=torch.float32)

    if operator == "value_first_swiglu":
        import value_first_swiglu as kernels

        stride = case["input_stride"]
        assert stride >= 2 * width, "the packed row does not hold both halves"
        packed_buffer = torch.randn(rows, stride, generator=generator)
        if scenario == "saturating gates":
            packed_buffer[:, width:] = (
                torch.linspace(-80.0, 80.0, width).unsqueeze(0).repeat(rows, 1)
            )
        if scenario == "beyond fp32's exp":
            packed_buffer[:, width:] *= 100.0
        if scenario == "a zeroed row":
            packed_buffer[0, :] = 0.0
        packed_buffer = packed_buffer.to(dtype)
        packed = packed_buffer[:, : 2 * width]
        if packed.stride(0) != stride:
            raise AssertionError(
                f"the packed view's row stride is {packed.stride(0)}, not {stride}"
            )
        bias = (
            torch.randn(2 * width, generator=generator).to(dtype)
            if case["bias"]
            else None
        )
        kernels._value_first_swiglu_kernel[(programs,)](
            packed,
            bias if bias is not None else out.reshape(-1),
            out,
            partials if case["absmax"] else out.reshape(-1),
            elements,
            stride,
            width=width,
            HAS_BIAS=bias is not None,
            RETURN_ABSMAX=case["absmax"],
            BLOCK=block,
        )
        exact = packed_buffer.cpu().double()
        exact_value = exact[:, :width]
        exact_gate = exact[:, width : 2 * width]
        if bias is not None:
            exact_bias = bias.cpu().double()
            exact_value = exact_value + exact_bias[:width]
            exact_gate = exact_gate + exact_bias[width:]
        label = f"dtype {case['dtype']}, bias {str(case['bias']).lower()}"
        strides = f"of stride {stride}"
    else:
        import swiglu as kernels

        # Rows that are not adjacent: each half is its own buffer with its own stride, and the
        # padding holds a value a wrong stride would read.
        value_buffer = torch.randn(
            rows, case["value_stride"], generator=generator
        )
        gate_buffer = torch.randn(
            rows, case["gate_stride"], generator=generator
        )
        if scenario == "saturating gates":
            gate_buffer[:, :width] = torch.linspace(-80.0, 80.0, width)
        if scenario == "beyond fp32's exp":
            gate_buffer *= 100.0
        if scenario == "a zeroed row":
            value_buffer[0, :] = 0.0
            gate_buffer[0, :] = 0.0
        value_buffer = value_buffer.to(dtype)
        gate_buffer = gate_buffer.to(dtype)
        value = value_buffer[:, :width]
        gate = gate_buffer[:, :width]
        value_bias = (
            torch.randn(width, generator=generator).to(dtype)
            if case["value_bias"]
            else None
        )
        gate_bias = (
            torch.randn(width, generator=generator).to(dtype)
            if case["gate_bias"]
            else None
        )
        kernels._swiglu_kernel[(programs,)](
            value,
            gate,
            value_bias if value_bias is not None else out.reshape(-1),
            gate_bias if gate_bias is not None else out.reshape(-1),
            out,
            partials if case["absmax"] else out.reshape(-1),
            elements,
            width=width,
            value_row_stride=case["value_stride"],
            gate_row_stride=case["gate_stride"],
            HAS_VALUE_BIAS=value_bias is not None,
            HAS_GATE_BIAS=gate_bias is not None,
            RETURN_ABSMAX=case["absmax"],
            BLOCK=block,
        )
        exact_value = value_buffer.cpu().double()[:, :width]
        exact_gate = gate_buffer.cpu().double()[:, :width]
        if value_bias is not None:
            exact_value = exact_value + value_bias.cpu().double()
        if gate_bias is not None:
            exact_gate = exact_gate + gate_bias.cpu().double()
        label = (
            f"dtype {case['dtype']}, value_bias {str(case['value_bias']).lower()}, "
            f"gate_bias {str(case['gate_bias']).lower()}"
        )
        strides = (
            f"of value stride {case['value_stride']} and gate stride "
            f"{case['gate_stride']}"
        )

    expected = exact_value * silu(exact_gate)
    if scenario == "beyond fp32's exp":
        # Only the report matters here: the formula cannot represent these values.
        got = out.cpu().double()
        beyond = int((exact_gate.abs() > EXP_LIMIT).sum().item())
        stored_zero = int((got == 0.0).sum().item())
        print(
            f"report only: {beyond} of {expected.numel()} gates are past FP32's exp, "
            f"where the exact silu is below {torch.finfo(torch.float32).tiny:.1e} and the "
            f"formula returns zero ({stored_zero} outputs are zero)"
        )
        return 0

    expected = store(expected, case["dtype"], truncated)
    got = out.cpu().double()
    difference = (got - expected).abs()
    allowed = (
        ROUNDINGS[operator] * UNIT_ROUNDOFF[case["dtype"]] * expected.abs()
    )
    outside = (~(difference <= allowed)).sum().item()
    largest = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    magnitudes = (
        magnitude_failures(partials, expected, block, programs)
        if case["absmax"]
        else 0
    )

    print(
        f"{'exact' if outside == 0 and magnitudes == 0 else 'FAIL'}: {operator}, "
        f"{scenario} — {rows} x {width} {strides}, block {block}, {label}, "
        f"absmax {str(case['absmax']).lower()}"
        f"{', interpreter cast truncated' if truncated else ''} — "
        f"max relative error {largest:.3e}, {outside} of {expected.numel()} outside"
        + (
            f", {magnitudes} of {programs} magnitudes differ"
            if case["absmax"]
            else ""
        )
    )
    return outside + magnitudes


def cases(operator, flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {
        "dtype": flags.dtype,
        "rows": flags.rows,
        "width": flags.width,
        "block": flags.block,
        "absmax": flags.absmax == "true",
    }
    if operator == "value_first_swiglu":
        return [
            {
                **common,
                "input_stride": flags.input_stride,
                "bias": flags.bias == "true",
            }
        ]
    return [
        {
            **common,
            "value_stride": flags.value_stride or 2 * flags.width,
            "gate_stride": flags.gate_stride or flags.width,
            "value_bias": flags.value_bias == "true",
            "gate_bias": flags.gate_bias == "true",
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPERATORS, default=default_op())
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument(
        "--input-stride",
        type=int,
        default=8192,
        help="value_first_swiglu: elements between the packed rows",
    )
    parser.add_argument(
        "--value-stride",
        type=int,
        default=None,
        help="swiglu: elements between the value rows; twice the width by default, so a "
        "kernel that assumes rows are adjacent reads the wrong channels",
    )
    parser.add_argument(
        "--gate-stride",
        type=int,
        default=None,
        help="swiglu: elements between the gate rows; the width by default, so the two "
        "strides differ from each other as well as from the rows' extent",
    )
    parser.add_argument("--bias", choices=["true", "false"], default="false")
    parser.add_argument(
        "--value-bias", choices=["true", "false"], default="false"
    )
    parser.add_argument(
        "--gate-bias", choices=["true", "false"], default="false"
    )
    parser.add_argument("--absmax", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if (
        arguments.op == "value_first_swiglu"
        and arguments.input_stride < 2 * arguments.width
    ):
        parser.error(
            f"a packed row of stride {arguments.input_stride} holds no room for "
            f"{arguments.width} values and {arguments.width} gates"
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

    if arguments.op == "value_first_swiglu":
        key = (
            f"dtype_{arguments.dtype}_width_{arguments.width}"
            f"_block_{arguments.block}_input_stride_{arguments.input_stride}"
            f"_bias_{arguments.bias}_absmax_{arguments.absmax}"
        )
    else:
        key = (
            f"dtype_{arguments.dtype}_width_{arguments.width}"
            f"_block_{arguments.block}"
            f"_value_stride_{arguments.value_stride or 2 * arguments.width}"
            f"_gate_stride_{arguments.gate_stride or arguments.width}"
            f"_value_bias_{arguments.value_bias}_gate_bias_{arguments.gate_bias}"
            f"_absmax_{arguments.absmax}"
        )
    print(f"descriptor: {key} ({arguments.op})")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments.op, arguments):
        case["truncate"] = arguments.interpreter and arguments.dtype == "bf16"
        for seed, scenario in enumerate(
            SCENARIOS + ("beyond fp32's exp",), start=1
        ):
            case["seed"] = seed
            failures += check(arguments.op, scenario, case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
