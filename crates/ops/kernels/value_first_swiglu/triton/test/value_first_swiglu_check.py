#!/usr/bin/env python3
"""Run the value_first_swiglu program and check its output and its magnitude record.

    python value_first_swiglu_check.py                           # the fixture's 14336 cell
    python value_first_swiglu_check.py --interpreter              # the same on the CPU
    python value_first_swiglu_check.py --width 8192 --input-stride 16384 --dtype f16

The operator reads a packed projection whose rows start with the **value** half and end
with the gate half, `value · silu(gate)`, optionally with the packed bias added to each
half, and optionally records each program's largest output magnitude into an FP32 buffer
for a later quantization. `width` and the two flags are compiled in; the row stride is a
descriptor field, so the packed buffer is always built here as `[rows, input_stride]`
sliced to its first `2 · width` columns — the load must use the buffer's row stride and
not assume rows are adjacent.

Three data scenarios run under one descriptor and decide the run: random values; gates at
±80, where `sigmoid` has saturated and the `silu` of a very negative gate is already ~1e-35
but `exp(-gate)` is still finite in FP32; and a zeroed row, whose output and magnitude are
exactly zero.

**A fourth is reported and does not fail, because what it measures is the formula's own
limit.** Every FP32 implementation of `silu` as `gate / (1 + exp(-gate))` returns zero once
`-gate` passes `log(FLT_MAX) ≈ 88.7`, since `exp` overflows to infinity and the ratio
underflows: at `gate = −89.5` the exact value is `3.3e-37` and the kernel stores `0`. A
careful form (`gate · exp(gate) / (1 + exp(gate))` for negative gates) would keep it, and
so would any absolute term in the bound, but neither is this program's to choose: with
`atol = 0` the difference of `3e-37` reads as a full miss. The run prints how many lanes are
past FP32's `exp` so the number is visible and never mistaken for a passing comparison.

The bound is the operator's — four roundings in the output type (the sigmoid, its product
with the gate, the bias add, the product with the value) and no reduction, so a relative
comparison at `4 · u_out`. The magnitudes are compared **exactly**: a maximum of values
already in the output type has nothing to round.

A `bf16` run under Triton's interpreter truncates its reference the way the interpreter
truncates narrow stores (one ulp, not round-to-nearest) and says so in its output; `f16`
is device-only.
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

#: The bound's factor, from `activation.rs::tolerance`.
ROUNDINGS = 4.0


def silu(gate):
    """`gate · sigmoid(gate)` in torch, in FP64."""
    return gate * torch.sigmoid(gate)


def truncate(values):
    """The reference as Triton's interpreter stores it: one ulp low, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own mantissa.
    """
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def check(scenario, case, device):
    """Evaluate one packed projection and compare the output and the magnitudes."""
    import value_first_swiglu as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, width = case["rows"], case["width"]
    stride = case["input_stride"]
    assert stride >= 2 * width, "the packed row does not hold both halves"

    # The packed buffer carries the descriptor's row stride; the view is its first half.
    packed_buffer = torch.randn(rows, stride, generator=generator)
    if scenario == "saturating gates":
        # The whole finite range of FP32's `exp` is walked, both saturations included and
        # room left for the bias: past ±88.7 a gate is the next scenario's subject, and it
        # is reported rather than compared.
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
    out = torch.full((rows, width), float("nan"), dtype=dtype)
    elements = rows * width
    block = case["block"]
    programs = -(-elements // block)
    partials = torch.full((programs,), float("nan"), dtype=torch.float32)

    if bias is not None:
        # The bias arrives as `[value_bias, gate_bias]`; the kernel indexes the halves.
        bias_pointer = bias
    else:
        bias_pointer = out.reshape(-1)
    kernels._value_first_swiglu_kernel[(programs,)](
        packed,
        bias_pointer,
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
    expected = exact_value * silu(exact_gate)
    if case["truncate"]:
        # Triton's interpreter truncates a narrow store; the reference takes the same cast
        # so that what is compared is the arithmetic and not the interpreter's store. The
        # truncation is of the value's own mantissa: rounding to `f32` first would step
        # values across a boundary the kernel's arithmetic stays below.
        expected = truncate(expected)

    got = out.cpu().double()
    if scenario == "beyond fp32's exp":
        # Only the report matters here: the formula cannot represent these values.
        beyond = int((exact_gate.abs() > EXP_LIMIT).sum().item())
        stored_zero = int((got == 0.0).sum().item())
        print(
            f"report only: {beyond} of {expected.numel()} gates are past FP32's exp, "
            f"where the exact silu is below {torch.finfo(torch.float32).tiny:.1e} and the "
            f"formula returns zero ({stored_zero} outputs are zero)"
        )
        return 0
    difference = (got - expected).abs()
    allowed = ROUNDINGS * UNIT_ROUNDOFF[case["dtype"]] * expected.abs()
    outside = (difference > allowed).sum().item()
    largest = (difference / expected.abs().clamp(min=1e-30)).max().item()

    magnitude_failures = 0
    if case["absmax"]:
        # The record is a maximum over the program's own lanes, padded lanes counting zero.
        record = partials.cpu().double()
        flat = expected.to(dtype).reshape(-1)
        for program in range(programs):
            lanes = flat[program * block : (program + 1) * block]
            want = lanes.abs().max().item() if lanes.numel() else 0.0
            have = record[program].item()
            if have != want:
                magnitude_failures += 1
                if magnitude_failures == 1:
                    print(
                        f"  program {program}: magnitude {have:.9e}, the lanes' largest "
                        f"{want:.9e}"
                    )

    print(
        f"{'exact' if outside == 0 and magnitude_failures == 0 else 'FAIL'}: "
        f"{scenario} — {rows} x {width} of stride {stride}, block {block}, "
        f"{case['dtype']}, bias {str(case['bias']).lower()}, "
        f"absmax {str(case['absmax']).lower()}"
        f"{', interpreter cast truncated' if case['truncate'] else ''} — "
        f"max relative error {largest:.3e}, {outside} of {expected.numel()} outside"
        + (
            f", {magnitude_failures} of {programs} magnitudes differ"
            if case["absmax"]
            else ""
        )
    )
    return outside + magnitude_failures


SCENARIOS = ("random values", "saturating gates", "a zeroed row")

#: `log` of FP32's largest finite value: past it `exp(-gate)` overflows.
EXP_LIMIT = 88.72


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--width", type=int, default=14336)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--input-stride", type=int, default=28672)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--bias", choices=["true", "false"], default="false")
    parser.add_argument("--absmax", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.input_stride < 2 * arguments.width:
        parser.error(
            f"a packed row of stride {arguments.input_stride} holds no room for "
            f"{arguments.width} values and {arguments.width} gates"
        )
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"
    elif not torch.cuda.is_available():
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

    key = (
        f"dtype_{arguments.dtype}_width_{arguments.width}_block_{arguments.block}"
        f"_input_stride_{arguments.input_stride}_bias_{arguments.bias}"
        f"_absmax_{arguments.absmax}"
    )
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    device = "cpu" if arguments.interpreter else "cuda"
    failures = 0
    for seed, scenario in enumerate(
        SCENARIOS + ("beyond fp32's exp",), start=1
    ):
        case = {
            "dtype": arguments.dtype,
            "rows": arguments.rows,
            "width": arguments.width,
            "block": arguments.block,
            "input_stride": arguments.input_stride,
            "bias": arguments.bias == "true",
            "absmax": arguments.absmax == "true",
            "seed": seed,
            # The interpreter's narrow store truncates; simulate it only where it is known.
            "truncate": arguments.interpreter and arguments.dtype == "bf16",
        }
        failures += check(scenario, case, device)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
