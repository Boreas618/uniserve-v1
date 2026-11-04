#!/usr/bin/env python3
"""Run one of the E4M3 activation programs and check the encoded rows.

    python fp8_check.py --op act_and_mul_fp8 --activation silu
    python fp8_check.py --op value_first_swiglu_fp8
    python fp8_check.py --op act_and_mul_fp8 --stepwise true

One script covers both because they are one thing: an activation's product with the value half,
encoded as E4M3 with one FP32 dequantization scale per row — `act_and_mul_fp8` over separate
gate and value views with a selectable activation, `value_first_swiglu_fp8` over a packed
projection whose rows start with the value half. The same file is kept in both packages, byte for
byte, and a test in `crates/program-server/tests/it/packages.rs` holds the copies together.

**Both need a device, and that is a measurement rather than a convenience.** Storing through a
trivial one-lane kernel into an E4M3 buffer through Triton's interpreter gives `-64` for
`-126.4941`, `-16` for `-31.6235` and `104` for `100.0`, where the format says `-128`, `-32` and
`96`: not a truncation and not a tie-break, so no reference can model it. A green from this
operator under the interpreter would be a statement about a broken store, so on a host without a
device the script says that and stops.

What a device run checks, per row: the dequantized code against the exact activation and product
quantized the same way the module quantizes it (by the row's own scale, inside the format's
range, one E4M3 rounding), with E4M3's unit roundoff as the floor the declared tolerance cannot
carry; and the scale itself against the row's largest magnitude over 448. Cases: an ordinary
row, a row whose largest value is near the scale's epsilon clamp, a zeroed row — whose scale is
the clamp and whose codes are all zero — and a saturating row for the two activations that have
one.
"""

import argparse
import pathlib
import struct
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The pair's other package, so the harness finds its sibling wherever it is run from.
SIBLINGS = ("act_and_mul_fp8", "value_first_swiglu_fp8")

for _sibling in SIBLINGS:
    _source = (
        pathlib.Path(__file__).resolve().parents[3]
        / _sibling
        / "triton"
        / "src"
    )
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

OPERATORS = ("act_and_mul_fp8", "value_first_swiglu_fp8")

#: The activations `act_and_mul_fp8` can apply to its gate half.
ACTIVATIONS = ("silu", "gelu_tanh", "gelu")

#: The format's largest finite magnitude and the scale's floor, from `activation.rs`.
E4M3_MAX = 448.0
SCALE_EPS = 1.0e-12

#: E4M3's unit roundoff: three mantissa bits.
E4M3_ROUNDOFF = 2.0**-4


def default_op():
    """The operator this copy belongs to, from the package it is kept in."""
    package = pathlib.Path(__file__).resolve().parents[2].name
    return package if package in OPERATORS else OPERATORS[0]


def activate(gate, activation):
    """The gate activation in FP64, as the kernel's three forms compute it."""
    if activation == "silu":
        return gate * torch.sigmoid(gate)
    if activation == "gelu_tanh":
        inner = (2.0 / torch.pi) ** 0.5 * (gate + 0.044715 * gate**3)
        return gate * torch.sigmoid(2.0 * inner)
    return 0.5 * gate * (1.0 + torch.erf(gate / 2.0**0.5))


def codes_of(values, scale):
    """The E4M3 codes a row-scoped value takes, and the row it dequantizes back to."""
    scaled = (values / scale).clamp(-E4M3_MAX, E4M3_MAX)
    stamped = scaled.to(torch.float8_e4m3fn)
    return stamped, stamped.float().double() * scale


def check(operator, case):
    """Encode one case's rows and compare the codes' dequantization and the scales."""
    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, width, block = case["rows"], case["width"], case["block"]

    if operator == "value_first_swiglu_fp8":
        import value_first_swiglu_fp8 as kernels

        stride = case["input_stride"]
        assert stride >= 2 * width, "the packed row does not hold both halves"
        packed_buffer = torch.randn(rows, stride, generator=generator)
        if case["scenario"] == "a zeroed row":
            packed_buffer[0, :] = 0.0
        if case["scenario"] == "a tiny row":
            packed_buffer[0, :] *= 1.0e-30
        # The stored buffer is the operand and the reference at once: casting a copy would
        # compare the kernel's arithmetic against values it never saw.
        packed_buffer = packed_buffer.to(dtype)
        packed = packed_buffer[:, : 2 * width]
        out = torch.zeros(rows, width, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(rows, dtype=torch.float32)
        kernels._value_first_swiglu_fp8_kernel[(rows,)](
            packed,
            out,
            scale,
            stride,
            width=width,
            BLOCK=block,
        )
        # The packed row starts with the value half and ends with the gate half.
        exact = packed_buffer.cpu().double()
        value = exact[:, :width]
        gate = exact[:, width : 2 * width]
        expected = value * activate(gate, "silu")
        label = f"{rows} x {width} of stride {stride}"
    else:
        import act_and_mul_fp8 as kernels

        gate_buffer = torch.randn(
            rows, case["gate_stride"], generator=generator
        )
        value_buffer = torch.randn(
            rows, case["value_stride"], generator=generator
        )
        if case["scenario"] == "a zeroed row":
            gate_buffer[0, :] = 0.0
            value_buffer[0, :] = 0.0
        if case["scenario"] == "a tiny row":
            gate_buffer[0, :] *= 1.0e-30
            value_buffer[0, :] *= 1.0e-30
        if case["scenario"] == "a saturating row":
            torch.nn.functional.normalize(
                gate_buffer[0], dim=0, out=gate_buffer[0]
            )
            gate_buffer[0] *= 400.0
        # As above: one tensor is both what the kernel reads and what the reference derives.
        gate_buffer = gate_buffer.to(dtype)
        value_buffer = value_buffer.to(dtype)
        gate_stored = gate_buffer[:, :width]
        value_stored = value_buffer[:, :width]
        out = torch.zeros(rows, width, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(rows, dtype=torch.float32)
        kernels._act_and_mul_fp8_kernel[(rows,)](
            gate_stored,
            value_stored,
            out,
            scale,
            case["gate_stride"],
            case["value_stride"],
            n_cols=width,
            ACTIVATION=case["activation"],
            block=block,
            stepwise=case["stepwise"],
        )
        gate = gate_buffer.cpu().double()[:, :width]
        value = value_buffer.cpu().double()[:, :width]
        # The stepwise recipe rounds the activated gate and the product to the input type
        # before the row's scale encodes them; the plain one keeps both in FP32.
        activated = activate(gate, case["activation"])
        if case["stepwise"]:
            activated = activated.to(dtype).double()
        expected = activated * value
        if case["stepwise"]:
            expected = expected.to(dtype).double()
        label = (
            f"{rows} x {width} of gate stride {case['gate_stride']} and value stride "
            f"{case['value_stride']}, {case['activation']}, "
            f"stepwise {str(case['stepwise']).lower()}"
        )

    # The scale is stored as FP32, so the reference rounds it there too: a stored scale compared
    # against a double one would count the store's own rounding as a disagreement.
    wanted_scale = (
        (
            expected.abs().max(dim=-1, keepdim=True).values.clamp(min=SCALE_EPS)
            / E4M3_MAX
        )
        .to(torch.float32)
        .double()
    )
    _, wanted = codes_of(expected, wanted_scale)
    got_scale = scale.cpu().double().unsqueeze(-1)
    got = out.float().double() * got_scale
    scale_difference = (
        (got_scale.squeeze(-1) != wanted_scale.squeeze(-1)).sum().item()
    )
    difference = (got - wanted).abs()
    floor = E4M3_ROUNDOFF * wanted.abs()
    declared = 4.0 * UNIT_ROUNDOFF[case["dtype"]] + 2.0 * E4M3_ROUNDOFF
    outside = (~(difference <= declared * wanted.abs() + floor)).sum().item()
    print(
        f"{'exact' if outside == 0 and scale_difference == 0 else 'FAIL'}: {operator}, "
        f"{case['scenario']} — {label}, block {block}, {case['dtype']} -> e4m3, "
        f"{outside} of {wanted.numel()} elements outside, "
        f"{scale_difference} of {rows} scales differ"
    )
    return outside + scale_difference


def cases(operator, flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {
        "dtype": flags.dtype,
        "width": flags.width,
        "block": flags.block,
        "rows": flags.rows,
    }
    if operator == "value_first_swiglu_fp8":
        return [
            {
                **common,
                "input_stride": flags.input_stride,
                "scenario": scenario,
                "seed": seed,
            }
            for seed, scenario in enumerate(
                (
                    "an ordinary row",
                    "a tiny row",
                    "a zeroed row",
                ),
                start=1,
            )
        ]
    return [
        {
            **common,
            "gate_stride": flags.gate_stride or 2 * flags.width,
            "value_stride": flags.value_stride or flags.width,
            "activation": flags.activation,
            "stepwise": flags.stepwise == "true",
            "scenario": scenario,
            "seed": seed,
        }
        for seed, scenario in enumerate(
            (
                (
                    "an ordinary row",
                    "a tiny row",
                    "a zeroed row",
                    "a saturating row",
                )
                if operator == "act_and_mul_fp8"
                else ("an ordinary row", "a tiny row", "a zeroed row")
            ),
            start=1,
        )
    ]


def bits(value):
    """The f32 a descriptor prints, in hex."""
    return f"0x{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPERATORS, default=default_op())
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--width", type=int, default=14336)
    parser.add_argument("--block", type=int, default=16384)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--input-stride", type=int, default=28672)
    parser.add_argument("--gate-stride", type=int, default=None)
    parser.add_argument("--value-stride", type=int, default=None)
    parser.add_argument("--activation", choices=ACTIVATIONS, default="silu")
    parser.add_argument(
        "--stepwise", choices=["true", "false"], default="false"
    )
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter; an E4M3 store cannot be checked there",
    )
    arguments = parser.parse_args()
    if arguments.block < arguments.width:
        parser.error(
            f"a block of {arguments.block} lanes cannot cover a row of {arguments.width}"
        )
    if not arguments.interpreter and not torch.cuda.is_available():
        print(
            "no GPU on this host: running through Triton's interpreter instead",
            file=sys.stderr,
        )
        arguments.interpreter = True
    if arguments.interpreter:
        print(
            "an E4M3 output needs a device: Triton's interpreter stores E4M3 wrong "
            "(-126.4941 -> -64, -31.6235 -> -16, 100.0 -> 104 where the format says -128, "
            "-32 and 96), so no reference can model it and no comparison here would be "
            "about the kernel",
            file=sys.stderr,
        )
        return 0
    import os

    os.environ.pop("TRITON_INTERPRET", None)

    if arguments.op == "value_first_swiglu_fp8":
        key = (
            f"dtype_{arguments.dtype}_output_f8e4m3_width_{arguments.width}"
            f"_block_{arguments.block}_input_stride_{arguments.input_stride}"
        )
    else:
        key = (
            f"dtype_{arguments.dtype}_output_f8e4m3_width_{arguments.width}"
            f"_block_{arguments.block}_activation_{arguments.activation}"
            f"_gate_stride_{arguments.gate_stride or 2 * arguments.width}"
            f"_value_stride_{arguments.value_stride or arguments.width}"
            f"_stepwise_{arguments.stepwise}"
        )
    print(f"descriptor: {key} ({arguments.op})")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments.op, arguments):
        failures += check(arguments.op, case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
