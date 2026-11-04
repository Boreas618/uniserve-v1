#!/usr/bin/env python3
"""Run the act_and_mul program and check both halves against torch's own arithmetic.

    python act_and_mul_check.py                          # the fixture's 25600 cell, silu
    python act_and_mul_check.py --interpreter            # the same on the CPU
    python act_and_mul_check.py --value-stride 26112 --rows 3

The operator is `act(gate) · value` over a row of `width` elements, with **each half's
rows a stride apart**: `gate_stride` and `value_stride` are separate, because the two
halves are views of a fused projection and nothing requires them to be adjacent. The
program therefore reads both strides; a check that packed the halves contiguously would
pass whether or not the program honoured them, so every case here is built with the two
strides *different* as well as equal, and one case interleaves nothing at all.

The reference is torch in FP64: `silu` as `g · sigmoid(g)`, the tanh GELU in the form the
kernel uses (the sigmoid of the cubic, `g / (1 + exp(-sqrt(8/π)(g + 0.044715 g³)))`, which
saturates to exact 0 and `g`), and the exact GELU as `0.5g(1 + erf(g/√2))`. The bound is
the operator's: no reduction, so `roundings · u_out` with three roundings for `silu` and
four for either GELU, and a relative comparison with no tolerance of our own choosing.

`gelu` is not run through Triton's interpreter: its exact form needs `libdevice.erf`,
which the interpreter does not provide. The device run covers it.
"""

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The operator crate's `DType::as_i64()` numbering, by element type name.
DTYPES = {"f16": 3, "bf16": 4, "f32": 5}

#: The relative bound's factor per activation, from `activation.rs::tolerance`.
ROUNDINGS = {"silu": 3.0, "gelu": 4.0, "gelu_tanh": 4.0}

#: `2**-8` for f16, `2**-8` for bf16 and `2**-24` for f32: the output type's unit roundoff,
#: as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


def activate(gate, name):
    """The reference activation, evaluated in FP64."""
    if name == "silu":
        return gate * torch.sigmoid(gate)
    if name == "gelu_tanh":
        inner = math.sqrt(8.0 / math.pi) * (gate + 0.044715 * gate**3)
        return gate * torch.sigmoid(inner)
    return 0.5 * gate * (1.0 + torch.erf(gate * math.sqrt(0.5)))


def reference(gate, value, name):
    """The exact product the operator declares."""
    return activate(gate.double(), name) * value.double()


def strided(flat, rows, width, stride):
    """The `[rows, width]` view of `flat` whose rows are `stride` elements apart."""
    return flat.as_strided((rows, width), (stride, 1))


def check(case, device):
    """Build one case's two halves at their own strides, launch, and compare."""
    import act_and_mul as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, width = case["rows"], case["width"]
    # Gates over a wide range, so a saturating activation is exercised too, and values
    # of both signs so a product cannot pass by being zero.
    gate_flat = (
        torch.randn(rows * case["gate_stride"], generator=generator) * 4.0
    ).to(dtype)
    value_flat = torch.randn(
        rows * case["value_stride"], generator=generator
    ).to(dtype)
    gate = strided(gate_flat, rows, width, case["gate_stride"])
    value = strided(value_flat, rows, width, case["value_stride"])
    out = torch.full((rows, width), float("nan"), dtype=dtype)
    if case["zero_row"]:
        gate[0, :] = 0.0

    elements = rows * width
    block = case["block"]
    kernels._act_and_mul_kernel[(math.ceil(elements / block),)](
        gate,
        value,
        out,
        elements,
        case["gate_stride"],
        case["value_stride"],
        n_cols=width,
        ACTIVATION=case["activation"],
        BLOCK=block,
        WIDE=case["wide"],
        PDL=case["pdl"],
        stepwise=case["stepwise"],
        num_warps=8,
    )

    expected = reference(
        gate.cpu().to(torch.float32),
        value.cpu().to(torch.float32),
        case["activation"],
    )
    got = out.cpu().double()
    bound = ROUNDINGS[case["activation"]] * UNIT_ROUNDOFF[case["dtype"]]
    difference = (got - expected).abs()
    allowed = bound * expected.abs()
    # A difference that is not a number is never inside a bound: `>` says otherwise.
    outside = (~(difference <= allowed)).sum().item()
    largest = (
        (difference / expected.abs().clamp(min=1e-30)).max().item()
        if expected.numel()
        else 0.0
    )
    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {rows} x {width}, {case['dtype']}, "
        f"{case['activation']}, gate_stride {case['gate_stride']}, "
        f"value_stride {case['value_stride']} — max relative error "
        f"{largest:.3e}, bound {bound:.3e}, {outside} of {expected.numel()} outside"
    )
    return outside


def cases(flags, interpreter):
    """The cases one run checks, cheapest and most diverse first."""
    found = [
        {
            "label": "the fixture's width at the stride a fused projection leaves",
            "dtype": flags.dtype,
            "width": flags.width,
            "rows": flags.rows,
            "block": flags.block,
            "activation": flags.activation,
            "gate_stride": flags.gate_stride or flags.width,
            "value_stride": flags.value_stride or flags.width,
            "wide": flags.wide == "true",
            "pdl": flags.pdl == "true" and not interpreter,
            "stepwise": flags.stepwise == "true",
            "zero_row": False,
            "seed": 1,
        },
        {
            "label": "the two halves at different strides: nothing is adjacent",
            "dtype": flags.dtype,
            "width": flags.width,
            "rows": flags.rows,
            "block": flags.block,
            "activation": flags.activation,
            "gate_stride": flags.width + 512,
            "value_stride": flags.width + 1024,
            "wide": flags.wide == "true",
            "pdl": flags.pdl == "true" and not interpreter,
            "stepwise": flags.stepwise == "true",
            "zero_row": True,
            "seed": 2,
        },
        {
            "label": "a row narrower than one block: the tail is masked",
            "dtype": flags.dtype,
            "width": 100,
            "rows": 3,
            "block": flags.block,
            "activation": flags.activation,
            "gate_stride": 100,
            "value_stride": 100,
            "wide": flags.wide == "true",
            "pdl": False,
            "stepwise": flags.stepwise == "true",
            "zero_row": False,
            "seed": 3,
        },
    ]
    if not interpreter:
        found.append(
            {
                "label": "the exact GELU: its error function is the device's",
                "dtype": flags.dtype,
                "width": flags.width,
                "rows": flags.rows,
                "block": flags.block,
                "activation": "gelu",
                "gate_stride": flags.width,
                "value_stride": flags.width,
                "wide": flags.wide == "true",
                "pdl": False,
                "stepwise": False,
                "zero_row": False,
                "seed": 4,
            }
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--width", type=int, default=25600)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument(
        "--activation", choices=sorted(ROUNDINGS), default="silu"
    )
    parser.add_argument("--gate-stride", type=int, default=0)
    parser.add_argument("--value-stride", type=int, default=0)
    parser.add_argument("--wide", choices=["true", "false"], default="false")
    parser.add_argument("--pdl", choices=["true", "false"], default="false")
    parser.add_argument(
        "--stepwise", choices=["true", "false"], default="false"
    )
    parser.add_argument("--expect-key", default=None)
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
        f"dtype_{arguments.dtype}_width_{arguments.width}"
        f"_block_{arguments.block}_activation_{arguments.activation}"
        f"_gate_stride_{arguments.gate_stride or arguments.width}"
        f"_value_stride_{arguments.value_stride or arguments.width}"
        f"_wide_{arguments.wide}_pdl_{arguments.pdl}"
        f"_stepwise_{arguments.stepwise}"
    )
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    device = "cpu" if arguments.interpreter else "cuda"
    outside = 0
    for case in cases(arguments, arguments.interpreter):
        outside += check(case, device)
    return 1 if outside else 0


if __name__ == "__main__":
    sys.exit(main())
