#!/usr/bin/env python3
"""Run `snake`'s kernel and check the decoder's periodic activation, on the CPU.

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python snake_check.py [--package DIR]

The operator is one line of the H3 audio VAE (`uniserve/models/minimax_h3/audio_vae.py`,
`Snake.forward`):

```text
out = x + sin²(α · x) / (α + 1e-9)
```

with `α` linear, one per channel, and the epsilon **part of the formula rather than a guard**. Three
roundings in FP32 (the sine, its square, the reciprocal) and the output's own are what the operator's
bound covers: `rounding(3, f32, dtype)` with `8 · 8 · u` of absolute room for the addend.

The reference is the same expression in FP64 over the narrow input, compared at that bound. The
cases are chosen where a plausible-but-wrong kernel parts from this one, and each is named for the
mistake it catches:

- **`α · x = π/2` exactly**, the sine's maximum: a kernel that used `cos`, or squared before the
  sine, or dropped the square, is furthest from the right answer here;
- **`α = 0`**, where the epsilon is the only thing between the result and a division by zero — the
  case that catches a kernel that treats the epsilon as an optimisation to drop;
- **`α` negative**, where the addend's sign follows the reciprocal and `sin²` cannot be negative:
  a kernel that squared the whole addend, or used `|α|`, agrees everywhere above zero and fails here;
- **a large `α · x`** where `sin²` has saturated against the bound;
- **a row count and width that do not fill the last block**, and **many channels against few**, so the
  per-channel parameter's index (`(offset / width) % channels`) is exercised at each block boundary.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import sys

# This is an interpreter check: the arithmetic is compared against a CPU reference, so the kernel runs
# on Triton's interpreter even where a device exists. The flag is set before any kernel module is
# imported, because Triton reads it when it launches.
if os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    print("running on Triton's interpreter (no device needed)")

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: `snake.rs`: the three FP32 roundings and the addend's envelope.
ROUNDINGS = 3
VALUE_ENVELOPE = 8.0


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(element):
    """`Tolerance::rounding(3, f32, dtype).with_atol(8 · envelope · u)`."""
    unit = {
        torch.bfloat16: 2.0**-8,
        torch.float16: 2.0**-11,
        torch.float32: 2.0**-24,
    }[element]
    relative = 2.0 * gamma(ROUNDINGS, 2.0**-24) + unit
    return relative, 8.0 * VALUE_ENVELOPE * unit


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "snake.py"
    spec = importlib.util.spec_from_file_location("snake_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values, alpha):
    """The operator's expression the way it evaluates it: **one FP32 step at a time**.

    Every step of the formula is FP32 in the kernel, so the reference rounds each result to FP32 and
    takes only the transcendental exactly, in FP64, before rounding it in again. A reference that
    kept the angle in FP64 instead would differ from the kernel by the FP32 rounding of `α · x`
    amplified by `x` — which is how a correct kernel read red on the first run of the `alpha = 3`
    case, at an angle of about 120 radians.
    """
    x = values.to(torch.float32)
    a = alpha.to(torch.float32).reshape(1, -1, 1)
    angle = a * x
    sine = torch.sin(angle.to(torch.float64)).to(torch.float32)
    square = sine * sine
    reciprocal = torch.full_like(a, 1e-9)
    addend = square / (a + reciprocal)
    return (x + addend).to(torch.float64)


def run(module, *, dtype, rows, channels, width, alpha, values, block, seed=41):
    """One launch and the operator's own expression as the reference."""
    element = DTYPES[dtype]
    if values is None:
        torch.manual_seed(seed)
        values = torch.randn(rows, channels, width, dtype=torch.float32) * 2.0
    values = values.to(element)
    alpha = torch.as_tensor(alpha, dtype=torch.float32).reshape(channels)
    out = torch.full_like(values, float("nan"))

    module._snake_kernel[((values.numel() + block - 1) // block, 1, 1)](
        values, alpha, out, rows, channels, width, 0, BLOCK=block, PDL=False
    )

    expected = reference(values, alpha)
    got = out.to(torch.float64)
    relative, absolute = bound(element)
    allowed = absolute + relative * expected.abs()
    error = (got - expected).abs()
    violations = int((~(error <= allowed)).sum())
    if violations:
        position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{dtype} {rows}x{channels}x{width} block {block} alpha {alpha.tolist()[:3]}…: "
            f"{violations} of {out.numel()} elements outside the bound; first at {position}: "
            f"{float(out.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}, "
            f"rtol {relative:.3e} atol {absolute:.3e}"
        )
    return int(out.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/snake.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit(
            "run this with TRITON_INTERPRET=1: the kernel is checked on the CPU"
        )
    module = load_kernel(options.package)

    checked = 0
    # The sine's maximum: alpha · x = pi/2 for every element.
    alpha = torch.full((4,), math.pi / 2.0)
    values = torch.ones(2, 4, 32, dtype=torch.float32)
    checked += run(
        module,
        dtype="f32",
        rows=2,
        channels=4,
        width=32,
        alpha=alpha,
        values=values,
        block=256,
    )
    # A zero frequency: only the epsilon keeps the reciprocal finite.
    checked += run(
        module,
        dtype="f32",
        rows=1,
        channels=2,
        width=16,
        alpha=[0.0, 0.0],
        values=torch.full((1, 2, 16), 0.75),
        block=256,
    )
    # A negative frequency: the addend is negative and sin² is not.
    checked += run(
        module,
        dtype="f32",
        rows=2,
        channels=3,
        width=24,
        alpha=[-0.5, -2.0, 1.25],
        values=None,
        block=256,
    )
    # A large angle, where sin² has saturated, in the fixture's narrow types.
    checked += run(
        module,
        dtype="bf16",
        rows=2,
        channels=8,
        width=40,
        alpha=[0.9] * 8,
        values=torch.randn(2, 8, 40) * 12.0,
        block=256,
    )
    checked += run(
        module,
        dtype="f16",
        rows=3,
        channels=5,
        width=7,
        alpha=[0.1, 0.5, 1.0, 2.0, 7.0],
        values=None,
        block=256,
    )
    # Many channels against few, and a block that ends inside a 300-wide row, so one program spans a
    # row boundary and the channel index changes inside it.
    checked += run(
        module,
        dtype="bf16",
        rows=1,
        channels=128,
        width=300,
        alpha=[0.25] * 128,
        values=None,
        block=256,
    )
    print(
        f"ok: {checked} activated values over six cases, within rounding(3)'s bound"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
