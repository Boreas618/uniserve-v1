#!/usr/bin/env python3
"""Run `snake_beta`'s kernel and check BigVGAN's periodic activation, on the CPU.

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python snake_beta_check.py [--package DIR]

```text
out = x + sin²(exp(α) · x) / (exp(β) + 1e-9)
```

with `α` and `β` **in log space**, one pair per channel, so a channel's frequency and its gain are
learned independently — which is what separates this operator from `snake`, whose one linear
parameter serves as both. The epsilon keeps the reciprocal finite as `β → -∞` and is part of the
formula. The operator's bound is `rounding(1, f32, dtype)` with `8 · 8 · u` of absolute room.

The reference is the expression in FP64 over the narrow input, compared at that bound. The cases:

- **`α = β = 0`**, the neutral channel: frequency one and gain one, where a kernel that read either
  parameter from the wrong pointer still agrees if the two happen to be equal — so this is the case
  that says the *others* are meaningful;
- **`α ≠ β`**: the case the swapped-reference mutation must fail, because it is the only place the
  two roles differ observably;
- **`α` large** (`α = 3`, frequency `e³ ≈ 20`), where the angle is large enough that a kernel taking
  `exp` of the wrong thing lands far away;
- **`β = -100`**, where `exp(β) ≈ 0` and only the epsilon keeps the reciprocal finite — the case a
  kernel that drops the epsilon turns into an infinity;
- **`β = +20`**, where the gain is tiny and the addend is a rounding-scale correction on `x`;
- **dtypes `bf16`, `f16`, `f32`**, and a block that ends inside a row.
"""

import argparse
import importlib.util
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

#: `snake_beta.rs`: one FP32 rounding of the output plus the addend's envelope.
ROUNDINGS = 1
VALUE_ENVELOPE = 8.0


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(element):
    """`Tolerance::rounding(1, f32, dtype).with_atol(8 · envelope · u)`."""
    unit = {
        torch.bfloat16: 2.0**-8,
        torch.float16: 2.0**-11,
        torch.float32: 2.0**-24,
    }[element]
    relative = 2.0 * gamma(ROUNDINGS, 2.0**-24) + unit
    return relative, 8.0 * VALUE_ENVELOPE * unit


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "snake_beta.py"
    spec = importlib.util.spec_from_file_location("snake_beta_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values, alpha, beta):
    """The operator's expression the way it evaluates it: **one FP32 step at a time**.

    `exp(α)`, the product with `x`, the square, the reciprocal and the sum are each FP32 in the
    kernel, so the reference rounds each of them and takes only the sine exactly, in FP64. A
    reference that kept the angle in FP64 differs from the kernel by the FP32 rounding of the
    frequency amplified by `x`, which is how a correct kernel read red at `α = 3` (a frequency of
    about 20) on the first run of this check.
    """
    x = values.to(torch.float32)
    frequency = torch.exp(alpha.to(torch.float32)).reshape(1, -1, 1)
    gain = torch.exp(beta.to(torch.float32)).reshape(1, -1, 1)
    angle = frequency * x
    sine = torch.sin(angle.to(torch.float64)).to(torch.float32)
    square = sine * sine
    addend = square / (gain + torch.full_like(gain, 1e-9))
    return (x + addend).to(torch.float64)


def run(
    module, *, dtype, rows, channels, width, alpha, beta, values, block, seed=43
):
    """One launch and the operator's own expression as the reference."""
    element = DTYPES[dtype]
    if values is None:
        torch.manual_seed(seed)
        values = torch.randn(rows, channels, width, dtype=torch.float32) * 2.0
    values = values.to(element)
    alpha = torch.as_tensor(alpha, dtype=torch.float32).reshape(channels)
    beta = torch.as_tensor(beta, dtype=torch.float32).reshape(channels)
    out = torch.full_like(values, float("nan"))

    module._snake_beta_kernel[((values.numel() + block - 1) // block, 1, 1)](
        values,
        alpha,
        beta,
        out,
        rows,
        channels,
        width,
        0,
        BLOCK=block,
        PDL=False,
    )

    expected = reference(values, alpha, beta)
    got = out.to(torch.float64)
    relative, absolute = bound(element)
    allowed = absolute + relative * expected.abs()
    error = (got - expected).abs()
    violations = int((~(error <= allowed)).sum())
    if violations:
        position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{dtype} {rows}x{channels}x{width} block {block} alpha {alpha.tolist()[:3]}… "
            f"beta {beta.tolist()[:3]}…: {violations} of {out.numel()} elements outside the bound; "
            f"first at {position}: {float(out.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}, rtol {relative:.3e} atol {absolute:.3e}"
        )
    return int(out.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/snake_beta.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit(
            "run this with TRITON_INTERPRET=1: the kernel is checked on the CPU"
        )
    module = load_kernel(options.package)

    checked = 0
    # The neutral channel, then the same shape with the two roles pulled apart.
    checked += run(
        module,
        dtype="f32",
        rows=2,
        channels=4,
        width=32,
        alpha=[0.0] * 4,
        beta=[0.0] * 4,
        values=None,
        block=256,
    )
    checked += run(
        module,
        dtype="f32",
        rows=2,
        channels=4,
        width=32,
        alpha=[1.0, 0.0, 2.0, 0.5],
        beta=[1.0, 2.0, 0.0, -1.0],
        values=None,
        block=256,
    )
    # A large frequency: the angle is far from zero on most elements.
    checked += run(
        module,
        dtype="f32",
        rows=1,
        channels=3,
        width=48,
        alpha=[3.0, 3.0, 3.0],
        beta=[0.0, 1.0, -1.0],
        values=None,
        block=256,
    )
    # The gain's two ends: `exp(beta)` at zero and at huge.
    checked += run(
        module,
        dtype="f32",
        rows=1,
        channels=2,
        width=24,
        alpha=[0.0, 0.0],
        beta=[-100.0, 20.0],
        values=torch.full((1, 2, 24), 0.5),
        block=256,
    )
    # The fixture's narrow types, and a block that ends inside a row.
    checked += run(
        module,
        dtype="bf16",
        rows=2,
        channels=8,
        width=40,
        alpha=[0.5] * 8,
        beta=[-0.5] * 8,
        values=torch.randn(2, 8, 40) * 10.0,
        block=256,
    )
    checked += run(
        module,
        dtype="f16",
        rows=1,
        channels=128,
        width=300,
        alpha=[0.25] * 128,
        beta=[0.75] * 128,
        values=None,
        block=256,
    )
    print(
        f"ok: {checked} activated values over six cases, within rounding(1)'s bound"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
