"""The snake program against its periodic activation.

The operator is the Snake activation, with one linear parameter per channel:

```text
out = x + sin^2(alpha x) / (alpha + 1e-9)
```

with the epsilon **part of the formula rather than a guard**. Three roundings
in FP32 (the sine, its square, the reciprocal) and the output's own are what
the operator's bound covers: ``rounding(3, f32, dtype)`` with ``8 * 8 * u`` of
absolute room for the addend.

The reference is the same expression over the narrow input, one FP32 step at
a time. The cases are chosen where a plausible-but-wrong kernel parts from this
one, and each is named for the mistake it catches:

- **``alpha x = pi/2`` exactly**, the sine's maximum: a kernel that used
  ``cos``, or squared before the sine, or dropped the square, is furthest from
  the right answer here;
- **``alpha = 0``**, where the epsilon is the only thing between the result and
  a division by zero: the case that catches a kernel that treats the epsilon
  as an optimisation to drop;
- **``alpha`` negative**, where the addend's sign follows the reciprocal and
  ``sin^2`` cannot be negative: a kernel that squared the whole addend, or
  used ``|alpha|``, agrees everywhere above zero and fails here;
- **a large ``alpha x``** where ``sin^2`` has saturated against the bound;
- **a row count and width that do not fill the last block**, and **many
  channels against few**, so the per-channel parameter's index
  (``(offset / width) % channels``) is exercised at each block boundary.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: `snake.rs`: the three FP32 roundings and the addend's envelope.
ROUNDINGS = 3
VALUE_ENVELOPE = 8.0

#: The block one program covers.
BLOCK = 256

#: The cases: the element type, the rows, channels and width, each channel's
#: `alpha`, and the inputs: a constant `fill`, or normal values times `scale`
#: drawn from `seed`.
CASES = {
    "the sine's maximum, alpha x = pi/2": {
        "dtype": "f32",
        "shape": (2, 4, 32),
        "alpha": [math.pi / 2.0] * 4,
        "fill": 1.0,
    },
    "a zero frequency: only the epsilon keeps the reciprocal finite": {
        "dtype": "f32",
        "shape": (1, 2, 16),
        "alpha": [0.0, 0.0],
        "fill": 0.75,
    },
    "a negative frequency: the addend is negative and sin^2 is not": {
        "dtype": "f32",
        "shape": (2, 3, 24),
        "alpha": [-0.5, -2.0, 1.25],
        "scale": 2.0,
        "seed": 41,
    },
    "a large angle, where sin^2 has saturated, bf16": {
        "dtype": "bf16",
        "shape": (2, 8, 40),
        "alpha": [0.9] * 8,
        "scale": 12.0,
        "seed": 42,
    },
    "a block ending inside the last row, f16": {
        "dtype": "f16",
        "shape": (3, 5, 7),
        "alpha": [0.1, 0.5, 1.0, 2.0, 7.0],
        "scale": 2.0,
        "seed": 41,
    },
    # One program spans a row boundary, and the channel index changes inside
    # it.
    "many channels against few, a block ending inside a 300-wide row": {
        "dtype": "bf16",
        "shape": (1, 128, 300),
        "alpha": [0.25] * 128,
        "scale": 2.0,
        "seed": 41,
    },
}


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(element):
    """`Tolerance::rounding(3, f32, dtype).with_atol(8 * envelope * u)`."""
    unit = {
        torch.bfloat16: 2.0**-8,
        torch.float16: 2.0**-11,
        torch.float32: 2.0**-24,
    }[element]
    relative = 2.0 * gamma(ROUNDINGS, 2.0**-24) + unit
    return relative, 8.0 * VALUE_ENVELOPE * unit


def reference(values, alpha):
    """The operator's expression the way it evaluates it: **one FP32 step at a
    time**.

    Every step of the formula is FP32 in the kernel, so the reference rounds
    each result to FP32 and takes only the transcendental exactly, in FP64,
    before rounding it in again. A reference that kept the angle in FP64
    instead would differ from the kernel by the FP32 rounding of `alpha x`
    amplified by `x`, which at an angle of about 120 radians reads a correct
    kernel as wrong.
    """  # noqa: D205
    x = values.to(torch.float32)
    a = alpha.to(torch.float32).reshape(1, -1, 1)
    angle = a * x
    sine = torch.sin(angle.to(torch.float64)).to(torch.float32)
    square = sine * sine
    reciprocal = torch.full_like(a, 1e-9)
    addend = square / (a + reciprocal)
    return (x + addend).to(torch.float64)


def launch(kernels, values, alpha, out):
    """One launch over every element of `values`."""
    rows, channels, width = values.shape
    kernels._snake_kernel[((values.numel() + BLOCK - 1) // BLOCK, 1, 1)](
        values, alpha, out, rows, channels, width, BLOCK=BLOCK, PDL=False
    )


@pytest.mark.parametrize("case", CASES)
def test_every_element_matches_the_formula_within_the_operators_bound(
    case, torch_device, kernel_source
):
    spec = CASES[case]
    element = DTYPES[spec["dtype"]]
    shape = spec["shape"]
    if "fill" in spec:
        values = torch.full(shape, spec["fill"], dtype=torch.float32)
    else:
        generator = torch.Generator().manual_seed(spec["seed"])
        values = torch.randn(shape, generator=generator) * spec["scale"]
    values = values.to(element).to(torch_device)
    alpha = torch.tensor(spec["alpha"], dtype=torch.float32).to(torch_device)
    out = torch.full_like(values, float("nan"))

    launch(kernel_source("snake"), values, alpha, out)

    expected = reference(values, alpha)
    got = out.to(torch.float64)
    relative, absolute = bound(element)
    error = (got - expected).abs()
    outside = (~(error <= absolute + relative * expected.abs())).reshape(-1)
    if outside.any():
        position = int(outside.nonzero()[0])
        pytest.fail(
            f"{spec['dtype']} {shape} block {BLOCK} alpha {spec['alpha'][:3]}: "
            f"{int(outside.sum())} of {out.numel()} elements outside the "
            f"bound; first at {position}: {float(out.reshape(-1)[position])} "
            f"vs {float(expected.reshape(-1)[position])}, rtol "
            f"{relative:.3e} atol {absolute:.3e}"
        )
