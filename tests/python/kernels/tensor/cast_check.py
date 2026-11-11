#!/usr/bin/env python3
"""Run `cast`'s kernel and check every pair of element types against numpy, on the CPU.

    python cast_check.py             # Triton's interpreter, no device
    python cast_check.py --device    # compiled for a device

Each case launches the module of its descriptor key as `Cast::launch` does — the input and the
output merged into the axes a program traverses, the extents and the input's strides packed
right-aligned — and compares the output with the conversion the operator states:

- **into an integer from an integer** the low bits are kept, which numpy's `astype` computes; the
  inputs reach past the target's range on both sides, so a conversion that saturated would differ;
- **into an integer from a float** the value truncates toward zero; the inputs carry fractions of
  both signs (non-negative for an unsigned target, whose range a negative value leaves);
- **into a float** the value rounds to nearest even: the output is held to the exact value inside
  `Cast::tolerance`, which is equality wherever the target holds every value of the source.

All 72 ordered pairs run over a contiguous input and a permuted one, and four pairs run over every
layout of `operands.views`: rank 1 to 6, windows, steps, permutations, zero-stride broadcasts, an
input of several programs and a single element.
"""

import functools
import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import numpy as np
import operands
import torch

DTYPES = operands.FLOATS + operands.INTEGERS

#: The bits a value of each type is exact to (`precision` in `crates/ops/src/cast.rs`).
PRECISION = {
    "u8": 8,
    "bf16": 8,
    "f16": 11,
    "i16": 15,
    "f32": 24,
    "i32": 31,
    "u32": 32,
    "f64": 53,
    "i64": 63,
}

#: The range integer inputs are drawn from: past every narrower type's range, and exact in float64.
RANGE = {
    "u8": (0, 256),
    "u32": (0, 1 << 32),
    "i16": (-(1 << 15), 1 << 15),
    "i32": (-(1 << 31), 1 << 31),
    "i64": (-(1 << 40), 1 << 40),
}


def holds(source, target):
    """Whether `target` holds every value of `source` (`holds` in `crates/ops/src/cast.rs`)."""
    if target in operands.INTEGERS:
        return True
    wide = target in ("f32", "f64")
    return PRECISION[source] <= PRECISION[target] and (
        wide or source in operands.INTEGERS
    )


def inputs(source, shape, seed, target):
    """A contiguous input of `source` elements whose conversion to `target` is defined."""
    if source in operands.INTEGERS:
        generator = torch.Generator().manual_seed(seed)
        low, high = RANGE[source]
        return operands.tensor(
            torch.randint(low, high, shape, generator=generator), source
        )

    drawn = operands.values(source, shape, seed, scale=30.0)
    if target in ("u8", "u32"):
        # A negative value has no unsigned counterpart: the conversion leaves it unspecified.
        drawn = drawn.abs()
    return drawn


def expected_of(x, target):
    """The conversion of `x` as the operator states it, exact in float64 or int64."""
    exact = operands.real(x)
    if target in operands.FLOATS:
        return exact
    numpy_target = np.dtype(
        str(operands.ELEMENTS[target]).removeprefix("torch.")
    )
    if x.dtype.is_floating_point:
        return np.trunc(exact).astype(np.int64)
    return exact.astype(numpy_target).astype(np.int64)


def check(source, target, name, x):
    """One launch held to the stated conversion."""
    out = operands.empty(x.shape, operands.ELEMENTS[target])
    extents, (x_strides, _) = operands.merged([x, out])
    elements = out.numel()

    # `Descriptor`'s `Display`.
    key = f"input_{source}_output_{target}_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    operands.launch(
        "cast",
        key,
        operands.grid(elements),
        x,
        out,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(x_strides),
        0,
    )

    # `Cast::tolerance`.
    bound = (
        operands.EXACT
        if holds(source, target)
        else operands.rounding(1, target)
    )
    label = f"{source} to {target} {name} {tuple(x.shape)}"
    return operands.compare(label, out, expected_of(x, target), bound)


def main():
    if not operands.ready():
        return 0

    checked = 0
    pairs = [
        (source, target)
        for source in DTYPES
        for target in DTYPES
        if source != target
    ]

    # Every pair: a contiguous input and a permuted one.
    for seed, (source, target) in enumerate(pairs):
        contiguous = inputs(source, (5, 7), seed, target)
        permuted = inputs(source, (3, 4, 5), seed + 500, target).permute(
            2, 0, 1
        )
        checked += check(source, target, "contiguous", contiguous)
        checked += check(source, target, "permuted", permuted)

    # Decided values: a fraction of either sign truncates toward zero, and a half does not round up.
    fractions = [
        0.0,
        0.5,
        -0.5,
        0.99,
        -0.99,
        1.5,
        -1.5,
        2.5,
        -2.5,
        127.75,
        -127.75,
    ]
    for source in operands.FLOATS:
        x = operands.tensor(
            torch.tensor(fractions, dtype=torch.float64), source
        )
        for target in ("i16", "i32", "i64"):
            checked += check(source, target, "fractions", x)

    # Every layout, over a narrowing, a rounding of an integer, a truncation and a widening.
    layouts = 0
    for source, target in (
        ("f32", "bf16"),
        ("i64", "f32"),
        ("bf16", "i32"),
        ("u8", "f64"),
    ):
        make = functools.partial(inputs, target=target)
        for name, view in operands.views(source, 41, make=make):
            checked += check(source, target, name, view)
            layouts += 1

    print(
        f"ok: {checked} elements over {len(pairs)} ordered pairs of {len(DTYPES)} element types and "
        f"{layouts} strided launches of rank 1 to 6 on {operands.where()}, exact wherever the "
        f"target holds the source and within rounding(1) elsewhere"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
