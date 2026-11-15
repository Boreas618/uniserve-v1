"""The cast program against numpy, for every pair of element types.

Each case launches the module of its descriptor key as ``Cast::launch`` does
(the input and the output merged into the axes a program traverses, the
extents and the input's strides packed right-aligned) and compares the output
with the conversion the operator states:

- **into an integer from an integer** the low bits are kept, which numpy's
  ``astype`` computes; the inputs reach past the target's range on both sides,
  so a conversion that saturated would differ;
- **into an integer from a float** the value truncates toward zero; the inputs
  carry fractions of both signs (non-negative for an unsigned target, whose
  range a negative value leaves);
- **into a float** the value rounds to nearest even: the output is held to the
  exact value inside ``Cast::tolerance``, which is equality wherever the target
  holds every value of the source.

All 72 ordered pairs run over a contiguous input and a permuted one, decided
fractions run into every signed integer type, and four pairs run over every
layout of :data:`~uniserve_testing.kernels.tensor.operands.LAYOUTS`.
"""

import functools

import numpy as np
import pytest
import torch
from uniserve_testing.kernels.tensor import operands

pytestmark = pytest.mark.unit

DTYPES = operands.FLOATS + operands.INTEGERS

#: The bits a value of each type is exact to (`precision` in
#: `crates/ops/src/cast.rs`).
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

#: The range integer inputs are drawn from: past every narrower type's range,
#: and exact in float64.
RANGE = {
    "u8": (0, 256),
    "u32": (0, 1 << 32),
    "i16": (-(1 << 15), 1 << 15),
    "i32": (-(1 << 31), 1 << 31),
    "i64": (-(1 << 40), 1 << 40),
}

#: Every ordered pair of distinct element types, with its seed.
PAIRS = [
    pytest.param(source, target, seed, id=f"{source}-to-{target}")
    for seed, (source, target) in enumerate(
        (source, target)
        for source in DTYPES
        for target in DTYPES
        if source != target
    )
]

#: Decided values: a fraction of either sign truncates toward zero, and a half
#: does not round up.
FRACTIONS = [0.0, 0.5, -0.5, 0.99, -0.99, 1.5, -1.5, 2.5, -2.5, 127.75, -127.75]

#: The pairs every layout is cast over: a narrowing, a rounding of an integer,
#: a truncation and a widening.
LAYOUT_PAIRS = (("f32", "bf16"), ("i64", "f32"), ("bf16", "i32"), ("u8", "f64"))


def holds(source, target):
    """Whether `target` holds every value of `source` (`holds` in
    `crates/ops/src/cast.rs`).
    """  # noqa: D205
    if target in operands.INTEGERS:
        return True
    wide = target in ("f32", "f64")
    return PRECISION[source] <= PRECISION[target] and (
        wide or source in operands.INTEGERS
    )


def inputs(primitives, source, shape, seed, target):
    """A contiguous input of `source` elements whose conversion to `target` is
    defined.
    """  # noqa: D205
    if source in operands.INTEGERS:
        generator = torch.Generator().manual_seed(seed)
        low, high = RANGE[source]
        drawn = torch.randint(low, high, shape, generator=generator)
        return primitives.tensor(drawn, source)

    drawn = primitives.values(source, shape, seed, scale=30.0)
    if target in ("u8", "u32"):
        # A negative value has no unsigned counterpart: the conversion leaves
        # it unspecified.
        drawn = drawn.abs()
    return drawn


def expected_of(x, target):
    """The conversion of `x` as the operator states it, exact in float64 or
    int64.
    """  # noqa: D205
    exact = operands.real(x)
    if target in operands.FLOATS:
        return exact
    numpy_target = np.dtype(
        str(operands.ELEMENTS[target]).removeprefix("torch.")
    )
    if x.dtype.is_floating_point:
        return np.trunc(exact).astype(np.int64)
    return exact.astype(numpy_target).astype(np.int64)


def check(primitives, source, target, name, x):
    """One launch held to the stated conversion."""
    out = primitives.empty(x.shape, operands.ELEMENTS[target])
    extents, (x_strides, _) = operands.merged([x, out])
    elements = out.numel()
    # `Descriptor`'s `Display`.
    key = (
        f"input_{source}_output_{target}_rank_{len(extents)}"
        f"_block_{operands.BLOCK}_pdl_false"
    )
    primitives.launch(
        "cast",
        key,
        operands.grid(elements),
        x,
        out,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(x_strides),
    )

    # `Cast::tolerance`.
    bound = (
        operands.EXACT
        if holds(source, target)
        else operands.rounding(1, target)
    )
    label = f"{source} to {target} {name} {tuple(x.shape)}"
    primitives.assert_close(label, out, expected_of(x, target), bound)


@pytest.mark.parametrize(("source", "target", "seed"), PAIRS)
def test_a_pair_is_converted_as_the_operator_states(
    source, target, seed, primitives
):
    contiguous = inputs(primitives, source, (5, 7), seed, target)
    permuted = inputs(primitives, source, (3, 4, 5), seed + 500, target)
    check(primitives, source, target, "contiguous", contiguous)
    check(primitives, source, target, "permuted", permuted.permute(2, 0, 1))


@pytest.mark.parametrize("target", ["i16", "i32", "i64"])
@pytest.mark.parametrize("source", operands.FLOATS)
def test_a_fraction_truncates_toward_zero(source, target, primitives):
    x = primitives.tensor(torch.tensor(FRACTIONS, dtype=torch.float64), source)
    check(primitives, source, target, "fractions", x)


@pytest.mark.parametrize("layout", operands.LAYOUTS)
@pytest.mark.parametrize(("source", "target"), LAYOUT_PAIRS)
def test_every_layout_is_converted(source, target, layout, primitives):
    make = functools.partial(inputs, primitives, target=target)
    view = primitives.view(layout, source, 41, make=make)
    check(primitives, source, target, layout, view)
