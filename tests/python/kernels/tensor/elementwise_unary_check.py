#!/usr/bin/env python3
"""Run `elementwise_unary`'s kernel and check every function against numpy, on the CPU.

    python elementwise_unary_check.py             # Triton's interpreter, no device
    python elementwise_unary_check.py --device    # compiled for a device

Each case launches the package's kernel as `ElementwiseUnary::launch` does — the input and the
output merged into the axes a program traverses, the extents and the strides packed right-aligned —
and compares the output with the function's formula evaluated in float64 on the same values, inside
the operator's own bound (`ElementwiseUnary::tolerance`).

What is covered:

- **every function** on every floating element type, over a contiguous input and a permuted one, and
  on values where a formula is decided: a half (`round`), a signed zero and both signs (`sign`,
  `relu`, `abs`), the two ends of a saturating activation (`gelu`, `gelu_erf`, `silu`, `sigmoid`),
  and the neighbourhood of zero where `elu`'s difference cancels;
- **the functions closed over the integers** on every integer element type, wrapping as the element
  type does;
- **every layout** of `operands.views`: rank 1 to 6, an offset window, a stepped axis, a
  permutation, a zero-stride broadcast, an input of several programs with a masked tail and a single
  element.

`tanh`, `powf` and every exponential (`exp`, and the one inside `gelu`, `silu`, `sigmoid` and `elu`)
call libdevice, which the interpreter cannot evaluate. Numpy's `tanh`, `power` and `exp` stand in
for those calls there, so the interpreter run holds which values reach them, in which order, and
where the results are stored; the calls' own last place is libdevice's, and `--device` is the run
that holds it to the bound. A BF16 output is rounded to nearest on a device and truncated by the
interpreter, which `operands.compare` allows for.
"""

import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import numpy as np
import operands
import torch
import triton.language as tl
from triton.runtime import interpreter

kernels = operands.load("elementwise_unary")

#: `VALUE_ENVELOPE` of `crates/ops/src/elementwise.rs`.
VALUE_ENVELOPE = 8.0

#: The scalars of the three functions that take them.
SCALARS = {"affine": (1.5, -0.25), "powf": (2.5, 0.0), "elu": (0.75, 0.0)}

#: The integer scalars of `affine`: a factor and an addend every integer type holds.
INTEGER_SCALARS = (3, 5)

#: Each function's rounded operations (`UnaryFunction::roundings`).
ROUNDINGS = {
    "exp": 1, "log": 1, "sin": 1, "cos": 1, "tanh": 1, "neg": 0, "recip": 1, "sqr": 1, "sqrt": 1,
    "abs": 0, "gelu": 8, "gelu_erf": 4, "erf": 1, "relu": 0, "silu": 3, "sigmoid": 3, "ceil": 0,
    "floor": 0, "round": 0, "sign": 0, "affine": 2, "powf": 1, "elu": 3,
}  # fmt: skip

#: The functions closed over the integers (`UnaryFunction::integral`).
INTEGRAL = (
    "neg",
    "sqr",
    "abs",
    "relu",
    "ceil",
    "floor",
    "round",
    "sign",
    "affine",
)


class Libdevice:
    """Numpy in the place of the libdevice calls the kernel makes."""

    @staticmethod
    def _applied(function, *arguments):
        data = function(*(argument.handle.data for argument in arguments))
        first = arguments[0]
        return tl.tensor(
            interpreter.TensorHandle(data, first.dtype.scalar), first.type
        )

    @staticmethod
    def exp(value):
        # An exponential past the type's largest value is an infinity, as on a device.
        with np.errstate(over="ignore"):
            return Libdevice._applied(np.exp, value)

    @staticmethod
    def tanh(value):
        return Libdevice._applied(np.tanh, value)

    @staticmethod
    def pow(base, exponent):
        return Libdevice._applied(np.power, base, exponent)


if not operands.DEVICE:
    kernels.libdevice = Libdevice


def away_from_zero(x):
    """The nearest integer, a half away from zero, decided on the exact fraction."""
    whole = np.trunc(x)
    return np.where(np.abs(x - whole) >= 0.5, whole + np.sign(x), whole)


def erf(x):
    """The error function of a float64 array."""
    return torch.erf(torch.from_numpy(np.ascontiguousarray(x))).numpy()


def erfc(x):
    """The complementary error function of a float64 array."""
    return torch.erfc(torch.from_numpy(np.ascontiguousarray(x))).numpy()


def formula(function, x, a, b):
    """`f(x)` in float64: the formulas of `crates/ops/src/elementwise.rs`."""
    with np.errstate(all="ignore"):
        return {
            "exp": lambda: np.exp(x),
            "log": lambda: np.log(x),
            "sin": lambda: np.sin(x),
            "cos": lambda: np.cos(x),
            "tanh": lambda: np.tanh(x),
            "neg": lambda: -x,
            "recip": lambda: 1.0 / x,
            "sqr": lambda: x * x,
            "sqrt": lambda: np.sqrt(x),
            "abs": lambda: np.abs(x),
            # Both forms add one to a function that reaches minus one, which cancels in float64
            # too. The reference takes the sum whole: `0.5 (1 + tanh(u))` is the logistic of
            # `2u`, and `1 + erf(z)` is `erfc(-z)`.
            "gelu": lambda: (
                x
                / (
                    1.0
                    + np.exp(
                        -np.sqrt(8.0 / np.pi) * x * (1.0 + 0.044715 * x * x)
                    )
                )
            ),
            "gelu_erf": lambda: erfc(-x / np.sqrt(2.0)) * 0.5 * x,
            "erf": lambda: erf(x),
            "relu": lambda: np.maximum(x, 0.0),
            "silu": lambda: x / (1.0 + np.exp(-x)),
            "sigmoid": lambda: 1.0 / (1.0 + np.exp(-x)),
            "ceil": lambda: np.ceil(x),
            "floor": lambda: np.floor(x),
            "round": lambda: away_from_zero(x),
            "sign": lambda: np.sign(x),
            "affine": lambda: x * a + b,
            "powf": lambda: np.power(x, a),
            "elu": lambda: np.where(x > 0, x, np.expm1(x) * a),
        }[function]()


def integer_formula(function, x, a, b):
    """`f(x)` on an integer array in its own element type, wrapping as the type does."""
    with np.errstate(over="ignore"):
        return {
            "neg": lambda: (0 - x).astype(x.dtype),
            "sqr": lambda: x * x,
            "abs": lambda: np.where(x < 0, (0 - x).astype(x.dtype), x),
            "relu": lambda: np.maximum(x, 0).astype(x.dtype),
            "ceil": lambda: x,
            "floor": lambda: x,
            "round": lambda: x,
            "sign": lambda: (
                (x > 0).astype(np.int64) - (x < 0).astype(np.int64)
            ).astype(x.dtype),
            "affine": lambda: (x * x.dtype.type(a) + x.dtype.type(b)).astype(
                x.dtype
            ),
        }[function]()


def bound(function, dtype):
    """`ElementwiseUnary::tolerance` for a floating element type."""
    roundings = ROUNDINGS[function]
    if roundings == 0:
        return operands.EXACT
    if function == "affine":
        return operands.dot(roundings, dtype)
    if function in ("gelu", "gelu_erf", "elu"):
        unit = operands.UNIT[operands.accumulate(dtype)]
        return operands.with_atol(
            operands.rounding(roundings, dtype), 2.0 * VALUE_ENVELOPE * unit
        )
    return operands.rounding(roundings, dtype)


def in_domain(function, x):
    """`x` moved into the function's domain, in its own element type."""
    if function in ("log", "sqrt", "powf"):
        # A positive argument, kept away from zero where the logarithm has no bound.
        return (x.abs() + 0.125).to(x.dtype)
    if function == "recip":
        return torch.where(x.abs() < 0.125, torch.full_like(x, 0.5), x)
    return x


def launch(function, dtype, x):
    """One launch over the view `x`, and the output it wrote."""
    out = operands.empty(x.shape, x.dtype)
    extents, (x_strides, _) = operands.merged([x, out])
    elements = out.numel()

    if dtype in operands.FLOATS:
        a, b = SCALARS.get(function, (0.0, 0.0))
    else:
        a, b = INTEGER_SCALARS if function == "affine" else (0, 0)

    # `UnaryDescriptor`'s `Display`.
    key = (
        f"function_{function}_dtype_{dtype}_rank_{len(extents)}"
        f"_block_{operands.BLOCK}_pdl_false"
    )
    operands.launch(
        "elementwise_unary",
        key,
        operands.grid(elements),
        x,
        out,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(x_strides),
        a,
        b,
        0,
    )
    return out, a, b


def check(function, dtype, name, x):
    """One launch held to the function's formula and the operator's bound."""
    label = f"{function} {dtype} {name} {tuple(x.shape)}"
    if dtype in operands.FLOATS:
        x = in_domain(function, x)
        out, a, b = launch(function, dtype, x)
        expected = formula(function, operands.real(x), a, b)
        return operands.compare(label, out, expected, bound(function, dtype))

    out, a, b = launch(function, dtype, x)
    numpy_dtype = np.dtype(str(x.dtype).removeprefix("torch."))
    expected = integer_formula(
        function, operands.real(x).astype(numpy_dtype), a, b
    )
    return operands.compare(
        label, out, expected.astype(np.int64), operands.EXACT
    )


#: Values where a formula is decided, all exact in every floating element type.
EDGES = [
    0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.25, -0.25, 0.75, -0.75, 1.0, -1.0, 3.0, -3.0,
    6.0, -6.0, 0.0078125, -0.0078125, 40.0, -40.0,
]  # fmt: skip


def main():
    if not operands.ready():
        return 0

    checked = 0
    functions = list(ROUNDINGS)

    # Every function on every floating type: a contiguous input, a permuted one, and the edges.
    for dtype in operands.FLOATS:
        for seed, function in enumerate(functions):
            contiguous = operands.values(dtype, (5, 7), seed)
            permuted = operands.values(dtype, (3, 4, 5), seed + 100).permute(
                2, 0, 1
            )
            edges = operands.tensor(
                torch.tensor(EDGES, dtype=torch.float64), dtype
            )
            checked += check(function, dtype, "contiguous", contiguous)
            checked += check(function, dtype, "permuted", permuted)
            checked += check(function, dtype, "edges", edges)

    # A base of either sign under a whole exponent, which the positive domain above leaves out.
    SCALARS["powf"] = (3.0, 0.0)
    for dtype in ("f32", "f64"):
        signed = operands.values(dtype, (6, 5), 7).t()
        out, a, b = launch("powf", dtype, signed)
        expected = formula("powf", operands.real(signed), a, b)
        checked += operands.compare(
            f"powf {dtype} whole exponent", out, expected, bound("powf", dtype)
        )

    # The functions closed over the integers, on every integer type.
    for dtype in operands.INTEGERS:
        for seed, function in enumerate(INTEGRAL):
            contiguous = operands.values(dtype, (5, 7), seed)
            permuted = operands.values(dtype, (3, 4, 5), seed + 100).permute(
                2, 0, 1
            )
            checked += check(function, dtype, "contiguous", contiguous)
            checked += check(function, dtype, "permuted", permuted)

    # Every layout, on a function with arithmetic, one that moves a value, and an integer one.
    layouts = 0
    for dtype, function in (
        ("f32", "affine"),
        ("bf16", "silu"),
        ("f64", "neg"),
        ("i32", "affine"),
        ("u8", "sqr"),
    ):
        for name, view in operands.views(dtype, 31):
            checked += check(function, dtype, name, view)
            layouts += 1

    print(
        f"ok: {checked} elements over {len(functions)} functions on {len(operands.FLOATS)} floating "
        f"types, {len(INTEGRAL)} on {len(operands.INTEGERS)} integer types, and {layouts} strided "
        f"launches of rank 1 to 6, within the operator's bounds on {operands.where()}"
        + (
            ""
            if operands.DEVICE
            else " (numpy stands in for libdevice's exp, tanh and pow)"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
