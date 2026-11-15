"""The elementwise_unary program against numpy, for every function.

Each case launches the package's kernel as ``ElementwiseUnary::launch`` does
(the input and the output merged into the axes a program traverses, the
extents and the strides packed right-aligned) and compares the output with
the function's formula evaluated in float64 on the same values, inside the
operator's own bound (``ElementwiseUnary::tolerance``).

What is covered:

- **every function** on every floating element type, over a contiguous input
  and a permuted one, and on values where a formula is decided: a half
  (``round``), a signed zero and both signs (``sign``, ``relu``, ``abs``), the
  two ends of a saturating activation (``gelu``, ``gelu_erf``, ``silu``,
  ``sigmoid``), and the neighbourhood of zero where ``elu``'s difference
  cancels; ``powf`` also over a base of either sign under a whole exponent;
- **the functions closed over the integers** on every integer element type,
  wrapping as the element type does;
- **every layout** of :data:`~tests.python.kernels.tensor.operands.LAYOUTS`.

``tanh``, ``powf`` and every exponential (``exp``, and the one inside
``gelu``, ``silu``, ``sigmoid`` and ``elu``) call libdevice, which Triton's
interpreter cannot evaluate. Numpy's ``tanh``, ``power`` and ``exp`` stand in
for those calls there (:func:`interpreted_libdevice`), so the interpreter run
holds which values reach them, in which order, and where the results are
stored; the calls' own last place is libdevice's, and a device run holds it
to the bound. A BF16 output is rounded to nearest on a device and truncated by
the interpreter, which the comparison allows for.
"""

import numpy as np
import pytest
import torch
import triton.language as tl
from triton.runtime import interpreter

from tests.python.kernels.tensor import operands

pytestmark = pytest.mark.unit

#: `VALUE_ENVELOPE` of `crates/ops/src/elementwise.rs`.
VALUE_ENVELOPE = 8.0

#: The scalars of the three functions that take them.
SCALARS = {"affine": (1.5, -0.25), "powf": (2.5, 0.0), "elu": (0.75, 0.0)}

#: `powf`'s scalars over a base of either sign: a whole exponent.
WHOLE_POWER = (3.0, 0.0)

#: The integer scalars of `affine`: a factor and an addend every integer type
#: holds.
INTEGER_SCALARS = (3, 5)

#: Each function's rounded operations (`UnaryFunction::roundings`).
ROUNDINGS = {
    "exp": 1, "log": 1, "sin": 1, "cos": 1, "tanh": 1, "neg": 0, "recip": 1,
    "sqr": 1, "sqrt": 1, "abs": 0, "gelu": 8, "gelu_erf": 4, "erf": 1,
    "relu": 0, "silu": 3, "sigmoid": 3, "ceil": 0, "floor": 0, "round": 0,
    "sign": 0, "affine": 2, "powf": 1, "elu": 3,
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

#: Values where a formula is decided, all exact in every floating element
#: type.
EDGES = [
    0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.25, -0.25, 0.75, -0.75, 1.0,
    -1.0, 3.0, -3.0, 6.0, -6.0, 0.0078125, -0.0078125, 40.0, -40.0,
]  # fmt: skip

#: The functions every layout is launched with: one with arithmetic, one that
#: moves a value, and integer ones.
LAYOUT_FUNCTIONS = (
    ("f32", "affine"),
    ("bf16", "silu"),
    ("f64", "neg"),
    ("i32", "affine"),
    ("u8", "sqr"),
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
        # An exponential past the type's largest value is an infinity, as on a
        # device.
        with np.errstate(over="ignore"):
            return Libdevice._applied(np.exp, value)

    @staticmethod
    def tanh(value):
        return Libdevice._applied(np.tanh, value)

    @staticmethod
    def pow(base, exponent):
        return Libdevice._applied(np.power, base, exponent)


@pytest.fixture(scope="module", autouse=True)
def interpreted_libdevice(interpreted, kernel_source):
    """Numpy's functions in the place of the kernel's libdevice calls, on the
    interpreter, for this module's tests.
    """  # noqa: D205
    if not interpreted:
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            kernel_source("elementwise_unary"), "libdevice", Libdevice
        )
        yield


def away_from_zero(x):
    """The nearest integer, a half away from zero, decided on the exact
    fraction.
    """  # noqa: D205
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
            # Both forms add one to a function that reaches minus one, which
            # cancels in float64 too. The reference takes the sum whole:
            # `0.5 (1 + tanh(u))` is the logistic of `2u`, and `1 + erf(z)` is
            # `erfc(-z)`.
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
    """`f(x)` on an integer array in its own element type, wrapping as the
    type does.
    """  # noqa: D205
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
        # A positive argument, kept away from zero where the logarithm has no
        # bound.
        return (x.abs() + 0.125).to(x.dtype)
    if function == "recip":
        return torch.where(x.abs() < 0.125, torch.full_like(x, 0.5), x)
    return x


def launch(primitives, function, dtype, x, scalars):
    """One launch over the view `x` with the function's scalars `(a, b)`, and
    the output it wrote.
    """  # noqa: D205
    out = primitives.empty(x.shape, x.dtype)
    extents, (x_strides, _) = operands.merged([x, out])
    elements = out.numel()
    # `UnaryDescriptor`'s `Display`.
    key = (
        f"function_{function}_dtype_{dtype}_rank_{len(extents)}"
        f"_block_{operands.BLOCK}_pdl_false"
    )
    primitives.launch(
        "elementwise_unary",
        key,
        operands.grid(elements),
        x,
        out,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(x_strides),
        *scalars,
        0,
    )
    return out


def check(primitives, function, dtype, name, x):
    """One launch held to the function's formula and the operator's bound."""
    label = f"{function} {dtype} {name} {tuple(x.shape)}"
    if dtype in operands.FLOATS:
        x = in_domain(function, x)
        scalars = SCALARS.get(function, (0.0, 0.0))
        out = launch(primitives, function, dtype, x, scalars)
        expected = formula(function, operands.real(x), *scalars)
        primitives.assert_close(label, out, expected, bound(function, dtype))
        return

    scalars = INTEGER_SCALARS if function == "affine" else (0, 0)
    out = launch(primitives, function, dtype, x, scalars)
    numpy_dtype = np.dtype(str(x.dtype).removeprefix("torch."))
    expected = integer_formula(
        function, operands.real(x).astype(numpy_dtype), *scalars
    )
    primitives.assert_close(
        label, out, expected.astype(np.int64), operands.EXACT
    )


@pytest.mark.parametrize(("seed", "function"), list(enumerate(ROUNDINGS)))
@pytest.mark.parametrize("dtype", operands.FLOATS)
def test_a_function_matches_its_formula_on_a_floating_type(
    dtype, seed, function, primitives
):
    contiguous = primitives.values(dtype, (5, 7), seed)
    permuted = primitives.values(dtype, (3, 4, 5), seed + 100).permute(2, 0, 1)
    edges = primitives.tensor(torch.tensor(EDGES, dtype=torch.float64), dtype)
    check(primitives, function, dtype, "contiguous", contiguous)
    check(primitives, function, dtype, "permuted", permuted)
    check(primitives, function, dtype, "edges", edges)


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_a_base_of_either_sign_takes_a_whole_power(dtype, primitives):
    """The positive domain of the other cases leaves this one out."""
    signed = primitives.values(dtype, (6, 5), 7).t()
    out = launch(primitives, "powf", dtype, signed, WHOLE_POWER)
    expected = formula("powf", operands.real(signed), *WHOLE_POWER)
    primitives.assert_close(
        f"powf {dtype} whole exponent", out, expected, bound("powf", dtype)
    )


@pytest.mark.parametrize(("seed", "function"), list(enumerate(INTEGRAL)))
@pytest.mark.parametrize("dtype", operands.INTEGERS)
def test_a_function_wraps_on_an_integer_type(dtype, seed, function, primitives):
    contiguous = primitives.values(dtype, (5, 7), seed)
    permuted = primitives.values(dtype, (3, 4, 5), seed + 100).permute(2, 0, 1)
    check(primitives, function, dtype, "contiguous", contiguous)
    check(primitives, function, dtype, "permuted", permuted)


@pytest.mark.parametrize("layout", operands.LAYOUTS)
@pytest.mark.parametrize(("dtype", "function"), LAYOUT_FUNCTIONS)
def test_every_layout_is_traversed(dtype, function, layout, primitives):
    view = primitives.view(layout, dtype, 31)
    check(primitives, function, dtype, layout, view)
