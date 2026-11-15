"""``out[i] = f(x[i])`` over a strided input: ``elementwise_unary``'s source.

One program evaluates one block of the output's row-major elements. The
input is read through its strides (``strided.offsets``), the output is
contiguous, and the function, the computing type and the number of axes are
compiled in; the shape and the strides are words of the launch.

A half-precision element is widened to FP32 and the result rounds to nearest
even once into the element type; FP32 and FP64 evaluate in their own type and
an integer in its own, wrapping. The formulas are the tensor library's host
formulas:

```text
recip     1 / x                     sqr      x * x
relu      max(x, 0)                 sign     (x > 0) - (x < 0)
sigmoid   1 / (1 + exp(-x))         silu     x / (1 + exp(-x))
gelu      x / (1 + exp(-sqrt(8/pi) * x * (1 + 0.044715 * x^2)))
gelu_erf  (erf(x / sqrt(2)) + 1) * 0.5 * x
round     the nearest integer, a half away from zero
affine    x * a + b                 powf     x^a
elu       x > 0 ? x : (exp(x) - 1) * a
```

``gelu`` is the tanh approximation written as one logistic: the same function
with one exponential and no cancellation at either end. ``a`` and ``b`` are the
launch's scalars, in the computing type. On an integer element only the
functions closed over the integers are compiled, and ``ceil``, ``floor`` and
``round`` are the identity there.

Every exponential is libdevice's (``_exponential``), whose error is a few
units in the last place of its result whatever the argument.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from uniserve_kernels import dependent_launch, strided

#: ``sqrt(8 / pi)`` and ``sqrt(1 / 2)``: the two GELU forms' constants,
#: instantiated so a kernel may read them.
_SQRT_8_OVER_PI = tl.constexpr(1.5957691216057308)
_SQRT_HALF = tl.constexpr(0.7071067811865476)


@triton.jit
def _exponential(value):
    """``exp(value)``, relative to its own result over the whole range.

    Triton compiles ``tl.exp`` of an FP32 value to PTX's ``ex2.approx`` of the
    rounded product ``value * log2(e)``. That product's rounding is an
    absolute error of the exponent, so the result's relative error grows with
    ``|value|`` (about ``|value|`` unit round-offs), past any bound stated in
    roundings. libdevice reduces the argument first and stays within a few
    units in the last place.
    """
    return libdevice.exp(value)


@triton.jit
def _direction(value):
    """One where ``value`` is positive, minus one where negative, else zero."""
    return (value > 0).to(value.dtype) - (value < 0).to(value.dtype)


@triton.jit
def _evaluated(
    value,
    a,
    b,
    FUNCTION: tl.constexpr,  # noqa: N803
    COMPUTE: tl.constexpr,  # noqa: N803
):
    """``f(value)`` in the computing type, before the store's rounding."""
    if FUNCTION == "exp":
        result = _exponential(value)
    elif FUNCTION == "log":
        result = tl.log(value)
    elif FUNCTION == "sin":
        result = tl.sin(value)
    elif FUNCTION == "cos":
        result = tl.cos(value)
    elif FUNCTION == "tanh":
        result = libdevice.tanh(value)
    elif FUNCTION == "neg":
        result = -value
    elif FUNCTION == "recip":
        result = 1.0 / value
    elif FUNCTION == "sqr":
        result = value * value
    elif FUNCTION == "sqrt":
        result = tl.sqrt(value)
    elif FUNCTION == "abs":
        if COMPUTE == "int":
            # An unsigned element is never below zero, so it is left alone.
            result = tl.where(value < 0, -value, value)
        else:
            result = tl.abs(value)
    elif FUNCTION == "gelu":
        inner = _SQRT_8_OVER_PI * value * (1.0 + 0.044715 * value * value)
        result = value / (1.0 + _exponential(-inner))
    elif FUNCTION == "gelu_erf":
        result = (tl.erf(value * _SQRT_HALF) + 1.0) * 0.5 * value
    elif FUNCTION == "erf":
        result = tl.erf(value)
    elif FUNCTION == "relu":
        result = tl.maximum(value, 0)
    elif FUNCTION == "silu":
        result = value / (1.0 + _exponential(-value))
    elif FUNCTION == "sigmoid":
        result = 1.0 / (1.0 + _exponential(-value))
    elif FUNCTION == "ceil":
        if COMPUTE == "int":
            result = value
        else:
            result = tl.ceil(value)
    elif FUNCTION == "floor":
        if COMPUTE == "int":
            result = value
        else:
            result = tl.floor(value)
    elif FUNCTION == "round":
        if COMPUTE == "int":
            result = value
        else:
            # The fraction `value - whole` is exact, so the half is decided
            # without the rounding `floor(value + 0.5)` would add.
            whole = tl.where(value < 0, tl.ceil(value), tl.floor(value))
            away = tl.abs(value - whole) >= 0.5
            result = tl.where(away, whole + _direction(value), whole)
    elif FUNCTION == "sign":
        result = _direction(value)
    elif FUNCTION == "affine":
        if COMPUTE == "int":
            result = value * a.to(value.dtype) + b.to(value.dtype)
        else:
            result = value * a + b
    elif FUNCTION == "powf":
        result = libdevice.pow(value, a + tl.zeros_like(value))
    else:
        result = tl.where(value > 0, value, (_exponential(value) - 1.0) * a)
    return result


@triton.jit
def _elementwise_unary_kernel(
    x,
    out,
    elements,
    d1,
    d2,
    d3,
    d4,
    d5,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    a,
    b,
    pdl,
    FUNCTION: tl.constexpr,  # noqa: N803
    COMPUTE: tl.constexpr,  # noqa: N803
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of ``f(x)``.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the output's
    extents and ``s0..s5`` the input's strides, right-aligned; ``a`` and ``b``
    are the function's scalars, which a function that takes none ignores.
    """
    dependent_launch.prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < elements
    source = strided.offsets(
        index, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5, RANK
    )

    value = strided.widened(tl.load(x + source, mask=live, other=0), COMPUTE)
    result = _evaluated(value, a, b, FUNCTION, COMPUTE)
    tl.store(out + index, strided.narrowed(result, out, COMPUTE), mask=live)
