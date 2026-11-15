"""``out[i] = lhs[i] o rhs[i]`` over strided inputs: ``elementwise_binary``.

One program evaluates one block of the output's row-major elements. Both
inputs are read through their strides (``strided.offsets``), so a broadcast
operand is one whose stride is zero along the axes it repeats; the output is
contiguous. The function, the computing type and the number of axes are
compiled in; the shape and the strides are words of the launch.

Arithmetic (``add``, ``sub``, ``mul``, ``div``) widens a half-precision
element to FP32 and rounds to nearest even once into the element type; FP32
and FP64 evaluate in their own type, and an integer in its own, wrapping, its
quotient truncated toward zero. ``maximum`` and ``minimum`` return the left
operand unless the right one is strictly beyond it, which is the tensor
library's host rule and decides what a NaN does. A comparison (``eq``, ``ne``,
``lt``, ``le``, ``gt``, ``ge``) writes one ``u8`` per element: one where it
holds, zero where it does not.

Six functions exist on the integer element types alone and are exact, as all
integer arithmetic is. ``remainder`` is ``torch.remainder``: the remainder of
the floored quotient, zero or of the divisor's sign. ``bitwise_and``,
``bitwise_or`` and ``bitwise_xor`` combine the two elements' bits, and
``bitwise_left_shift`` and ``bitwise_right_shift`` shift the left element by
the right one, a right shift repeating the sign bit of a signed element. A
shift count outside the element's width and a zero divisor leave their
element unspecified.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch, strided


@triton.jit
def _elementwise_binary_kernel(
    lhs,
    rhs,
    out,
    elements,
    d1,
    d2,
    d3,
    d4,
    d5,
    l0,
    l1,
    l2,
    l3,
    l4,
    l5,
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    FUNCTION: tl.constexpr,  # noqa: N803
    COMPARES: tl.constexpr,  # noqa: N803
    COMPUTE: tl.constexpr,  # noqa: N803
    SIGNED: tl.constexpr,  # noqa: N803
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of ``lhs o rhs``.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the output's
    extents, ``l0..l5`` and ``r0..r5`` the two inputs' strides, right-aligned.
    ``COMPARES`` says that ``FUNCTION`` is a comparison, whose output is ``u8``,
    and ``SIGNED`` that the element type is an integer with a sign.
    """
    dependent_launch.prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < elements
    left_at = strided.offsets(
        index, d1, d2, d3, d4, d5, l0, l1, l2, l3, l4, l5, RANK
    )
    right_at = strided.offsets(
        index, d1, d2, d3, d4, d5, r0, r1, r2, r3, r4, r5, RANK
    )

    # A position past the last element divides by one, never by zero.
    left = strided.widened(tl.load(lhs + left_at, mask=live, other=0), COMPUTE)
    right = strided.widened(
        tl.load(rhs + right_at, mask=live, other=1), COMPUTE
    )

    if FUNCTION == "add":
        result = left + right
    elif FUNCTION == "sub":
        result = left - right
    elif FUNCTION == "mul":
        result = left * right
    elif FUNCTION == "div":
        if COMPUTE == "int":
            result = left // right
        else:
            result = left / right
    elif FUNCTION == "maximum":
        result = tl.where(left < right, right, left)
    elif FUNCTION == "minimum":
        result = tl.where(left > right, right, left)
    elif FUNCTION == "eq":
        result = left == right
    elif FUNCTION == "ne":
        result = left != right
    elif FUNCTION == "lt":
        result = left < right
    elif FUNCTION == "le":
        result = left <= right
    elif FUNCTION == "gt":
        result = left > right
    elif FUNCTION == "ge":
        result = left >= right
    elif FUNCTION == "remainder":
        # The quotient truncates toward zero, so what it leaves has the sign
        # of the dividend; the floored quotient leaves one divisor more where
        # that sign is not the divisor's.
        divisor = right
        if SIGNED:
            # The smallest value by minus one overflows as a quotient. Every
            # value is a multiple of minus one, as it is of one.
            divisor = tl.where(right == -1, 1, right)
        truncated = left - (left // divisor) * divisor
        if SIGNED:
            crosses = (truncated != 0) & ((truncated < 0) != (divisor < 0))
            result = tl.where(crosses, truncated + divisor, truncated)
        else:
            result = truncated
    elif FUNCTION == "bitwise_and":
        result = left & right
    elif FUNCTION == "bitwise_or":
        result = left | right
    elif FUNCTION == "bitwise_xor":
        result = left ^ right
    elif FUNCTION == "bitwise_left_shift":
        result = left << right
    else:
        # A signed element shifts arithmetically, an unsigned one logically.
        result = left >> right

    if COMPARES:
        tl.store(out + index, result.to(tl.uint8), mask=live)
    else:
        tl.store(out + index, strided.narrowed(result, out, COMPUTE), mask=live)


#: The functions whose output is a ``u8`` truth value.
COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")

#: The functions that exist on the integer element types alone.
INTEGRAL = (
    "remainder",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
)

#: The integer element types that hold a sign.
SIGNED = ("i16", "i32", "i64")


def derive(fields, parameters, target):
    """The computing type, the output's element type and the sign, for the launch table.

    A function of the integers alone is refused for a floating element type.
    """
    function, dtype = fields["function"], fields["dtype"]
    compute = strided.compute(dtype)
    if function in INTEGRAL and compute != "int":
        raise SystemExit(
            f"{function} exists on the integer element types alone, not on "
            f"{dtype}"
        )
    compares = function in COMPARISONS
    return {
        "compute": compute,
        "compares": compares,
        "output": "u8" if compares else dtype,
        "signed": dtype in SIGNED,
    }
