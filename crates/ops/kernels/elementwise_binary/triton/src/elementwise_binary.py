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
"""

import strided
import triton
import triton.language as tl


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
    pdl,
    FUNCTION: tl.constexpr,  # noqa: N803
    COMPARES: tl.constexpr,  # noqa: N803
    COMPUTE: tl.constexpr,  # noqa: N803
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of ``lhs o rhs``.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the output's
    extents, ``l0..l5`` and ``r0..r5`` the two inputs' strides, right-aligned.
    ``COMPARES`` says that ``FUNCTION`` is a comparison, whose output is ``u8``.
    """
    strided.pdl_prologue(PDL)

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
    else:
        result = left >= right

    if COMPARES:
        tl.store(out + index, result.to(tl.uint8), mask=live)
    else:
        tl.store(out + index, strided.narrowed(result, out, COMPUTE), mask=live)
