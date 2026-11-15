"""One element stored over a row-major window: the ``fill`` operator's source.

```text
destination[i] = value
```

One program stores one block of the destination's row-major elements. The
destination is addressed through its strides (``strided.offsets``): a
contiguous tensor, or a block of a larger one read with the larger one's
strides. The element type and the number of axes are compiled in; the shape
and the strides are words of the launch.

The value arrives as its bits, in an unsigned word as wide as the element,
and is stored as those bits: nothing is converted, so what a launch writes
does not depend on a rounding mode or on how a compiler converts a scalar.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch, strided


@triton.jit
def _fill_kernel(
    destination,
    elements,
    d1,
    d2,
    d3,
    d4,
    d5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    bits,
    WIDTH: tl.constexpr,  # noqa: N803
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store the element over one block of the destination.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the
    destination's extents and ``t0..t5`` its strides, right-aligned. ``bits``
    is the element's bit pattern and ``WIDTH`` the element's width in bits.
    """
    dependent_launch.prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < elements
    write_at = strided.offsets(
        index, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5, RANK
    )

    # The word is narrowed to the element's width before it is reinterpreted:
    # a bit cast keeps every bit, so both sides have to be as wide.
    if WIDTH == 8:
        word = bits.to(tl.uint8)
    elif WIDTH == 16:
        word = bits.to(tl.uint16)
    elif WIDTH == 32:
        word = bits.to(tl.uint32)
    else:
        word = bits.to(tl.uint64)

    pattern = tl.full([BLOCK], word, word.dtype)
    element = pattern.to(destination.dtype.element_ty, bitcast=True)
    tl.store(destination + write_at, element, mask=live)


#: The width in bits of each unsigned word an element's bits travel in.
WIDTHS = {"u8": 8, "u16": 16, "u32": 32, "u64": 64}


def derive(fields, parameters, target):
    """The element addressed, and the word its bits travel in, for the launch table."""
    word = strided.BITS[fields["dtype"]]
    return {
        "address": strided.address(fields["dtype"]),
        "word": word,
        "width": WIDTHS[word],
    }
