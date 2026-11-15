"""A strided source moved into a row-major window: ``copy_strided``'s source.

```text
destination[i] = source[i]
```

One program moves one block of the shape's row-major elements. Both sides
are addressed through their strides (``strided.offsets``): the source may be
any view, and the destination is a contiguous tensor or a block of a larger
one, read with the larger one's strides. The element type and the number of
axes are compiled in; the shape and both stride vectors are words of the
launch.

No element is converted, so the destination receives the source's bits. The
operator holds the destination to a layout in which no two elements share an
address, so the programs of one launch never write the same element.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch, strided


@triton.jit
def _copy_strided_kernel(
    source,
    destination,
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
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Move one block of the source to where the destination holds it.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the shape's
    extents; ``s0..s5`` and ``t0..t5`` are the strides of the source and of
    the destination, right-aligned.
    """
    dependent_launch.prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < elements
    read_at = strided.offsets(
        index, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5, RANK
    )
    write_at = strided.offsets(
        index, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5, RANK
    )

    moved = tl.load(source + read_at, mask=live, other=0)
    tl.store(destination + write_at, moved, mask=live)
