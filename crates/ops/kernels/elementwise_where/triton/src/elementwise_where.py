"""A selection between two strided inputs: ``elementwise_where``'s source.

```text
out[i] = condition[i] != 0 ? on_true[i] : on_false[i]
```

One program selects one block of the output's row-major elements. The
condition and both values are read through their strides
(``strided.offsets``), so any of them may be a permuted or a broadcast view;
the output is contiguous. The element types and the number of axes are
compiled in; the shape and the strides are words of the launch.

A selection moves one of its two values as it is: no element is converted,
so the output holds the bits of the value it chose.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch, strided


@triton.jit
def _elementwise_where_kernel(
    condition,
    on_true,
    on_false,
    out,
    elements,
    d1,
    d2,
    d3,
    d4,
    d5,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    f0,
    f1,
    f2,
    f3,
    f4,
    f5,
    pdl,
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of the selection.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the output's
    extents; ``c0..c5``, ``t0..t5`` and ``f0..f5`` are the strides of the
    condition, of ``on_true`` and of ``on_false``, right-aligned.
    """
    dependent_launch.prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < elements
    condition_at = strided.offsets(
        index, d1, d2, d3, d4, d5, c0, c1, c2, c3, c4, c5, RANK
    )
    true_at = strided.offsets(
        index, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5, RANK
    )
    false_at = strided.offsets(
        index, d1, d2, d3, d4, d5, f0, f1, f2, f3, f4, f5, RANK
    )

    holds = tl.load(condition + condition_at, mask=live, other=0) != 0
    chosen = tl.where(
        holds,
        tl.load(on_true + true_at, mask=live, other=0),
        tl.load(on_false + false_at, mask=live, other=0),
    )
    tl.store(out + index, chosen, mask=live)
