"""Elements read or written at indexed positions: ``index``'s source.

```text
gather        dst[.., j, ..]                  = src[.., indices[.., j, ..], ..]
scatter_set   dst[.., indices[.., j, ..], ..]  = src[.., j, ..]
scatter_add   dst[.., indices[.., j, ..], ..] += src[.., j, ..]
```

``src`` is the source and ``dst`` the destination.

Only the position along one axis goes through the indices. The source and
the indices are read through their strides (``strided.offsets``) and the
destination is contiguous. An index vector that serves every position off
the axis arrives as indices with a zero stride off the axis, so
``index_select`` is a ``gather`` and ``index_add`` a ``scatter_add``.

**A gather** is one kernel over the destination's elements: an element's
source is the offset of the source row that index zero selects, plus its
index times the source's step along the axis. The axis is therefore a word
of the launch like the strides, and the programs are independent.

**A scatter** is one kernel over columns, the positions off the axis. A
program takes a block of columns and traverses the axis in order, so the writes
of one column are sequential and no two programs share a destination
element: a repeated index is well defined, the last source along the axis
staying for ``scatter_set`` and the sources adding in axis order for
``scatter_add``. Each addition is evaluated in FP32 (FP64 for FP64, the
element's own type for an integer) and rounded to nearest even into the
element type.

An index equal to its type's largest value (``ABSENT``) names no element: a
gather stores zero for it and a scatter skips it. Any other index is used as
it is; one at or past the extent addresses outside the operand, which the
operator's caller excludes.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch, strided


@triton.jit
def _gather_kernel(
    source,
    indices,
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
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    step,
    ABSENT: tl.constexpr,  # noqa: N803
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of the destination from the indexed source elements.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the
    destination's extents; ``s0..s5`` are the source's strides with the
    indexed axis's zeroed, ``i0..i5`` the indices' strides, right-aligned;
    ``step`` is the source's stride along the indexed axis.
    """
    dependent_launch.prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < elements
    row_at = strided.offsets(
        index, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5, RANK
    )
    index_at = strided.offsets(
        index, d1, d2, d3, d4, d5, i0, i1, i2, i3, i4, i5, RANK
    )

    chosen = tl.load(indices + index_at, mask=live, other=0).to(tl.int64)
    present = live & (chosen != ABSENT)

    # An absent index reads nothing and stores the zero its masked load gives.
    value = tl.load(source + row_at + chosen * step, mask=present, other=0)
    tl.store(destination + index, value, mask=live)


@triton.jit
def _scatter_kernel(
    source,
    indices,
    destination,
    columns,
    length,
    c1,
    c2,
    c3,
    c4,
    c5,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    source_step,
    indices_step,
    destination_step,
    ADDS: tl.constexpr,  # noqa: N803
    COMPUTE: tl.constexpr,  # noqa: N803
    ABSENT: tl.constexpr,  # noqa: N803
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Scatter the source into one block of the destination's columns.

    Grid: ``ceil(columns / BLOCK)`` programs. ``c1..c5`` are the extents of
    the axes off the indexed one; ``s0..s5``, ``i0..i5`` and ``t0..t5`` are
    the strides of the source, the indices and the destination along them,
    right-aligned. ``length`` is the source's extent along the indexed axis
    and the three steps are the operands' strides along it. ``ADDS`` says
    that the function accumulates instead of replacing.
    """
    dependent_launch.prologue(PDL)

    column = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = column < columns
    source_at = strided.offsets(
        column, c1, c2, c3, c4, c5, s0, s1, s2, s3, s4, s5, RANK
    )
    index_at = strided.offsets(
        column, c1, c2, c3, c4, c5, i0, i1, i2, i3, i4, i5, RANK
    )
    target_at = strided.offsets(
        column, c1, c2, c3, c4, c5, t0, t1, t2, t3, t4, t5, RANK
    )

    # The axis is traversed in order: each step reads what the one before it
    # wrote, which is what makes a repeated index accumulate or be replaced.
    for along in range(length):
        chosen = tl.load(
            indices + index_at + along * indices_step, mask=live, other=0
        ).to(tl.int64)
        present = live & (chosen != ABSENT)

        value = tl.load(
            source + source_at + along * source_step, mask=present, other=0
        )
        target = destination + target_at + chosen * destination_step

        if ADDS:
            held = tl.load(target, mask=present, other=0)
            total = strided.widened(held, COMPUTE) + strided.widened(
                value, COMPUTE
            )
            value = strided.narrowed(total, destination, COMPUTE)
        tl.store(target, value, mask=present)


#: The index that names no element, by the indices' type: its largest value.
ABSENT = {"u8": 2**8 - 1, "u32": 2**32 - 1, "i64": 2**63 - 1}


def derive(fields, parameters, target):
    """The element addressed, the absent index and what a scatter computes in.

    Only an addition computes, and so only it names a computing type: a
    ``scatter_set`` of an element that only moves stores the bits it loaded.
    """
    adds = fields["function"] == "scatter_add"
    return {
        "address": strided.address(fields["dtype"]),
        "absent": ABSENT[fields["indices"]],
        "adds": adds,
        "compute": strided.compute(fields["dtype"]) if adds else "int",
    }
