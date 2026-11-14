"""A strided input folded along some of its axes: ``reduce``'s source.

```text
sum      out[k] = sum of x[k, r]         over every index r of the reduced axes
min      out[k] = the least x[k, r]      max likewise
absmax   out[k] = the largest |x[k, r]|  of a floating input
argmin   out[k] = the first r at which x[k, r] is least; argmax likewise
```

``k`` counts the kept axes and ``r`` the reduced ones, each in row-major
order, and an element's offset is the sum of the two (``strided.offsets``
over each index space). The output is contiguous over ``k``.

One program folds ``ROWS`` outputs, reading ``TILE`` elements of the
reduction for each of them at a time: a ``[ROWS, TILE]`` tile per step, and
``ceil(length / TILE)`` steps. ``ROWS * TILE`` is the block, so a long
reduction of few outputs (``TILE`` is the whole block) and a short one of
many (``TILE`` is the reduction's class) both read a full block per step.

A sum adds each tile into ``[ROWS, TILE]`` partial sums, one per position of
the tile, and folds a row's partial sums once after the last step. An
extremum folds its tile at every step.

A floating element accumulates in FP32, or in FP64 when it is FP64, and
rounds to nearest even once into the element type. An integer accumulates in
64 bits and is stored at its own width, which is the wrapped sum. Each
output starts at the function's neutral value: zero for a sum, the type's
extreme for an extremum. The first of equal values wins an index: within a
tile the leftmost, across tiles the earlier, because a later tile replaces
the best only when it is strictly better.
"""

import strided
import triton
import triton.language as tl


@triton.jit
def _widened(value, ACCUMULATE: tl.constexpr):  # noqa: N803
    """A loaded tile in the type the reduction accumulates in."""
    if ACCUMULATE == "fp32":
        result = value.to(tl.float32)
    elif ACCUMULATE == "fp64":
        result = value.to(tl.float64)
    else:
        result = value.to(tl.int64)
    return result


@triton.jit
def _reduce_kernel(
    x,
    out,
    outputs,
    length,
    k1,
    k2,
    k3,
    k4,
    k5,
    ks0,
    ks1,
    ks2,
    ks3,
    ks4,
    ks5,
    r1,
    r2,
    r3,
    r4,
    r5,
    rs0,
    rs1,
    rs2,
    rs3,
    rs4,
    rs5,
    pdl,
    FUNCTION: tl.constexpr,  # noqa: N803
    ACCUMULATE: tl.constexpr,  # noqa: N803
    NEUTRAL: tl.constexpr,  # noqa: N803
    KEPT: tl.constexpr,  # noqa: N803
    REDUCED: tl.constexpr,  # noqa: N803
    ROWS: tl.constexpr,  # noqa: N803
    TILE: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Fold ``ROWS`` outputs.

    Grid: ``ceil(outputs / ROWS)`` programs. ``k1..k5`` and ``ks0..ks5`` are
    the extents and the strides of the kept axes, ``r1..r5`` and ``rs0..rs5``
    those of the reduced axes, right-aligned; ``length`` is the number of
    elements one output folds. ``NEUTRAL`` is the value every output starts
    at.
    """
    strided.pdl_prologue(PDL)

    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live_rows = rows < outputs
    base = strided.offsets(
        rows, k1, k2, k3, k4, k5, ks0, ks1, ks2, ks3, ks4, ks5, KEPT
    )

    if ACCUMULATE == "fp32":
        best = tl.full([ROWS], NEUTRAL, tl.float32)
    elif ACCUMULATE == "fp64":
        best = tl.full([ROWS], NEUTRAL, tl.float64)
    else:
        best = tl.full([ROWS], NEUTRAL, tl.int64)
    position = tl.zeros([ROWS], dtype=tl.int64)

    # A sum is reduced along the tile once, after the loop, and never inside
    # it. Triton 3.7 rewrites a floating reduction of a loaded tile that
    # accumulates across a loop into per-thread partial sums, regrouping each
    # thread's elements; where a thread holds elements of several rows and
    # the rows are known to be adjacent in memory, the regrouping pairs
    # elements of different rows and the stored sums are wrong.
    if FUNCTION == "sum":
        partial = tl.zeros([ROWS, TILE], dtype=best.dtype)

    for start in range(0, length, TILE):
        columns = start + tl.arange(0, TILE).to(tl.int64)
        live = live_rows[:, None] & (columns[None, :] < length)
        inner = strided.offsets(
            columns, r1, r2, r3, r4, r5, rs0, rs1, rs2, rs3, rs4, rs5, REDUCED
        )

        loaded = _widened(
            tl.load(x + base[:, None] + inner[None, :], mask=live, other=0),
            ACCUMULATE,
        )
        # A position that holds no element repeats its row's best so far,
        # which changes neither an extremum nor where it was found; a sum reads
        # the zero its load left there.
        values = tl.where(live, loaded, best[:, None])

        if FUNCTION == "sum":
            partial += loaded
        elif FUNCTION == "min":
            best = tl.minimum(best, tl.min(values, axis=1))
        elif FUNCTION == "max":
            best = tl.maximum(best, tl.max(values, axis=1))
        elif FUNCTION == "absmax":
            # A position that holds no element repeats the row's best magnitude, which is its
            # own magnitude.
            best = tl.maximum(best, tl.max(tl.abs(values), axis=1))
        elif FUNCTION == "argmin":
            lowest = tl.min(values, axis=1)
            taken = lowest < best
            at = start + tl.argmin(values, axis=1).to(tl.int64)
            position = tl.where(taken, at, position)
            best = tl.where(taken, lowest, best)
        else:
            highest = tl.max(values, axis=1)
            taken = highest > best
            at = start + tl.argmax(values, axis=1).to(tl.int64)
            position = tl.where(taken, at, position)
            best = tl.where(taken, highest, best)

    if FUNCTION == "sum":
        best += tl.sum(partial, axis=1)

    if FUNCTION == "argmin":
        tl.store(out + rows, position.to(tl.uint32), mask=live_rows)
    elif FUNCTION == "argmax":
        tl.store(out + rows, position.to(tl.uint32), mask=live_rows)
    elif ACCUMULATE == "i64":
        tl.store(out + rows, best.to(out.dtype.element_ty), mask=live_rows)
    else:
        tl.store(
            out + rows,
            best.to(out.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=live_rows,
        )
