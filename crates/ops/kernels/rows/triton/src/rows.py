"""Place or accumulate picked-out rows: the ``rows`` operator's Triton source.

Two modalities feed rows into a sequence — an image's vision tokens replace the placeholder rows the
text reserved for them, and a deep-stack tap adds a tower's rows into rows that already hold a
residual — so they are one operator with a compiled mode:

```text
replace: destination[indices[j]] = source[j]        (a copy)
add:     destination[indices[j]] += source[j]       (one rounding into the output's dtype)
place:   destination[indices[j]] = source[j]        for indices[j] >= 0, nothing otherwise
```

``place`` is ``replace`` with a row that may name no destination: a negative index stores nothing,
so a batch's tokens that leave no row (and the padding tokens of a bucket) cost one masked store.

A program is one (source row, block of columns), which is the grid the operator launches: a row's
elements are never split across two blocks, so no two blocks write the same destination *element*
of an index that appears once, and the row's own writes are ordered inside its blocks. The indices
are i32 as the caller passes them; the destination is row-major and the row width is the
descriptor's, so a row's base is `index · columns`.

**A repeated index.** The operator allows one, and the grid above orders each source row's own
writes. Two source rows that name the *same* destination row land in different blocks, and nothing
in the call orders those two blocks against each other — so with a repeated index the result is
whichever block wrote last, and for `add` the lost update is silent. This source cannot fix that
without abandoning the operator's grid (one block per destination row and a scan of the indices,
which is a different program); it therefore does what the call says and this note says what that
means. A caller that repeats an index wants its own ordering, not the scheduler's.

``add`` reads the destination, adds in FP32 and rounds once on the store, which is the operator's
tolerance for it: two units of the destination dtype's own rounding. ``replace`` is a copy and
touches no arithmetic at all.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch


@triton.jit
def _rows_kernel(
    source_ptr,
    destination_ptr,
    indices_ptr,
    count,
    columns,
    block,
    COLUMNS: tl.constexpr,
    REPLACE: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
    SKIP: tl.constexpr,  # noqa: N803
):
    """Place one source row's block of columns at the row its index names.

    ``COLUMNS`` is the row width, ``REPLACE`` whether the row is copied and not added, ``SKIP``
    whether a negative index stores nothing, and ``BLOCK`` the lanes one program covers. ``count``, ``columns`` and ``block`` are the launch's words: the grid
    names the source row and the column slice.
    """
    dependent_launch.prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    lanes = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = lanes < COLUMNS

    destination_row = tl.load(indices_ptr + row).to(tl.int64)
    if SKIP:
        # A negative index names no row: every lane of the row is masked out of its store.
        live = live & (destination_row >= 0)
    source = tl.load(source_ptr + row * COLUMNS + lanes, mask=live, other=0.0)
    target = destination_ptr + destination_row * COLUMNS + lanes

    if REPLACE:
        value = source
    else:
        # One addition in FP32 and a single rounding on the store, which is the mode's tolerated
        # arithmetic: the sum and the store each round.
        value = tl.load(target, mask=live, other=0.0).to(tl.float32) + source.to(tl.float32)

    tl.store(target, value.to(destination_ptr.dtype.element_ty, fp_downcast_rounding="rtne"), mask=live)
