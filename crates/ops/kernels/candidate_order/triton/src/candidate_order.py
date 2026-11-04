"""Putting each row's candidates in ascending token order: the Triton source of
the ``candidate_order`` operator.

The operator is a permutation and nothing else, and its purpose is determinism:
the reference orders the candidates by token id before an inverse-CDF draw so a
draw depends on the random value and not on the order a top-k program happened
to produce.

Each program owns one row and computes, for every candidate, the number of
candidates that come before it — a rank — then writes that candidate at its
rank. The order is **ascending id, ties by source position**: a real row never
repeats a token id, but the rule is written down and implemented so a duplicated
id is a decided outcome rather than an accident of the sort.

Width is a **launch** value, not a compiled one (a request's selection depth
varies), so the kernel walks the row in tiles of ``TILE`` twice: once over the
candidates it writes and once over the candidates it compares against. Any width
is served by the same module; a row wider than one tile costs more passes, and a
row of a handful of candidates costs one.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["rows", "width"])
def _candidate_order_kernel(
    values,
    ids,
    ordered_values,
    ordered_ids,
    rows,
    width,
    TILE: tl.constexpr,
    PDL: tl.constexpr,
):
    """Write one row's candidates in ascending id order.

    Grid: ``[rows]``. ``values``/``ordered_values`` share the values' element
    type; ``ids``/``ordered_ids`` share the ids'.
    """
    _pdl_prologue(PDL)
    row = tl.program_id(0).to(tl.int64)
    if row >= rows:
        return
    base = row * width

    index = tl.arange(0, TILE)
    # The candidates are written at the rank their comparisons give them; the
    # comparison pass below visits the same row tile by tile.
    for write_start in range(0, width, TILE):
        position = write_start + index
        writing = position < width
        candidate = tl.load(ids + base + position, mask=writing, other=0).to(tl.int64)
        value = tl.load(values + base + position, mask=writing, other=0.0)

        rank = tl.zeros([TILE], tl.int32)
        for read_start in range(0, width, TILE):
            other_position = read_start + index
            reading = other_position < width
            other = tl.load(ids + base + other_position, mask=reading, other=0).to(tl.int64)
            before = other[None, :] < candidate[:, None]
            # Equal ids are ordered by where they arrived: the earlier candidate
            # keeps the lower rank, so the permutation is total.
            tie = (other[None, :] == candidate[:, None]) & (
                other_position[None, :] < position[:, None]
            )
            counted = (before | tie) & reading[None, :] & writing[:, None]
            rank += tl.sum(counted.to(tl.int32), axis=1)

        destination = base + rank.to(tl.int64)
        tl.store(ordered_values + destination, value, mask=writing)
        tl.store(ordered_ids + destination, candidate, mask=writing)
