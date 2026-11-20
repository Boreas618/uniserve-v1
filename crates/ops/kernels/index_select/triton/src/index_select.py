"""The best entries of each score row: the Triton source of the
``index_select`` operator.

One program per row keeps the ``kept`` largest of the row's pooled scores and
writes the entry each names, ``-1`` for one whose score is ``-inf``:

- ``POOL`` is one for a selection of columns, or the columns one block ranks
  by its largest, the block holding the row's newest visible column
  (``(visible - 1) // POOL``) pinned at ``+inf``;
- with ``CANDIDATES`` (the scores' columns are groups of candidate blocks of
  that many groups), a kept column ``c`` names group
  ``candidates[t, c // CANDIDATES] * CANDIDATES + c % CANDIDATES``.

A row of the request's own groups ranks only the blocks its ``visible``
groups lie in: the scores past them are ``-inf`` (``index_scores``), so a
row's passes read what its request holds rather than the whole table, and the
slots past the ranked blocks are ``-1``. A row of candidate columns ranks
every column.

The scores are FP32 or BF16; a BF16 score is read as its FP32 value, which
keeps its order, so both are ranked by the same keys.

The selection is ``top_k``'s radix select without its ordering: four passes
of eight bits over the row's monotone 32-bit keys, most significant first,
each counting the keys of the prefix fixed so far by their next byte and
descending into the byte where the ``kept``-th largest lies, then one
compaction pass in column order that writes the keys above the selected value
into the low slots and the equal ones after them until the quota is full. So
the kept entries leave as the operator states: the strictly greater ones by
column, then the ties by column, a tie keeping the lower column. The row is
walked in chunks of ``TILE`` scores, so a row of any length up to ``2^31``
columns is one program's, and nothing is kept but the output.
"""

import triton
import triton.language as tl

#: The bits one selection pass fixes, and its bins: a byte of a key, and the
#: extra bin holds the lanes the pass does not match.
SELECT_RADIX = tl.constexpr(8)
SELECT_BINS = tl.constexpr(256)


@triton.jit
def _monotone(values):
    """Map FP32 values to unsigned keys of the same order."""
    bits = values.to(tl.uint32, bitcast=True)
    negative = (bits >> 31) == 1
    # Negatives invert and positives flip their sign bit: the standard order
    # preserving map of IEEE floats onto unsigned integers. `^ 0xFFFFFFFF`
    # rather than `~`, which the interpreter refuses on an unsigned value.
    return tl.where(negative, bits ^ 0xFFFFFFFF, bits | 0x80000000)


@triton.jit
def _pooled(scores, base, lanes, columns, pin, POOL: tl.constexpr):
    """The pooled scores of block ids ``lanes`` of one row, FP32: a block's
    largest column (``-inf`` past the row), the pinned block ``+inf``.
    """  # noqa: D205
    if POOL == 1:
        value = tl.load(scores + base + lanes, mask=lanes < columns, other=float("-inf"))
        value = value.to(tl.float32)
    else:
        column = lanes[:, None] * POOL + tl.arange(0, POOL)[None, :]
        block = tl.load(scores + base + column, mask=column < columns, other=float("-inf"))
        value = tl.max(block.to(tl.float32), axis=1)
        value = tl.where(lanes == pin, float("inf"), value)
    return value


@triton.jit(do_not_specialize=["columns", "kept"])
def _index_select_kernel(
    scores,
    visible,
    candidates,
    out,
    columns,
    kept,
    POOL: tl.constexpr,
    CANDIDATES: tl.constexpr,
    LANES: tl.constexpr,
):
    """Keep the ``kept`` best pooled scores of one row. Grid: (rows,).

    ``scores`` is ``[rows, columns]`` FP32 or BF16, ``visible`` ``[rows]`` U32,
    ``candidates`` ``[rows, columns / CANDIDATES]`` I32 (read when
    ``CANDIDATES`` is not zero) and ``out`` ``[rows, kept]`` I32. A chunk is
    ``LANES`` pooled entries: ``TILE / POOL`` blocks, so one pass reads
    ``TILE`` scores whatever the pool.
    """
    row = tl.program_id(0).to(tl.int64)
    columns = columns.to(tl.int64)
    base = row * columns
    blocks = (columns + POOL - 1) // POOL
    lanes_of = tl.arange(0, LANES)

    seen = tl.load(visible + row).to(tl.int64)
    # Only the block mode pins: a row that sees nothing pins nothing.
    pin = tl.where(seen > 0, (seen - 1) // POOL, -1)
    # The blocks the row ranks: those of its visible groups, or every
    # candidate column. Below, a lane past them is not live.
    if CANDIDATES == 0:
        blocks = tl.minimum(blocks, (seen + POOL - 1) // POOL)

    # ---- The selection: four passes of eight bits, most significant first.
    # After each pass the top bytes of `prefix` are fixed; `taken` is the
    # number of slots still to fill below the keys already counted, and
    # `above` how many keys are strictly greater than the prefix. ----
    taken = kept.to(tl.int32)
    prefix = tl.zeros((), tl.uint32)
    above = 0
    for radix in tl.static_range(4):
        shift = 32 - SELECT_RADIX * (radix + 1)
        # Bin `SELECT_BINS` holds the lanes the pass does not match, so the
        # counts below are taken over the matched group alone.
        histogram = tl.zeros((2 * SELECT_BINS,), tl.int32)
        matched = 0
        for start in range(0, blocks, LANES):
            lanes = start + lanes_of
            live = lanes < blocks
            keys = _monotone(_pooled(scores, base, lanes, columns, pin, POOL))
            if radix == 0:
                match = live
            else:
                match = (
                    (keys >> (shift + SELECT_RADIX)) == (prefix >> (shift + SELECT_RADIX))
                ) & live
            buckets = tl.where(match, (keys >> shift) & (SELECT_BINS - 1), SELECT_BINS)
            histogram += tl.histogram(buckets, 2 * SELECT_BINS)
            matched += tl.sum(match.to(tl.int32), axis=0)

        # The count of keys strictly above each byte, descending from the
        # largest; the crossing byte is the first one the quota reaches.
        cumulative = tl.cumsum(histogram, axis=0)
        higher = matched - cumulative
        bins = tl.arange(0, 2 * SELECT_BINS)
        crossing = tl.min(
            tl.where((higher < taken) & (bins < SELECT_BINS), bins, SELECT_BINS), axis=0
        )
        counted = tl.sum(tl.where(bins <= crossing, histogram, 0), axis=0)
        above += matched - counted
        taken -= matched - counted
        prefix |= crossing.to(tl.uint32) << shift

    # ---- Compaction, in column order: the keys above `prefix` into the low
    # slots, then the equal keys the quota takes. Every slot below `kept` is
    # written exactly once. ----
    destination = out + row * kept
    filled = 0
    equal_before = 0
    for start in range(0, blocks, LANES):
        lanes = start + lanes_of
        live = lanes < blocks
        value = _pooled(scores, base, lanes, columns, pin, POOL)
        keys = _monotone(value)
        greater = (keys > prefix) & live
        equal = (keys == prefix) & live

        # The entry a kept lane names: its block, or its candidate block's
        # group; nothing for a score of -inf or an unnamed candidate.
        if CANDIDATES > 0:
            block = tl.load(
                candidates + row * (columns // CANDIDATES) + lanes // CANDIDATES,
                mask=live,
                other=-1,
            )
            entry = block * CANDIDATES + (lanes % CANDIDATES).to(tl.int32)
            named = block >= 0
        else:
            entry = lanes.to(tl.int32)
            named = live
        entry = tl.where(named & (value > float("-inf")), entry, -1)

        slot_greater = filled + tl.cumsum(greater.to(tl.int32), axis=0) - 1
        keep_greater = greater & (slot_greater < above)
        tl.store(destination + slot_greater, entry, mask=keep_greater)
        # An equal key's slot counts the equal keys of the chunks before this
        # one, so the second chunk's ties follow the first chunk's.
        slot_equal = above + equal_before + tl.cumsum(equal.to(tl.int32), axis=0) - 1
        keep_equal = equal & (slot_equal < kept)
        tl.store(destination + slot_equal, entry, mask=keep_equal)
        filled += tl.sum(greater.to(tl.int32), axis=0)
        equal_before += tl.sum(equal.to(tl.int32), axis=0)

    # Fewer ranked blocks than kept slots leave the slots past them, which
    # name nothing.
    for start in range(blocks, kept, LANES):
        slots = start + lanes_of
        tl.store(destination + slots, -1, mask=slots < kept)


def derive(fields, parameters, target):
    """The pooled entries one chunk holds: ``tile`` scores, ``tile / pool``
    blocks, and the scores' type, FP32 unless the key states it.

    Raises:
        SystemExit: When the pool does not divide the tile into a power of
            two of blocks, or pools candidate scores.
    """  # noqa: D205
    tile, pool = fields["tile"], fields["pool"]
    if tile % pool or (tile // pool) & (tile // pool - 1):
        raise SystemExit(f"blocks of {pool} do not divide a tile of {tile}")
    if pool > 1 and fields["candidates"]:
        raise SystemExit("the block mode ranks unrestricted scores")
    return {"lanes": tile // pool, "scores": fields.get("scores", "f32")}
