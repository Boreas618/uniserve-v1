"""The ``width`` largest logits of a row: the Triton source of the ``top_k``
operator.

One program per row leaves with the row's ``width`` largest logits in descending
order and the vocabulary id of each, ties keeping the lower id first. Nothing the
program is compiled to depends on the candidate count or on the vocabulary: the
row is walked in ``TILE``-lane chunks, both counts are launch scalars, and the
kept pairs are ordered by the passes rather than by a sort network of the kept
count padded to a power of two. One module therefore serves any kept count up to
the row's width.

1. eight radix passes over the row's 32-bit monotone keys, four bits at a time
   from the most significant end, counting how many keys share each nibble and
   descending into the nibble where the width-th largest lies;
2. one pass that compacts the kept keys and their ids into the scratch, in
   column order: the keys above the selected value into the low slots, then the
   equal ones until the quota is full;
3. eight stable passes over the scratch, least significant nibble first, each
   placing every entry at the slot the descending order needs - the compaction
   is in column order, so ties are in ascending id order before the passes and
   stability keeps them so, which is why the id needs no pass of its own;
4. one pass that writes the values back out of the ordered keys and the ids from
   their plane.

The selection reads the row five times; ordering the kept pairs reads and writes
the scratch eight times, and never the row.
"""

import triton
import triton.language as tl

#: The bits of a key one ordering pass fixes.
RADIX = tl.constexpr(4)
#: The bits one selection pass fixes.
SELECT_RADIX = tl.constexpr(8)
#: The bins of one ordering pass, and of the base table its program keeps.
BINS = tl.constexpr(16)
#: The bins of one selection pass: a byte of a key, and the extra bin holds the lanes the pass does
#: not match.
SELECT_BINS = tl.constexpr(256)


@triton.jit
def _monotone(values):
    """Map a row of FP32 values to unsigned keys with the same order."""
    bits = values.to(tl.uint32, bitcast=True)
    negative = (bits >> 31) == 1
    # Negatives invert and positives flip their sign bit, which is the standard
    # order-preserving map of IEEE floats onto unsigned integers.
    # `bits ^ 0xFFFFFFFF` rather than `~bits`: the interpreter `~` turns a uint32 into the Python
    # integer -1 and refuses it, and the XOR is the same value on both.
    return tl.where(negative, bits ^ 0xFFFFFFFF, bits | 0x80000000)


@triton.jit
def _value_of(keys):
    """Map order-preserving unsigned keys back to FP32 values."""
    sign = (keys >> 31) == 1
    bits = tl.where(sign, keys ^ 0x80000000, keys ^ 0xFFFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["rows", "vocabulary", "width"])
def _top_k_kernel(
    logits_ptr,
    values_ptr,
    ids_ptr,
    workspace_ptr,
    rows,
    vocabulary,
    width,
    TILE: tl.constexpr,
):
    """Keep the ``width`` largest logits of one row, sorted, with their ids.

    ``logits`` is ``[rows, vocabulary]``, ``values`` is ``[rows, width]`` f32, ``ids`` is
    ``[rows, width]`` in the descriptor's id dtype, and ``workspace`` is the row's scratch:
    ``[width]`` keys, ``[width]`` ids, ``[width]`` keys, ``[width]`` ids and then ``2 * BINS`` i32 -
    the descending base of each nibble and the count of it already placed. Grid: (rows,).

    ``TILE`` is the only size compiled in: the row is walked in ``TILE``-lane chunks, and every
    buffer is addressed with the live width, so the module serves any vocabulary and any kept count.
    """
    row = tl.program_id(0).to(tl.int64)
    base = row * vocabulary
    # `taken` is the quota the selected prefix's equal keys still hold, and it starts as the width;
    # `live_width` is the kept count, which is also the buffers' row stride.
    taken = width.to(tl.int32)
    live_width = taken
    stride = width.to(tl.int64)
    columns = tl.arange(0, TILE)

    # ---- The selection: four passes of eight bits over the row, most significant first. ----
    # After each pass the top byte of `prefix` is fixed and `taken` is the number of slots still to
    # fill below the keys already counted, `above` how many keys are strictly above `prefix`.
    prefix = tl.zeros((), tl.uint32)
    above = 0
    for radix in tl.static_range(4):
        shift = 32 - SELECT_RADIX * (radix + 1)
        # `2 * SELECT_BINS` bins, bin `SELECT_BINS` holding the lanes this pass does not match: a bin
        # is the only mask a histogram has, so the unmatched lanes are folded into one of their own
        # rather than dropped - dropped, the counts below are taken over the group and the numbers
        # they feed (`matched - counted`) are taken over the row.
        histogram = tl.zeros((2 * SELECT_BINS,), tl.int32)
        matched = 0
        for start in range(0, vocabulary, TILE):
            lanes = start + columns
            live = lanes < vocabulary
            values = tl.load(logits_ptr + base + lanes, mask=live, other=0.0).to(tl.float32)
            keys = _monotone(values)
            if radix == 0:
                match = live
            else:
                match = ((keys >> (shift + SELECT_RADIX)) == (prefix >> (shift + SELECT_RADIX))) & live
            buckets = tl.where(match, (keys >> shift) & (SELECT_BINS - 1), SELECT_BINS)
            histogram += tl.histogram(buckets, 2 * SELECT_BINS)
            matched += tl.sum(match.to(tl.int32), axis=0)

        # The count of keys strictly above each bucket, descending from the largest; the crossing
        # bucket is the first one the width reaches.
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

    # ---- The scratch's planes, for this row. ----
    span = 4 * width + 2 * BINS
    row_span = row * span
    keys_in = workspace_ptr + row_span
    keys_out = keys_in + width
    ids_in = keys_out + width
    ids_out = ids_in + width
    bases = ids_out + width
    running = bases + BINS
    bins = tl.arange(0, 2 * BINS)

    # ---- Compaction, in column order: the keys above `prefix`, then the equal keys the quota
    # takes. Every slot below `live_width` is written exactly once, and the equal group's own order
    # is ascending id because the columns are. ----
    filled = 0
    equal_before = 0
    for start in range(0, vocabulary, TILE):
        lanes = start + columns
        live = lanes < vocabulary
        values = tl.load(logits_ptr + base + lanes, mask=live, other=0.0).to(tl.float32)
        keys = _monotone(values)
        gt = (keys > prefix) & live
        eq = (keys == prefix) & live
        slot_gt = filled + tl.cumsum(gt.to(tl.int32), axis=0) - 1
        keep_gt = gt & (slot_gt < above)
        tl.store(keys_in + slot_gt, keys.to(tl.int32, bitcast=True), mask=keep_gt)
        tl.store(ids_in + slot_gt, lanes.to(tl.int32), mask=keep_gt)
        # The equal keys' slots need the count from the tiles before this one: a rank computed
        # inside a tile alone restarts at `above` in every tile, so the second tile's equals
        # overwrite the first tile's and the slots their ranks should have filled are never written.
        slot_eq = above + equal_before + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        keep_eq = eq & (slot_eq < live_width)
        tl.store(keys_in + slot_eq, keys.to(tl.int32, bitcast=True), mask=keep_eq)
        tl.store(ids_in + slot_eq, lanes.to(tl.int32), mask=keep_eq)
        filled += tl.sum(gt.to(tl.int32), axis=0)
        equal_before += tl.sum(eq.to(tl.int32), axis=0)

    # ---- The order: eight stable passes, least significant nibble first, each placing every entry
    # where the descending order needs it. A pass is a histogram of the nibble over the whole kept
    # set, then a scatter into the other plane at `descending base + the count of that nibble already
    # placed`. Stability is what keeps ties in ascending id order, and there are no ties left over
    # for the id to break once the key is ordered. ----
    src_keys, src_ids, dst_keys, dst_ids = keys_in, ids_in, keys_out, ids_out
    for radix in tl.static_range(8):
        shift = RADIX * radix
        histogram = tl.zeros((2 * BINS,), tl.int32)
        for start in range(0, width, TILE):
            lanes = start + columns
            live = lanes < width
            keys = tl.load(src_keys + lanes, mask=live, other=0).to(tl.uint32, bitcast=True)
            digit = ((keys >> shift) & (BINS - 1)).to(tl.int32)
            histogram += tl.histogram(tl.where(live, digit, BINS), 2 * BINS)
        counts = tl.where(bins < BINS, histogram, 0)
        total = tl.sum(counts, axis=0)
        # Nibble `d`'s slot is the count of entries whose nibble is greater: descending order.
        base_of = total - tl.cumsum(counts, axis=0)
        tl.store(bases + bins, base_of, mask=bins < BINS)
        tl.store(running + bins, tl.zeros((2 * BINS,), tl.int32), mask=bins < BINS)

        for start in range(0, width, TILE):
            lanes = start + columns
            live = lanes < width
            keys = tl.load(src_keys + lanes, mask=live, other=0).to(tl.uint32, bitcast=True)
            identities = tl.load(src_ids + lanes, mask=live, other=0)
            digit = ((keys >> shift) & (BINS - 1)).to(tl.int32)
            rank = tl.zeros((TILE,), tl.int32)
            for which in tl.static_range(BINS):
                same = live & (digit == which)
                rank += tl.where(same, tl.cumsum(same.to(tl.int32), axis=0) - 1, 0)
            slot = (
                tl.load(bases + digit, mask=live, other=0)
                + tl.load(running + digit, mask=live, other=0)
                + rank
            )
            tl.store(dst_keys + slot, keys.to(tl.int32, bitcast=True), mask=live)
            tl.store(dst_ids + slot, identities, mask=live)
            tl.store(
                running + bins,
                tl.load(running + bins, mask=bins < BINS, other=0)
                + tl.where(bins < BINS, tl.histogram(tl.where(live, digit, BINS), 2 * BINS), 0),
                mask=bins < BINS,
            )

        src_keys, src_ids, dst_keys, dst_ids = dst_keys, dst_ids, src_keys, src_ids

    # ---- The write-out: the values come back out of the ordered keys. ----
    for start in range(0, width, TILE):
        lanes = start + columns
        live = lanes < width
        keys = tl.load(src_keys + lanes, mask=live, other=0).to(tl.uint32, bitcast=True)
        tl.store(values_ptr + row * stride + lanes, _value_of(keys), mask=live)
        tl.store(
            ids_ptr + row * stride + lanes,
            tl.load(src_ids + lanes, mask=live, other=0),
            mask=live,
        )
