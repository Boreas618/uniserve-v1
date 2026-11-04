"""The ``width`` largest logits of a row: the Triton source of the ``top_k``
operator.

One program per row leaves with the row's ``width`` largest logits in descending
order and the vocabulary id of each, ties keeping the lower id first. It finds
the width-th largest value exactly, without materializing a sort of the
vocabulary:

1. four radix passes over the row's 32-bit monotone keys, eight bits at a time
   from the most significant end, counting how many keys share each byte and
   descending into the byte where the width-th largest lies;
2. one pass that emits every key above that value into the row's slots, then the
   equal ones in column order until the width is full;
3. one sort of the emitted ``width`` pairs, which is what makes the result
   descending with the lower id first among equals.

The selection is exact for any integer width and reads the row five times.
"""

import triton
import triton.language as tl

#: The bits of a key one radix pass fixes.
RADIX = tl.constexpr(8)
#: The bins of one radix pass.
BINS = tl.constexpr(1 << 8)


@triton.jit
def _monotone(values):
    """Map a row of FP32 values to unsigned keys with the same order."""
    bits = values.to(tl.uint32, bitcast=True)
    negative = (bits >> 31) == 1
    # Negatives invert and positives flip their sign bit, which is the standard
    # order-preserving map of IEEE floats onto unsigned integers.
    return tl.where(negative, ~bits, bits | 0x80000000)


@triton.jit
def _value_of(keys):
    """Map order-preserving unsigned keys back to FP32 values."""
    sign = (keys >> 31) == 1
    bits = tl.where(sign, keys ^ 0x80000000, ~keys)
    return bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["rows", "vocabulary", "width"])
def _top_k_kernel(
    logits_ptr,
    values_ptr,
    ids_ptr,
    rows,
    vocabulary,
    width,
    WIDTH: tl.constexpr,
    STEP: tl.constexpr,
):
    """Keep the ``WIDTH`` largest logits of one row, sorted, with their ids.

    ``logits`` is ``[rows, vocabulary]``, ``values`` ``[rows, WIDTH]`` and ``ids``
    ``[rows, WIDTH]`` in the descriptor's id dtype. Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    base = row * vocabulary
    taken = width.to(tl.int32)
    columns = tl.arange(0, STEP)

    # Four radix passes: after each one the top byte of `prefix` is fixed and
    # `taken` is the number of slots still to fill below the keys already
    # counted, `above` how many keys are strictly above `prefix`.
    prefix = tl.zeros((), tl.uint32)
    above = 0
    for radix in tl.static_range(32 // RADIX):
        shift = 32 - RADIX * (radix + 1)
        histogram = tl.zeros((BINS,), tl.int32)
        matched = 0
        for start in range(0, vocabulary, STEP):
            lanes = start + columns
            live = lanes < vocabulary
            values = tl.load(logits_ptr + base + lanes, mask=live, other=0.0).to(tl.float32)
            keys = _monotone(values)
            if radix == 0:
                match = live
            else:
                match = ((keys >> (shift + RADIX)) == (prefix >> (shift + RADIX))) & live
            buckets = (keys >> shift) & (BINS - 1)
            histogram += tl.histogram(buckets, BINS, mask=match)
            matched += tl.sum(match.to(tl.int32), axis=0)

        # The count of keys strictly above each bucket, descending from the
        # largest; the crossing bucket is the first one the width reaches.
        cumulative = tl.cumsum(histogram, axis=0)
        higher = matched - cumulative
        bins = tl.arange(0, BINS)
        crossing = tl.min(tl.where(higher < taken, bins, BINS), axis=0)
        counted = tl.sum(tl.where(bins <= crossing, histogram, 0), axis=0)
        above += matched - counted
        taken -= matched - counted
        prefix |= crossing.to(tl.uint32) << shift

    # Emit: the keys above `prefix` fill the low slots, then the equal keys in
    # column order until the width is full.
    filled = 0
    equal_count = 0
    for start in range(0, vocabulary, STEP):
        lanes = start + columns
        live = lanes < vocabulary
        values = tl.load(logits_ptr + base + lanes, mask=live, other=0.0).to(tl.float32)
        keys = _monotone(values)
        gt = (keys > prefix) & live
        eq = (keys == prefix) & live
        slot_gt = filled + tl.cumsum(gt.to(tl.int32), axis=0) - 1
        keep_gt = gt & (slot_gt < WIDTH)
        tl.store(values_ptr + row * WIDTH + slot_gt, values, mask=keep_gt)
        tl.store(ids_ptr + row * WIDTH + slot_gt, lanes.to(tl.int32), mask=keep_gt)
        slot_eq = above + equal_count + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        keep_eq = eq & (slot_eq < WIDTH)
        tl.store(values_ptr + row * WIDTH + slot_eq, values, mask=keep_eq)
        tl.store(ids_ptr + row * WIDTH + slot_eq, lanes.to(tl.int32), mask=keep_eq)
        filled += tl.sum(gt.to(tl.int32), axis=0)
        equal_count += tl.sum(eq.to(tl.int32), axis=0)

    # Slots the row did not fill keep a sentinel that sorts last.
    tl.debug_barrier()
    slots = tl.arange(0, WIDTH)
    empty = slots >= tl.minimum(tl.maximum(filled, above + equal_count), WIDTH)
    tl.store(values_ptr + row * WIDTH + slots, -float("inf"), mask=empty)
    tl.store(ids_ptr + row * WIDTH + slots, 0, mask=empty)

    # Sort the emitted pairs: the key above the id makes a descending integer
    # sort descending in value, and the id's complement ascending among equals.
    tl.debug_barrier()
    pairs = tl.load(values_ptr + row * WIDTH + slots)
    identities = tl.load(ids_ptr + row * WIDTH + slots).to(tl.int64)
    packed = (_monotone(pairs).to(tl.int64) << 32) | (0xFFFFFFFF - identities)
    ordered = tl.sort(packed, dim=0, descending=True)
    keys = (ordered >> 32).to(tl.uint32)
    tl.store(values_ptr + row * WIDTH + slots, _value_of(keys))
    tl.store(
        ids_ptr + row * WIDTH + slots,
        (0xFFFFFFFF - (ordered & 0xFFFFFFFF)).to(tl.int32),
    )
