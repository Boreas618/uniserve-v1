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
    rows,
    vocabulary,
    width,
    WIDTH: tl.constexpr,
    STEP: tl.constexpr,
):
    """Keep the ``width`` largest logits of one row, sorted, with their ids.

    ``logits`` is ``[rows, vocabulary]``, ``values`` and ``ids`` are ``[rows, width]`` in the
    descriptor's id dtype. Grid: (rows,).

    ``WIDTH`` is the **padded** candidate count - the power of two the final sort runs over - and
    ``width`` is the live one: the row's stride in the buffers is ``width``, and every emitted lane
    at or past it is masked out. A sort needs a power-of-two network; a deployment's candidate count
    (a vocabulary, 151936) is not one, so the network is widened and the buffers are not.
    """
    row = tl.program_id(0).to(tl.int64)
    base = row * vocabulary
    # `taken` is the quota the prefix's equal keys still hold, and it starts as the width; `live` is
    # the width itself, and every bound on a slot index is this one - a slot bound read from the
    # quota stopped at the last slot the equal keys filled and left the rest of the row stale.
    taken = width.to(tl.int32)
    live_width = taken
    # The buffers' row stride is the live candidate count, not the sort's padded width.
    stride = width.to(tl.int64)
    columns = tl.arange(0, STEP)

    # Four radix passes: after each one the top byte of `prefix` is fixed and
    # `taken` is the number of slots still to fill below the keys already
    # counted, `above` how many keys are strictly above `prefix`.
    prefix = tl.zeros((), tl.uint32)
    above = 0
    for radix in tl.static_range(32 // RADIX):
        shift = 32 - RADIX * (radix + 1)
        # `2 * BINS` bins, bin `BINS` holding the lanes this pass does not match: a bin is the only
        # mask a histogram has, so the unmatched lanes are folded into one of their own rather than
        # dropped - dropped, the counts below are taken over the group and the numbers they feed
        # (`matched - counted`) are taken over the row. The bin sits above every real bucket, so
        # `higher` is exact for the bins a crossing is chosen from; the shape is a power of two
        # because a tensor's is.
        histogram = tl.zeros((2 * BINS,), tl.int32)
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
            buckets = tl.where(match, (keys >> shift) & (BINS - 1), BINS)
            histogram += tl.histogram(buckets, 2 * BINS)
            matched += tl.sum(match.to(tl.int32), axis=0)

        # The count of keys strictly above each bucket, descending from the
        # largest; the crossing bucket is the first one the width reaches.
        cumulative = tl.cumsum(histogram, axis=0)
        higher = matched - cumulative
        bins = tl.arange(0, 2 * BINS)
        crossing = tl.min(tl.where((higher < taken) & (bins < BINS), bins, BINS), axis=0)
        counted = tl.sum(tl.where(bins <= crossing, histogram, 0), axis=0)
        above += matched - counted
        taken -= matched - counted
        prefix |= crossing.to(tl.uint32) << shift

    # Emit: the keys above `prefix` fill the low slots, then the equal keys in
    # column order until the width is full.
    filled = 0
    # The equal keys' slots need the count from the tiles before this one: a rank computed inside a
    # tile alone restarts at `above` in every tile, so the second tile's equals overwrite the first
    # tile's and the slots their ranks should have filled are left holding whatever the buffer had.
    equal_before = 0
    for start in range(0, vocabulary, STEP):
        lanes = start + columns
        live = lanes < vocabulary
        values = tl.load(logits_ptr + base + lanes, mask=live, other=0.0).to(tl.float32)
        keys = _monotone(values)
        gt = (keys > prefix) & live
        eq = (keys == prefix) & live
        slot_gt = filled + tl.cumsum(gt.to(tl.int32), axis=0) - 1
        # The keys strictly above the selected value fill the low slots, and their count is
        # `above`, not `taken`: `taken` is the quota the *equal* keys still hold (`above + taken`
        # is the candidate width the whole way down), so bounding the greater keys by it dropped
        # every one of them past the quota and left the row with a single slot written.
        keep_gt = gt & (slot_gt < above)
        tl.store(values_ptr + row * stride + slot_gt, values, mask=keep_gt)
        tl.store(ids_ptr + row * stride + slot_gt, lanes.to(tl.int32), mask=keep_gt)
        slot_eq = above + equal_before + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        keep_eq = eq & (slot_eq < above + taken)
        tl.store(values_ptr + row * stride + slot_eq, values, mask=keep_eq)
        tl.store(ids_ptr + row * stride + slot_eq, lanes.to(tl.int32), mask=keep_eq)
        filled += tl.sum(gt.to(tl.int32), axis=0)
        equal_before += tl.sum(eq.to(tl.int32), axis=0)

    # Slots the row did not fill keep a sentinel that sorts last: the lanes it did not fill inside
    # the live width, and - through the load's `other` below - the padding lanes of the network.
    #
    # The written slots are `[0, above)` and then the equal keys the quota still holds, so the count
    # is `above + min(equal, taken)` - never `filled + equal_before`, which counts the equal keys the
    # quota refuses too and so would leave slots below `taken` holding whatever the buffer had, and a
    # stale positive key sorts above every real candidate.
    tl.debug_barrier()
    slots = tl.arange(0, WIDTH)
    written = tl.minimum(above + equal_before, live_width)
    empty = (slots >= written) & (slots < live_width)
    tl.store(values_ptr + row * stride + slots, -float("inf"), mask=empty)
    tl.store(ids_ptr + row * stride + slots, 0, mask=empty)

    # Sort the emitted pairs: the key above the id makes a descending integer
    # sort descending in value, and the id's complement ascending among equals. The network is
    # `WIDTH` lanes wide; the lanes past the live width read a sentinel that sorts last and are
    # never stored, so the padded network touches nothing outside the row.
    tl.debug_barrier()
    in_row = slots < live_width
    # The buffers carry the descriptor's dtype - bf16 and f16 are declared - so the pair is widened
    # before its bits are read: `_monotone` reads 32 bits of a float, and a 16-bit lane has none.
    pairs = tl.load(values_ptr + row * stride + slots, mask=in_row, other=-float("inf")).to(
        tl.float32
    )
    identities = tl.load(ids_ptr + row * stride + slots, mask=in_row, other=0).to(tl.int64)
    # The key is packed into the high half of an int64 with its top bit taken off the key and put
    # back on the signed side: a monotone key of a positive logit has its top bit set, and shifting
    # it into place without this wrap overflowed into the sign bit, so the row's real candidates
    # sorted below every sentinel.
    packed = (_monotone(pairs).to(tl.int64) << 31) | (0x7FFFFFFF - identities)
    ordered = tl.sort(packed, dim=0, descending=True)
    keys = (ordered >> 31).to(tl.uint32)
    tl.store(values_ptr + row * stride + slots, _value_of(keys), mask=in_row)
    tl.store(
        ids_ptr + row * stride + slots,
        (0x7FFFFFFF - (ordered & 0x7FFFFFFF)).to(tl.int32),
        mask=in_row,
    )
