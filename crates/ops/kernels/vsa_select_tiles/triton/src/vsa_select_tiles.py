"""VSA tile selection: the Triton source of the ``vsa_select_tiles`` operator.

One program chooses one query tile's key tiles for one head: it ranks a segment's
keys by descending score with ties in ascending tile order, keeps each segment's
leading ``segment_keep`` tiles besides every live dense tile, and stores the kept
tiles first in ascending order followed by the others. A segment's threshold is
found by bisection over the 32-bit order key rather than by sorting.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/attention/vsa_segments.py``), unchanged
except that it imports nothing but Triton: the launch-time wrapper, its operand
checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["query_offset"])
def _select_tiles_kernel(
    scores,
    tile_segments,
    variable_block_sizes,
    segment_keep,
    indices,
    counts,
    query_tiles,
    tiles,
    query_offset,
    block: tl.constexpr,
):
    """Choose one query tile's key tiles for one head.

    Query ``q`` of the program grid is tile ``query_offset + q`` of the sequence;
    the tables hold every tile. ``block`` is at least the tile count, so every key
    tile has a lane.
    """
    query = tl.program_id(0)
    head = tl.program_id(1)
    row = (head * query_tiles + query).to(tl.int64)
    keys = tl.arange(0, block)
    present = keys < tiles
    segments = tl.load(tile_segments + keys, mask=present, other=-1)
    live = tl.load(variable_block_sizes + keys, mask=present, other=0) > 0
    query_segment = tl.load(tile_segments + query_offset + query)
    query_live = tl.load(variable_block_sizes + query_offset + query) > 0

    mask = live & query_live
    if query_live & (query_segment >= 0):
        values = tl.load(
            scores + row * tiles + keys, mask=present, other=0.0
        )
        # Ascending order key of the FP32 score: the sign-flipped bits, with -0
        # as +0 and every NaN above +inf, as the stable sort orders them.
        bits = (values + 0.0).to(tl.int32, bitcast=True).to(tl.int64)
        bits = bits & 0xFFFFFFFF
        ascending = tl.where(
            bits >= 0x80000000, 0xFFFFFFFF - bits, bits | 0x80000000
        )
        ascending = tl.where(values != values, 0xFFFFFFFF, ascending)

        # The same order on signed 32-bit keys, which halve the lanes' registers:
        # every key and threshold shifts down by 2**31.
        signed = (ascending - 0x80000000).to(tl.int32)

        chosen = present & (segments < -1)
        for segment in range(tl.max(segments, axis=0) + 1):
            members = present & (segments == segment)
            keep = tl.minimum(
                tl.load(segment_keep + segment),
                tl.sum(members.to(tl.int32), axis=0),
            )
            # Invariant: at least ``keep`` member keys reach ``lower`` and fewer
            # reach ``upper``; 32 halvings leave adjacent bounds.
            lower = tl.zeros((), dtype=tl.int64)
            upper = tl.full((), 1 << 32, dtype=tl.int64)
            for _ in tl.static_range(32):
                middle = lower + (upper - lower) // 2
                bound = (middle - 0x80000000).to(tl.int32)
                reach = tl.sum(
                    (members & (signed >= bound)).to(tl.int32), axis=0
                )
                lower = tl.where(reach >= keep, middle, lower)
                upper = tl.where(reach >= keep, upper, middle)
            threshold = (lower - 0x80000000).to(tl.int32)
            above = members & (signed > threshold)
            tied = members & (signed == threshold)
            room = keep - tl.sum(above.to(tl.int32), axis=0)
            tied_before = tl.cumsum(tied.to(tl.int32), axis=0) - 1
            chosen = chosen | above | (tied & (tied_before < room))
        mask = (live & (segments < 0)) | chosen
    mask = mask & present

    count = tl.sum(mask.to(tl.int32), axis=0)
    kept_before = tl.cumsum(mask.to(tl.int32), axis=0) - 1
    others_before = tl.cumsum((present & ~mask).to(tl.int32), axis=0) - 1
    slots = tl.where(mask, kept_before, count + others_before)
    tl.store(indices + row * tiles + slots, keys, mask=present)
    tl.store(counts + row, count)
