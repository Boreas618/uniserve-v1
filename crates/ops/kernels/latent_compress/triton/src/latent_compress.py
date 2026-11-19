"""The Triton source of the ``latent_compress`` operator.

One program handles one token (``ops::compress::LatentCompress``). A token at
position ``p`` completes its group when ``(p + 1) mod RATIO = 0``; it then
pools the group's ``RATIO`` projections, its own and its predecessors', per
channel:

    pooled[c] = Σ_m softmax_m(score[m, c]) · kv[m, c]        FP32
    latent    = rms_norm(bf16(pooled), gain, eps)            rounded to BF16

``projected`` holds each token's ``[2 · WIDTH]`` FP32 projection, its values
and then its gates. A predecessor at position ``q`` is the batch row ``t − (p
− q)`` when the token's state row holds it in this batch (its positions are
consecutive and end at ``seq_lens − 1``), and otherwise row ``q mod RING`` of
the request's ring, which the program only reads. The token's state row is
found by a binary search of the packed query offsets.

Besides the latent, every token answers the entry it completes (``p /
RATIO``, or ``-1``) and the ring row its projection is kept in for a later
step: ``slot · RING + p mod RING`` for the state row's last ``RING`` tokens,
``-1`` otherwise and for a token of no state row or a padding row's. The
launch table compiles without contracting products into sums, so the pooling
rounds at every operation as its formula is written.

The kernel body imports nothing but Triton: the shape checks and the
allocation stay with the operator's Rust side.
"""

import triton
import triton.language as tl

#: The bisection steps that find a token's state row among up to `2^31` rows.
SEARCH_STEPS = tl.constexpr(32)


@triton.jit
def _state_row(query_offsets, rows, token):
    """The state row whose packed tokens hold ``token``: the last row whose
    offset is at most ``token`` (an empty row before it shares its offset).
    """  # noqa: D205
    low = tl.zeros((), tl.int32)
    high = rows.to(tl.int32)
    for _ in tl.static_range(SEARCH_STEPS):
        middle = (low + high) // 2
        searching = high - low > 1
        offset = tl.load(query_offsets + middle, mask=searching, other=0)
        below = offset <= token
        low = tl.where(searching & below, middle, low)
        high = tl.where(searching & ~below, middle, high)
    return low


@triton.jit(do_not_specialize=["tokens", "rows", "projected_stride"])
def _latent_compress_kernel(
    projected,
    weight,
    ring,
    slots,
    query_offsets,
    seq_lens,
    positions,
    latent,
    entries,
    ring_rows,
    tokens,
    rows,
    projected_stride,
    eps,
    RATIO: tl.constexpr,
    WIDTH: tl.constexpr,
    RING: tl.constexpr,
):
    """Pool the group token ``program_id(0)`` completes. Grid: ``(tokens,)``.

    ``ring`` is ``[slots, RING, 2 · WIDTH]`` FP32; ``slots``, ``query_offsets``
    and ``seq_lens`` are the state rows' columns, ``positions`` the tokens'.
    ``latent`` receives a completing token's ``[WIDTH]`` BF16 row, ``entries``
    and ``ring_rows`` every token's I32 answer.
    """
    token = tl.program_id(0)
    row = _state_row(query_offsets, rows, token)
    first = tl.load(query_offsets + row)
    end = tl.load(query_offsets + row + 1)
    slot = tl.load(slots + row)

    # A token past the last state row's tokens, or of a padding row, keeps
    # nothing and completes nothing.
    if (token >= end) | (slot < 0) | (rows == 0):
        tl.store(entries + token, -1)
        tl.store(ring_rows + token, -1)
        return

    position = tl.load(positions + token)
    seen = tl.load(seq_lens + row)
    first_position = seen - (end - first)
    kept = tl.where(position >= seen - RING, slot * RING + position % RING, -1)
    tl.store(ring_rows + token, kept)

    if (position + 1) % RATIO != 0:
        tl.store(entries + token, -1)
        return
    tl.store(entries + token, position // RATIO)

    channels = tl.arange(0, WIDTH)
    group = position - (RATIO - 1)

    # The gates' maximum over the group, per channel, and then the softmax's
    # terms; each member is read from the batch when this batch holds it and
    # from the ring otherwise.
    peak = tl.full((WIDTH,), float("-inf"), tl.float32)
    for member in tl.static_range(RATIO):
        q = group + member
        batched = q >= first_position
        at_batch = (
            projected + (token - (position - q)).to(tl.int64) * projected_stride
        )
        at_ring = ring + (slot * RING + q % RING).to(tl.int64) * (2 * WIDTH)
        score = tl.where(
            batched,
            tl.load(at_batch + WIDTH + channels, mask=batched, other=0.0),
            tl.load(at_ring + WIDTH + channels, mask=~batched, other=0.0),
        )
        peak = tl.maximum(peak, score)

    total = tl.zeros((WIDTH,), tl.float32)
    for member in tl.static_range(RATIO):
        q = group + member
        batched = q >= first_position
        at_batch = (
            projected + (token - (position - q)).to(tl.int64) * projected_stride
        )
        at_ring = ring + (slot * RING + q % RING).to(tl.int64) * (2 * WIDTH)
        score = tl.where(
            batched,
            tl.load(at_batch + WIDTH + channels, mask=batched, other=0.0),
            tl.load(at_ring + WIDTH + channels, mask=~batched, other=0.0),
        )
        total += tl.exp(score - peak)

    pooled = tl.zeros((WIDTH,), tl.float32)
    for member in tl.static_range(RATIO):
        q = group + member
        batched = q >= first_position
        at_batch = (
            projected + (token - (position - q)).to(tl.int64) * projected_stride
        )
        at_ring = ring + (slot * RING + q % RING).to(tl.int64) * (2 * WIDTH)
        value = tl.where(
            batched,
            tl.load(at_batch + channels, mask=batched, other=0.0),
            tl.load(at_ring + channels, mask=~batched, other=0.0),
        )
        score = tl.where(
            batched,
            tl.load(at_batch + WIDTH + channels, mask=batched, other=0.0),
            tl.load(at_ring + WIDTH + channels, mask=~batched, other=0.0),
        )
        pooled += value * (tl.exp(score - peak) / total)

    # The pooled row rounds into BF16 before its normalization, which rounds
    # once more as it stores.
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    gain = tl.load(weight + channels).to(tl.float32)
    variance = tl.sum(pooled * pooled, axis=0) / WIDTH
    normalized = pooled * tl.rsqrt(variance + eps) * gain
    tl.store(
        latent + token.to(tl.int64) * WIDTH + channels,
        normalized.to(tl.bfloat16),
    )
