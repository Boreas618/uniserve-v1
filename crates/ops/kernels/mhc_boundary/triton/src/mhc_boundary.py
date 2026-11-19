"""One boundary of an mHC residual: the Triton source of ``mhc_boundary``.

The residual of a token is ``STREAMS`` streams of ``HIDDEN`` elements. At the
boundary between two sub-layers the program, for a block of ``BLOCK_T``
tokens::

    h    = round(post_k y + sum_j comb[j, k] h_j)     when POST (pending y)
    r    = rsqrt(mean(flat(h)^2) + rms_eps)
    z    = (flat(h) @ fn^T) r s_g + b                  s_g: mix group g's scale
    pre  = sigmoid(z[:n]) + eps
    post = 2 sigmoid(z[n:2n])
    comb = sinkhorn(z[2n:].view(n, n))
    x    = rms_norm(round(sum_j carried_j h_j)) w      the input, read with
                                                       the carried pre-mix

``sinkhorn`` is a row softmax plus ``eps``, a division of each column by its
sum plus ``eps``, then ``ITERATIONS - 1`` alternating row and column
divisions. Every sum and product is FP32; the written streams and the
collapsed sum round once into the streams' type before anything reads them,
as the reference rounds them, and the normalized input rounds once more.

**The walk.** A program walks ``BLOCK_D`` columns of every stream at a time.
Each stream's ``[BLOCK_T, BLOCK_D]`` tile (written back first when ``POST``,
from the read streams' tiles, which stay in the L1 cache across the written
ones) is a ``tl.dot`` operand against the projection's ``[MIX_TILE,
BLOCK_D]`` slab of the same columns, a three-pass TF32 product whose FP32
parts keep the projection's precision (the streams' 16-bit values are exact in
TF32); the same tile gives its squares and its term of the collapsed input,
whose columns are stored rounded with their squares summed.

**Splits.** With few tokens the token blocks alone do not fill the device, so
the grid's second axis splits each block's column steps across ``splits``
programs. Each stores its partial mixes and sums, ``[MIX_TILE + 2]`` FP32 per
token, and counts its arrival; the last to arrive (``acq_rel`` at GPU scope,
after a barrier that orders every thread's stores before it) sums the
partials in split order, so the result does not depend on the order of
arrival, sets the counter back to zero and finishes the block: the mixes, the
Sinkhorn iterations, and the normalization of the input columns every split
stored, read past the L1 cache, which does not see other programs' stores.

The kernel body imports nothing but Triton: the launch-time wrapper, its
operand checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _finish(
    acc,
    squares,
    collapsed,
    rows,
    live,
    scale_ptr,
    base_ptr,
    norm_ptr,
    input_ptr,
    next_pre_ptr,
    next_post_ptr,
    next_comb_ptr,
    rms_eps,
    eps,
    norm_eps,
    STREAMS: tl.constexpr,  # noqa: N803
    HIDDEN: tl.constexpr,  # noqa: N803
    ITERATIONS: tl.constexpr,  # noqa: N803
    BLOCK_T: tl.constexpr,  # noqa: N803
    NORM: tl.constexpr,  # noqa: N803
    MIXES: tl.constexpr,  # noqa: N803
    MIX_TILE: tl.constexpr,  # noqa: N803
):
    """The block's mixes from its summed products, and its normalized input.

    ``acc`` is ``[BLOCK_T, MIX_TILE]``, the products of every column;
    ``squares`` and ``collapsed`` the sums of the squares of the streams and
    of the collapsed input. The collapsed input's columns are in ``input_ptr``.
    """
    width: tl.constexpr = STREAMS * HIDDEN
    groups: tl.constexpr = MIX_TILE // STREAMS
    mixes = tl.arange(0, MIX_TILE)
    group = mixes // STREAMS

    # The mixes: the products scaled by the statistic of the flattened streams,
    # then by their group's scale (pre, post, then the combination logits), plus
    # their bias.
    inverse = tl.rsqrt(squares / width + rms_eps)
    scale = tl.load(scale_ptr + tl.minimum(group, 2))
    base = tl.load(base_ptr + mixes, mask=mixes < MIXES, other=0.0)
    z = acc * inverse[:, None] * scale[None, :] + base[None, :]

    pre = tl.sigmoid(z) + eps
    tl.store(
        next_pre_ptr + rows[:, None] * STREAMS + mixes[None, :],
        pre,
        mask=live[:, None] & (group == 0)[None, :],
    )
    post = 2.0 * tl.sigmoid(z)
    tl.store(
        next_post_ptr + rows[:, None] * STREAMS + (mixes - STREAMS)[None, :],
        post,
        mask=live[:, None] & (group == 1)[None, :],
    )

    # The combination logits as `[BLOCK_T, groups, STREAMS]`: group `2 + j` is
    # row `j`. Every group is normalized along its row; only the rows of the
    # matrix enter a column's sum.
    logits = tl.reshape(z, (BLOCK_T, groups, STREAMS))
    row = tl.arange(0, groups)[None, :, None] - 2
    in_matrix = (row >= 0) & (row < STREAMS)

    largest = tl.max(logits, axis=2)
    exponent = tl.exp(logits - largest[:, :, None])
    comb = exponent / tl.sum(exponent, axis=2)[:, :, None] + eps
    column_sums = tl.sum(tl.where(in_matrix, comb, 0.0), axis=1)
    comb = comb / (column_sums[:, None, :] + eps)
    for _ in range(ITERATIONS - 1):
        comb = comb / (tl.sum(comb, axis=2)[:, :, None] + eps)
        column_sums = tl.sum(tl.where(in_matrix, comb, 0.0), axis=1)
        comb = comb / (column_sums[:, None, :] + eps)

    lane = tl.arange(0, STREAMS)[None, None, :]
    tl.store(
        next_comb_ptr
        + rows[:, None, None] * (STREAMS * STREAMS)
        + row * STREAMS
        + lane,
        comb,
        mask=live[:, None, None] & in_matrix,
    )

    # The input: the collapsed columns normalized by their own statistic, in
    # tiles of `NORM` columns. Every thread of this program, and every split
    # before it, stored them.
    inverse = tl.rsqrt(collapsed / HIDDEN + norm_eps)
    tl.debug_barrier()
    lanes = tl.arange(0, NORM)
    for start in range(0, HIDDEN, NORM):
        columns = start + lanes
        inside = columns < HIDDEN
        offsets = rows[:, None] * HIDDEN + columns[None, :]
        x = tl.load(
            input_ptr + offsets,
            mask=live[:, None] & inside[None, :],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        gain = tl.load(norm_ptr + columns, mask=inside, other=0.0).to(
            tl.float32
        )
        tl.store(
            input_ptr + offsets,
            x * inverse[:, None] * gain[None, :],
            mask=live[:, None] & inside[None, :],
        )


@triton.jit(
    do_not_specialize=["tokens", "splits", "rms_eps", "eps", "norm_eps"]
)
def _mhc_boundary_kernel(
    streams_ptr,
    output_ptr,
    post_ptr,
    comb_ptr,
    written_ptr,
    weight_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    norm_ptr,
    input_ptr,
    next_pre_ptr,
    next_post_ptr,
    next_comb_ptr,
    partials_ptr,
    counters_ptr,
    tokens,
    splits,
    rms_eps,
    eps,
    norm_eps,
    STREAMS: tl.constexpr,  # noqa: N803
    HIDDEN: tl.constexpr,  # noqa: N803
    ITERATIONS: tl.constexpr,  # noqa: N803
    POST: tl.constexpr,  # noqa: N803
    BLOCK_T: tl.constexpr,  # noqa: N803
    BLOCK_D: tl.constexpr,  # noqa: N803
    MIXES: tl.constexpr,  # noqa: N803
    MIX_TILE: tl.constexpr,  # noqa: N803
    NORM: tl.constexpr,  # noqa: N803
    CHUNK: tl.constexpr,  # noqa: N803
):
    """One split of one token block; grid ``(cdiv(tokens, BLOCK_T), splits)``.

    ``streams``, ``written`` are ``[tokens, STREAMS, HIDDEN]``; ``output``,
    ``input`` ``[tokens, HIDDEN]``; ``post``, ``pre``, the next pre- and
    post-mixes ``[tokens, STREAMS]`` FP32; the combination matrices
    ``[tokens, STREAMS, STREAMS]`` FP32; ``weight`` ``[MIXES, STREAMS *
    HIDDEN]`` FP32; ``partials`` ``[splits, tokens, MIX_TILE + 2]`` FP32 and
    ``counters`` ``[cdiv(tokens, BLOCK_T)]`` I32 when ``splits > 1``.
    """
    width: tl.constexpr = STREAMS * HIDDEN
    steps: tl.constexpr = HIDDEN // BLOCK_D
    partial: tl.constexpr = MIX_TILE + 2

    block = tl.program_id(0)
    split = tl.program_id(1)
    rows = block * BLOCK_T + tl.arange(0, BLOCK_T)
    live = rows < tokens
    rows = rows.to(tl.int64)
    lanes = tl.arange(0, BLOCK_D)
    mixes = tl.arange(0, MIX_TILE)

    # This split's column steps: whole steps, the first `steps % splits` splits
    # one more.
    share = steps // splits
    first = split * share + tl.minimum(split, steps % splits)
    count = share + (split < steps % splits).to(tl.int32)

    acc = tl.zeros((BLOCK_T, MIX_TILE), dtype=tl.float32)
    squares = tl.zeros((BLOCK_T,), dtype=tl.float32)
    collapsed = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for step in range(first, first + count):
        columns = step * BLOCK_D + lanes
        tile = rows[:, None] * width + columns[None, :]
        if POST:
            output = tl.load(
                output_ptr + rows[:, None] * HIDDEN + columns[None, :],
                mask=live[:, None],
                other=0.0,
            ).to(tl.float32)

        x = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        for k in tl.static_range(STREAMS):
            if POST:
                # h'_k = post_k y + sum_j comb[j, k] h_j: the sum over the read
                # streams first, as the reference adds it. The streams' tiles of
                # a step stay in the L1 cache across the written streams.
                mixed = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
                for j in tl.static_range(STREAMS):
                    read = tl.load(
                        streams_ptr + tile + j * HIDDEN,
                        mask=live[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    weight = tl.load(
                        comb_ptr + rows * (STREAMS * STREAMS) + j * STREAMS + k,
                        mask=live,
                        other=0.0,
                    )
                    mixed += weight[:, None] * read
                gain = tl.load(
                    post_ptr + rows * STREAMS + k, mask=live, other=0.0
                )
                h = (gain[:, None] * output + mixed).to(
                    written_ptr.dtype.element_ty
                )
                tl.store(written_ptr + tile + k * HIDDEN, h, mask=live[:, None])
                h = h.to(tl.float32)
            else:
                h = tl.load(
                    streams_ptr + tile + k * HIDDEN,
                    mask=live[:, None],
                    other=0.0,
                ).to(tl.float32)

            squares += tl.sum(h * h, axis=1)
            carried = tl.load(
                pre_ptr + rows * STREAMS + k, mask=live, other=0.0
            )
            x += carried[:, None] * h

            # The projection's `[MIX_TILE, BLOCK_D]` slab of these columns of
            # stream k, the rows past the mixes zero.
            slab = tl.load(
                weight_ptr
                + mixes[:, None] * width
                + k * HIDDEN
                + columns[None, :],
                mask=(mixes < MIXES)[:, None],
                other=0.0,
            )
            acc = tl.dot(h, tl.trans(slab), acc, input_precision="tf32x3")

        # The input's columns, read with the carried pre-mix and rounded as the
        # reference rounds its collapse.
        x = x.to(input_ptr.dtype.element_ty)
        tl.store(
            input_ptr + rows[:, None] * HIDDEN + columns[None, :],
            x,
            mask=live[:, None],
        )
        x = x.to(tl.float32)
        collapsed += tl.sum(x * x, axis=1)

    if splits == 1:
        _finish(
            acc,
            squares,
            collapsed,
            rows,
            live,
            scale_ptr,
            base_ptr,
            norm_ptr,
            input_ptr,
            next_pre_ptr,
            next_post_ptr,
            next_comb_ptr,
            rms_eps,
            eps,
            norm_eps,
            STREAMS,
            HIDDEN,
            ITERATIONS,
            BLOCK_T,
            NORM,
            MIXES,
            MIX_TILE,
        )
    else:
        # Publish this split's partials, then count its arrival.
        published = partials_ptr + (split * tokens + rows) * partial
        tl.store(published[:, None] + mixes[None, :], acc, mask=live[:, None])
        tl.store(published + MIX_TILE, squares, mask=live)
        tl.store(published + MIX_TILE + 1, collapsed, mask=live)
        tl.debug_barrier()
        arrived = tl.atomic_add(
            counters_ptr + block, 1, sem="acq_rel", scope="gpu"
        )
        if arrived == splits - 1:
            # The partials of every split, `CHUNK` splits at a time, summed in
            # split order.
            acc = tl.zeros((BLOCK_T, MIX_TILE), dtype=tl.float32)
            squares = tl.zeros((BLOCK_T,), dtype=tl.float32)
            collapsed = tl.zeros((BLOCK_T,), dtype=tl.float32)
            chunk = tl.arange(0, CHUNK)
            for start in range(0, splits, CHUNK):
                held = (start + chunk) < splits
                summed = (
                    partials_ptr
                    + (
                        (start + chunk)[:, None].to(tl.int64) * tokens
                        + rows[None, :]
                    )
                    * partial
                )
                present = held[:, None] & live[None, :]
                acc += tl.sum(
                    tl.load(
                        summed[:, :, None] + mixes[None, None, :],
                        mask=present[:, :, None],
                        other=0.0,
                        cache_modifier=".cg",
                    ),
                    axis=0,
                )
                squares += tl.sum(
                    tl.load(
                        summed + MIX_TILE,
                        mask=present,
                        other=0.0,
                        cache_modifier=".cg",
                    ),
                    axis=0,
                )
                collapsed += tl.sum(
                    tl.load(
                        summed + MIX_TILE + 1,
                        mask=present,
                        other=0.0,
                        cache_modifier=".cg",
                    ),
                    axis=0,
                )
            tl.store(counters_ptr + block, 0)
            _finish(
                acc,
                squares,
                collapsed,
                rows,
                live,
                scale_ptr,
                base_ptr,
                norm_ptr,
                input_ptr,
                next_pre_ptr,
                next_post_ptr,
                next_comb_ptr,
                rms_eps,
                eps,
                norm_eps,
                STREAMS,
                HIDDEN,
                ITERATIONS,
                BLOCK_T,
                NORM,
                MIXES,
                MIX_TILE,
            )


def derive(fields, parameters, target):
    """The mixes and the lanes of a program's mix tile, for the launch table.

    A connection of ``n`` streams has ``n (n + 2)`` mixes; the tile is their
    count rounded up to a power of two of at least the sixteen columns a
    tensor-core product takes, a whole number of ``n``-wide groups.
    """
    streams = int(fields["streams"])
    hidden = int(fields["hidden"])
    tokens = int(fields["tokens"])
    mixes = streams * (streams + 2)
    tile = max(16, 1 << (mixes - 1).bit_length())
    # The epilogue's tiles hold about 8192 values: the normalization's
    # `[tokens, norm]` columns and the reduction's `[chunk, tokens, tile]`
    # partials.
    norm = min(max(8192 // tokens, 16), 1 << (hidden - 1).bit_length())
    chunk = max(8192 // (tokens * tile), 1)
    return {"mixes": mixes, "mix_tile": tile, "norm": norm, "chunk": chunk}
