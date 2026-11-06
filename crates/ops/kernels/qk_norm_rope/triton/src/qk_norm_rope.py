"""Per-head RMS normalization and rotation of queries and keys: the Triton
source of the ``qk_norm_rope`` operator.

One program normalizes and rotates the query heads and the key heads of a block
of ``ROWS`` tokens. Each head's row is normalized by its own RMS with FP32
statistics — over the whole head, so the channels that do not rotate are
normalized too — and then rotated by the token's cosine and sine factors:
split-half rotation ("half") or adjacent-pair rotation ("interleaved"). Only the
leading ``ROTATED`` channels of a head turn; a multimodal transformer rotates
part of a head (H3 rotates 96 of 128) and the rest keeps its normalized value.
The result is rounded once into the output dtype.

The learned scale is **one weight per channel**, a `[head_dim]` vector shared by every head of that
side (the operator requires that shape, `rope.rs`'s `q_weight`/`k_weight` check): the kernel loads
`weight + column`, not `weight + head · head_dim`. A caller that handed it a `[heads, head_dim]` table
would get the first head's weights for all of them, which looks like a kernel defect and is not.

The numbers are the ones UniServe's kernel computed
(``uniserve_kernels/src/uniserve_kernels/rope.py``): FP32 sum of squares over
the head, ``x · rsqrt(mean + eps) · weight``, one product and one sum per
rotated element, one rounding to the output. A descriptor without a learned
scale (the video VAE's per-head normalization) compiles a program with no
weight ports at all, so the two forms are two kernels rather than one with a
null pointer.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _turned_head(
    source,
    weight,
    cosine,
    sine,
    out,
    head_base,
    live,
    DIM: tl.constexpr,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    AFFINE: tl.constexpr,
    epsilon,
):
    """Normalize and rotate one head of every live token of the block.

    ``head_base`` is the block's element offset of that head per token, and
    ``epsilon`` the RMS floor. The whole row is stored normalized first; the
    leading ``ROTATED`` channels are then overwritten with the turned values, so
    a narrowed rotation costs only the pairs it turns.
    """
    columns = tl.arange(0, DIM)
    row = tl.load(
        source + head_base[:, None] + columns[None, :], mask=live[:, None], other=0.0
    ).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(row * row, axis=1) / DIM + epsilon)
    if AFFINE:
        row = row * inverse[:, None] * tl.load(weight + columns).to(tl.float32)[None, :]
    else:
        row = row * inverse[:, None]
    if ROTATED != DIM:
        tl.store(out + head_base[:, None] + columns[None, :], row, mask=live[:, None])

    # The turned pairs, in tiles of a power of two: a rotation narrower than the
    # head (96 of 128) has a pair count that no lane count equals.
    pair = tl.arange(0, PAIRS)
    turned = pair < ROTATED // 2
    if INTERLEAVED:
        first_column = 2 * pair
        second_column = 2 * pair + 1
    else:
        first_column = pair
        second_column = pair + ROTATED // 2
    pair_mask = live[:, None] & turned[None, :]
    first = tl.load(
        source + head_base[:, None] + first_column[None, :], mask=pair_mask, other=0.0
    ).to(tl.float32)
    second = tl.load(
        source + head_base[:, None] + second_column[None, :], mask=pair_mask, other=0.0
    ).to(tl.float32)
    if AFFINE:
        first = (
            first
            * inverse[:, None]
            * tl.load(weight + first_column).to(tl.float32)[None, :]
        )
        second = (
            second
            * inverse[:, None]
            * tl.load(weight + second_column).to(tl.float32)[None, :]
        )
    else:
        first = first * inverse[:, None]
        second = second * inverse[:, None]

    tl.store(
        out + head_base[:, None] + first_column[None, :],
        first * cosine - second * sine,
        mask=pair_mask,
    )
    tl.store(
        out + head_base[:, None] + second_column[None, :],
        second * cosine + first * sine,
        mask=pair_mask,
    )


@triton.jit
def _norm_rope(
    query_ptr,
    key_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cosine_ptr,
    sine_ptr,
    q_out_ptr,
    k_out_ptr,
    tokens,
    epsilon,
    Q_HEADS: tl.constexpr,  # noqa: N803
    K_HEADS: tl.constexpr,  # noqa: N803
    DIM: tl.constexpr,
    ROTATED: tl.constexpr,  # noqa: N803
    PAIRS: tl.constexpr,  # noqa: N803
    ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    AFFINE: tl.constexpr,  # noqa: N803
):
    """The block's work: every query head, then every key head."""
    program = tl.program_id(0).to(tl.int64)
    token = program * ROWS + tl.arange(0, ROWS)
    live = token < tokens

    # One factor pair per turned feature, shared by every head of the token.
    pair = tl.arange(0, PAIRS)
    factors = token[:, None] * (ROTATED // 2) + pair[None, :]
    factor_mask = live[:, None] & (pair < ROTATED // 2)[None, :]
    cosine = tl.load(cosine_ptr + factors, mask=factor_mask, other=0.0)
    sine = tl.load(sine_ptr + factors, mask=factor_mask, other=0.0)

    for head in tl.range(0, Q_HEADS):
        _turned_head(
            query_ptr,
            q_weight_ptr,
            cosine,
            sine,
            q_out_ptr,
            (token * Q_HEADS + head) * DIM,
            live,
            DIM,
            ROTATED,
            PAIRS,
            INTERLEAVED,
            AFFINE,
            epsilon,
        )

    for head in tl.range(0, K_HEADS):
        _turned_head(
            key_ptr,
            k_weight_ptr,
            cosine,
            sine,
            k_out_ptr,
            (token * K_HEADS + head) * DIM,
            live,
            DIM,
            ROTATED,
            PAIRS,
            INTERLEAVED,
            AFFINE,
            epsilon,
        )


@triton.jit
def _qk_norm_rope_strided_kernel(
    query_ptr,
    key_ptr,
    cosine_ptr,
    sine_ptr,
    tokens,
    row_stride,
    table_rows,
    epsilon,
    pdl,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTATED: tl.constexpr,
    ROWS: tl.constexpr,
    PDL: tl.constexpr,
):
    """No learned scale, heads turned in place through a row stride, a factor table that repeats.

    One program owns ``ROWS`` tokens of one head, which is the tile the reference's kernel
    (``uniserve_kernels/src/uniserve_kernels/rope.py``, ``_qk_rms_norm_partial_rope_kernel``)
    gives a wide row: a loop over heads inside one token would serialize what the machine can run
    side by side. The program index walks the heads fastest, so neighbouring programs read
    neighbouring columns of the same rows. The numbers are that kernel's: FP32 statistics over the
    head, the pair read back as its normalized value, one rounding at the store. Every value a
    program reads, partners included, belongs to its own tile and is loaded before it stores.
    """
    _pdl_prologue(PDL)
    program = tl.program_id(0)
    head = (program % HEADS).to(tl.int64)
    token = (program // HEADS).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    columns = tl.arange(0, DIM)
    live = (token < tokens)[:, None]
    half: tl.constexpr = ROTATED // 2

    partner = tl.where(columns < half, columns + half, columns - half)
    partner = tl.where(columns < ROTATED, partner, columns)
    base = token[:, None] * row_stride + head * DIM
    offsets = base + columns[None, :]
    partners = base + partner[None, :]

    # Compact factors hold one phase per rotated pair, shared by every head of the token and by
    # every tile of the launch.
    turned = live & (columns < ROTATED)[None, :]
    factors = (token % table_rows)[:, None] * half + (columns % half)[None, :]
    cosine = tl.load(cosine_ptr + factors, mask=turned, other=1.0).to(tl.float32)
    sine = tl.load(sine_ptr + factors, mask=turned, other=0.0).to(tl.float32)
    sign = tl.where(columns[None, :] < half, -1.0, 1.0)

    query = tl.load(query_ptr + offsets, mask=live, other=0.0).to(tl.float32)
    query_rstd = tl.rsqrt(tl.sum(query * query, axis=1) / DIM + epsilon)
    query = query * query_rstd[:, None]
    query_partner = tl.load(query_ptr + partners, mask=live, other=0.0).to(tl.float32)
    query_partner = query_partner * query_rstd[:, None]
    query_turned = query * cosine + sign * query_partner * sine
    tl.store(query_ptr + offsets, tl.where(turned, query_turned, query), mask=live)

    key = tl.load(key_ptr + offsets, mask=live, other=0.0).to(tl.float32)
    key_rstd = tl.rsqrt(tl.sum(key * key, axis=1) / DIM + epsilon)
    key = key * key_rstd[:, None]
    key_partner = tl.load(key_ptr + partners, mask=live, other=0.0).to(tl.float32)
    key_partner = key_partner * key_rstd[:, None]
    key_turned = key * cosine + sign * key_partner * sine
    tl.store(key_ptr + offsets, tl.where(turned, key_turned, key), mask=live)


# The four entry points a call's arguments can name.
#
# The operator's call carries the weight views exactly when the descriptor says
# the normalization has them, and a trailing width word exactly when the rotation
# is narrower than the head. A Triton program's parameters are its call's words in
# order, so each combination is its own kernel; the arithmetic is `_norm_rope`'s in
# all four.


@triton.jit
def _qk_norm_rope_kernel(
    query_ptr,
    key_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cosine_ptr,
    sine_ptr,
    q_out_ptr,
    k_out_ptr,
    tokens,
    epsilon,
    pdl,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    AFFINE: tl.constexpr,
    PDL: tl.constexpr,
):
    """Weighted normalization, full-width rotation."""
    _pdl_prologue(PDL)
    _norm_rope(
        query_ptr,
        key_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cosine_ptr,
        sine_ptr,
        q_out_ptr,
        k_out_ptr,
        tokens,
        epsilon,
        Q_HEADS,
        K_HEADS,
        DIM,
        ROTATED,
        PAIRS,
        ROWS,
        INTERLEAVED,
        AFFINE,
    )


@triton.jit
def _qk_norm_rope_narrowed_kernel(
    query_ptr,
    key_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cosine_ptr,
    sine_ptr,
    q_out_ptr,
    k_out_ptr,
    tokens,
    epsilon,
    pdl,
    rotated,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    AFFINE: tl.constexpr,
    PDL: tl.constexpr,
):
    """Weighted normalization, a rotation narrower than the head."""
    _pdl_prologue(PDL)
    _norm_rope(
        query_ptr,
        key_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cosine_ptr,
        sine_ptr,
        q_out_ptr,
        k_out_ptr,
        tokens,
        epsilon,
        Q_HEADS,
        K_HEADS,
        DIM,
        ROTATED,
        PAIRS,
        ROWS,
        INTERLEAVED,
        AFFINE,
    )


@triton.jit
def _qk_norm_rope_unweighted_kernel(
    query_ptr,
    key_ptr,
    cosine_ptr,
    sine_ptr,
    q_out_ptr,
    k_out_ptr,
    tokens,
    epsilon,
    pdl,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    PDL: tl.constexpr,
):
    """No learned scale, full-width rotation."""
    _pdl_prologue(PDL)
    _norm_rope(
        query_ptr,
        key_ptr,
        query_ptr,
        key_ptr,
        cosine_ptr,
        sine_ptr,
        q_out_ptr,
        k_out_ptr,
        tokens,
        epsilon,
        Q_HEADS,
        K_HEADS,
        DIM,
        ROTATED,
        PAIRS,
        ROWS,
        INTERLEAVED,
        False,
    )


@triton.jit
def _qk_norm_rope_unweighted_narrowed_kernel(
    query_ptr,
    key_ptr,
    cosine_ptr,
    sine_ptr,
    q_out_ptr,
    k_out_ptr,
    tokens,
    epsilon,
    pdl,
    rotated,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    PDL: tl.constexpr,
):
    """No learned scale, a rotation narrower than the head."""
    _pdl_prologue(PDL)
    _norm_rope(
        query_ptr,
        key_ptr,
        query_ptr,
        key_ptr,
        cosine_ptr,
        sine_ptr,
        q_out_ptr,
        k_out_ptr,
        tokens,
        epsilon,
        Q_HEADS,
        K_HEADS,
        DIM,
        ROTATED,
        PAIRS,
        ROWS,
        INTERLEAVED,
        False,
    )


