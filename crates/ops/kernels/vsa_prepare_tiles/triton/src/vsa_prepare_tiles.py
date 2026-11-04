"""Tile preparation of segment-wise video sparse attention: the Triton source of
the ``vsa_prepare_tiles`` operator.

One program prepares one tile of one head: with ``NORM`` it normalizes and
rotates the valid rows of Q and K exactly as ``qk_norm_rope`` evaluates
``Rounding.STEPWISE``, it stores zeros to the Q, K and V rows past the tile's
valid size, and once the program's stores are visible it writes the tile's FP32
means of the stored rows to the contiguous ``[tiles, heads, width]`` pooled
outputs, in the summation order of
``rows.view(tiles, tile, heads, width).sum(1, dtype=torch.float32)``.

The routine that normalizes and rotates a row block travels with this package
because it is the arithmetic the operator must reproduce: the word-level
rounding of the stepwise path, not a formula. Both parts are the code UniServe
launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/attention/vsa_segments.py`` and
``uniserve_kernels/src/uniserve_kernels/rope.py``), unchanged except that they
import nothing but Triton: the launch-time wrappers, their operand checks and
their torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl

@triton.jit
def _narrowed(values, dtype: tl.constexpr):
    """Round ``values`` to the nearest-even ``dtype`` value.

    FP32 values bound for a 16-bit dtype convert in adjacent pairs with
    one packed ``cvt``, which rounds exactly like the per-value
    conversion of ``.to`` at half its instruction count.
    """
    if values.dtype == dtype:
        return values
    elif values.dtype == tl.float32 and dtype == tl.bfloat16:
        return tl.inline_asm_elementwise(
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,r,r",
            [values],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
        )
    elif values.dtype == tl.float32 and dtype == tl.float16:
        return tl.inline_asm_elementwise(
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,r,r",
            [values],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )
    else:
        return values.to(dtype)

@triton.jit
def _widened(values, dtype: tl.constexpr):
    """Round FP32 ``values`` to ``dtype`` and return them in FP32."""
    return _narrowed(values, dtype).to(tl.float32)

@triton.jit
def _unpacked(words, dtype: tl.constexpr):
    """Return the FP32 values of uint32 ``words`` of two ``dtype`` halves.

    The first result holds each word's low half, the even feature.
    """
    if dtype == tl.bfloat16:
        high = tl.full((), 0xFFFF0000, tl.uint32)
        return (
            (words << 16).to(tl.float32, bitcast=True),
            (words & high).to(tl.float32, bitcast=True),
        )
    else:
        return tl.inline_asm_elementwise(
            "{ .reg .b16 low, high; mov.b32 {low, high}, $2; "
            "cvt.f32.f16 $0, low; cvt.f32.f16 $1, high; }",
            "=r,=r,r",
            [words],
            dtype=(tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

@triton.jit
def _packed(low, high, dtype: tl.constexpr):
    """Round FP32 ``low`` and ``high`` to ``dtype`` halves of uint32 words.

    One ``cvt`` rounds both values to nearest-even, exactly as ``.to``
    rounds each of them.
    """
    if dtype == tl.bfloat16:
        convert: tl.constexpr = "cvt.rn.bf16x2.f32 $0, $2, $1;"
    else:
        convert: tl.constexpr = "cvt.rn.f16x2.f32 $0, $2, $1;"
    return tl.inline_asm_elementwise(
        convert,
        "=r,r,r",
        [low, high],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def _paired(a, b, OP: tl.constexpr, dtype: tl.constexpr):  # noqa: N803
    """Return ``a OP b`` for each ``dtype`` half of uint32 words.

    ``OP`` is ``"mul"``, ``"add"`` or ``"sub"``, and each half rounds
    once to nearest-even. FP32 resolves every product, sum and
    difference of two 16-bit values finely enough that rounding the
    FP32 result equals rounding the exact one, so one packed
    instruction reproduces eager PyTorch's widen, operate and round
    sequence for both halves.
    """
    if dtype == tl.bfloat16:
        suffix: tl.constexpr = ".rn.bf16x2 $0, $1, $2;"
    else:
        suffix: tl.constexpr = ".rn.f16x2 $0, $1, $2;"
    return tl.inline_asm_elementwise(
        OP + suffix, "=r,r,r", [a, b], dtype=tl.uint32, is_pure=True, pack=1
    )

@triton.jit
def _load_pairs(vector, pairs, mask):
    """Return the FP32 even and odd entries of ``vector``'s pairs.

    ``pairs`` indexes pairs of adjacent entries, which one load of twice
    the entry width reads, so ``vector`` starts on a pair boundary.
    Pairs outside ``mask`` hold unspecified values.
    """
    entry = vector.dtype.element_ty
    if entry == tl.float32:
        words = tl.load(
            vector.to(tl.pointer_type(tl.uint64), bitcast=True) + pairs,
            mask=mask,
        )
        return (
            words.to(tl.uint32).to(tl.float32, bitcast=True),
            (words >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
        )
    else:
        words = tl.load(
            vector.to(tl.pointer_type(tl.uint32), bitcast=True) + pairs,
            mask=mask,
        )
        return (
            words.to(tl.uint16).to(entry, bitcast=True).to(tl.float32),
            (words >> 16)
            .to(tl.uint16)
            .to(entry, bitcast=True)
            .to(tl.float32),
        )

@triton.jit
def _inverse_rms(
    values,
    columns,
    START: tl.constexpr,  # noqa: N803
    END: tl.constexpr,  # noqa: N803
    EPS: tl.constexpr,  # noqa: N803
):
    """Return each row's inverse RMS over features ``[START, END)``.

    ``values`` is an FP32 ``[rows, BLOCK]`` tile and ``columns`` its
    feature indices. The row sum associates as the tile's layout
    dictates, so kernels that must agree bit for bit reduce tiles of one
    shape and source dtype with the same number of warps.
    """
    in_domain = (columns >= START) & (columns < END)
    squares = tl.where(in_domain[None, :], values * values, 0.0)
    return tl.rsqrt(tl.sum(squares, axis=1) / (END - START) + EPS)

@triton.jit
def _norm_rope_rows(
    source,
    weights,
    cosines,
    sines,
    output,
    first_row,
    row_count,
    HEADS: tl.constexpr,  # noqa: N803
    SOURCE_STRIDES: tl.constexpr,  # noqa: N803
    OUTPUT_STRIDES: tl.constexpr,  # noqa: N803
    DIM: tl.constexpr,  # noqa: N803
    DOMAIN_STARTS: tl.constexpr,  # noqa: N803
    DOMAIN_ENDS: tl.constexpr,  # noqa: N803
    AXIS_STARTS: tl.constexpr,  # noqa: N803
    AXIS_ROTATED: tl.constexpr,  # noqa: N803
    FACTOR_STRIDES: tl.constexpr,  # noqa: N803
    INTERLEAVED: tl.constexpr,  # noqa: N803
    EPS: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    ROWS: tl.constexpr,  # noqa: N803
    STEPWISE: tl.constexpr,  # noqa: N803
):
    """Normalize and rotate ``ROWS`` token/head rows of one tensor.

    Row ``r`` is head ``r % HEADS`` of token ``r // HEADS``, addressed
    through ``SOURCE_STRIDES`` / ``OUTPUT_STRIDES`` = ``(token, head)``
    element strides with unit feature stride. ``DOMAIN_STARTS`` and
    ``DOMAIN_ENDS`` partition ``[0, DIM)`` into RMS domains, each scaled by
    its ``weights[domain]`` vector; empty tuples skip normalization.
    Rotary axis ``a`` starts at ``AXIS_STARTS[a]`` and rotates its leading
    ``AXIS_ROTATED[a]`` features with the ``[tokens, rotated / 2]`` tables
    ``cosines[a]`` and ``sines[a]`` (token stride ``FACTOR_STRIDES[a]``):
    split-half pairs feature ``i`` with ``i + rotated / 2``,
    ``INTERLEAVED`` pairs adjacent features. Every other feature is only
    normalized. Each program loads all features and partners of its own
    rows before storing, so the output may alias the source.
    """
    rows = first_row + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK)
    tokens = (rows // HEADS).to(tl.int64)
    heads = rows % HEADS
    mask = (rows[:, None] < row_count) & (columns[None, :] < DIM)
    source_rows = tokens * SOURCE_STRIDES[0] + heads * SOURCE_STRIDES[1]

    # Per rotated feature: its pair index within the factor table, its
    # partner feature, and whether it leads the pair (sine sign -1).
    # Unrotated features keep themselves as partner with no factors.
    partner = columns
    rotated = columns < 0
    leading = columns < 0
    cosine = tl.full((ROWS, BLOCK), 1.0, tl.float32)
    sine = tl.zeros((ROWS, BLOCK), tl.float32)
    for axis in tl.static_range(len(AXIS_STARTS)):
        if AXIS_ROTATED[axis] > 0:
            local = columns - AXIS_STARTS[axis]
            in_axis = (local >= 0) & (local < AXIS_ROTATED[axis])
            if INTERLEAVED:
                pair = local // 2
                partner_local = local ^ 1
                first = local % 2 == 0
            else:
                pair = local % (AXIS_ROTATED[axis] // 2)
                first = local < AXIS_ROTATED[axis] // 2
                partner_local = tl.where(
                    first,
                    local + AXIS_ROTATED[axis] // 2,
                    local - AXIS_ROTATED[axis] // 2,
                )
            partner = tl.where(
                in_axis, AXIS_STARTS[axis] + partner_local, partner
            )
            rotated = rotated | in_axis
            leading = tl.where(in_axis, first, leading)

            factors = tokens[:, None] * FACTOR_STRIDES[axis] + pair[None, :]
            factor_mask = mask & in_axis[None, :]
            cosine = tl.where(
                in_axis[None, :],
                tl.load(
                    cosines[axis] + factors, mask=factor_mask, other=1.0
                ).to(tl.float32),
                cosine,
            )
            sine = tl.where(
                in_axis[None, :],
                tl.load(
                    sines[axis] + factors, mask=factor_mask, other=0.0
                ).to(tl.float32),
                sine,
            )

    # values, partners: [ROWS, BLOCK] FP32 tiles of the source rows.
    values = tl.load(
        source + source_rows[:, None] + columns[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    partners = tl.load(
        source + source_rows[:, None] + partner[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    if len(DOMAIN_ENDS) > 0:
        # A rotary axis lies inside one domain, so a partner shares its
        # feature's RMS denominator and reads its own weight entry.
        scale = tl.zeros((ROWS, BLOCK), tl.float32)
        partner_scale = tl.zeros((ROWS, BLOCK), tl.float32)
        for domain in tl.static_range(len(DOMAIN_ENDS)):
            start = DOMAIN_STARTS[domain]
            in_domain = (columns >= start) & (columns < DOMAIN_ENDS[domain])
            inverse = _inverse_rms(
                values, columns, start, DOMAIN_ENDS[domain], EPS
            )
            weight = tl.load(
                weights[domain] + (columns - start),
                mask=in_domain,
                other=0.0,
            ).to(tl.float32)
            partner_weight = tl.load(
                weights[domain] + (partner - start),
                mask=in_domain,
                other=0.0,
            ).to(tl.float32)
            domain_values = (
                (values * inverse[:, None]) * weight[None, :]
                if STEPWISE
                else inverse[:, None] * weight[None, :]
            )
            domain_partners = (
                (partners * inverse[:, None]) * partner_weight[None, :]
                if STEPWISE
                else inverse[:, None] * partner_weight[None, :]
            )
            scale = tl.where(
                in_domain[None, :],
                domain_values,
                scale,
            )
            partner_scale = tl.where(
                in_domain[None, :],
                domain_partners,
                partner_scale,
            )
        values = scale if STEPWISE else values * scale
        partners = partner_scale if STEPWISE else partners * partner_scale

    if STEPWISE:
        # Preserve eager arithmetic: weighted normalization, factors,
        # products, and finally the sum each round to the source dtype.
        dtype = source.dtype.element_ty
        values = _widened(values, dtype)
        partners = _widened(partners, dtype)
        cosine = _widened(cosine, dtype)
        sine = _widened(sine, dtype)
        direct = _widened(values * cosine, dtype)
        crossed = _widened(partners * sine, dtype)
    else:
        direct = values * cosine
        crossed = partners * sine
    turned = tl.where(leading[None, :], direct - crossed, direct + crossed)
    result = tl.where(rotated[None, :], turned, values)
    if STEPWISE:
        # Stepwise results already hold source-dtype values, which
        # convert in packed pairs. Under a single rounding the compiler
        # packs the conversion itself, together with the FP32
        # arithmetic before it.
        result = _narrowed(result, output.dtype.element_ty)
    output_rows = tokens * OUTPUT_STRIDES[0] + heads * OUTPUT_STRIDES[1]
    tl.store(
        output + output_rows[:, None] + columns[None, :],
        result,
        mask=mask,
    )

@triton.jit
def _norm_rope_halves(
    source,
    weights,
    cosine,
    sine,
    output,
    first_row,
    row_count,
    HEADS: tl.constexpr,  # noqa: N803
    SOURCE_STRIDES: tl.constexpr,  # noqa: N803
    OUTPUT_STRIDES: tl.constexpr,  # noqa: N803
    DIM: tl.constexpr,  # noqa: N803
    ROTATED: tl.constexpr,  # noqa: N803
    FACTOR_STRIDE: tl.constexpr,  # noqa: N803
    EPS: tl.constexpr,  # noqa: N803
    ROWS: tl.constexpr,  # noqa: N803
):
    """Stepwise-normalize and rotate ``ROWS`` 16-bit rows.

    This is the stepwise :func:`_norm_rope_rows` for a power-of-two
    head of ``DIM`` 16-bit features with one RMS domain, scaled by the
    ``weights`` vector, and one rotary axis whose leading ``ROTATED``
    features rotate split-half with the ``[tokens, ROTATED / 2]``
    tables ``cosine`` and ``sine`` (token stride ``FACTOR_STRIDE``),
    and it stores the same bits. Source and output share a dtype, rows
    start on 16-byte boundaries and ``ROTATED`` is a multiple of four.
    Kernels that call it compile without FP32 fusion, and their RMS
    tile keeps the row path's per-row layout: each thread holds a
    16-byte vector of a row, and one warp holds the whole row.

    Features travel as uint32 words of two adjacent 16-bit values. The
    thread that holds a word also holds the word of their rotation
    partners, so every pair rotates once, without reloading or
    renormalizing a partner. Each program loads all of its rows before
    storing, so the output may alias the source.
    """
    rows = first_row + tl.arange(0, ROWS)
    tokens = (rows // HEADS).to(tl.int64)
    heads = rows % HEADS
    valid = rows < row_count
    dtype = source.dtype.element_ty

    # The row path's RMS tile, reduced by the row path's code: a row's
    # sum associates as its per-row layout dictates.
    columns = tl.arange(0, DIM)
    source_rows = tokens * SOURCE_STRIDES[0] + heads * SOURCE_STRIDES[1]
    values = tl.load(
        source + source_rows[:, None] + columns[None, :],
        mask=valid[:, None] & (columns[None, :] < DIM),
        other=0.0,
    ).to(tl.float32)
    inverse = _inverse_rms(values, columns, 0, DIM, EPS)[:, None, None]

    # Word w = 2 * g + e of the [ROWS, DIM / 8, 2] word tiles holds
    # features 2w and 2w + 1 in ``first`` and their rotation partners,
    # half features later, in ``second``; words from half / 2 on hold
    # the first and second half of the unrotated tail. Word pairs load
    # as 8-byte vectors, which spreads each row over the threads that
    # reduce it in the RMS tile, so the inverse stays in registers.
    half: tl.constexpr = ROTATED // 2
    words = tl.arange(0, DIM // 8)[:, None] * 2 + tl.arange(0, 2)[None, :]
    turned = (words < half // 2)[None, :, :]
    first = tl.where(turned, words, words + half // 2)
    second = tl.where(turned, words + half // 2, words + DIM // 4)
    valid = valid[:, None, None]
    source_words = source.to(tl.pointer_type(tl.uint32), bitcast=True)
    source_rows = (
        tokens * (SOURCE_STRIDES[0] // 2) + heads * (SOURCE_STRIDES[1] // 2)
    )[:, None, None]
    # These loads reread rows the RMS tile just loaded.
    leading = tl.load(source_words + source_rows + first, mask=valid)
    trailing = tl.load(source_words + source_rows + second, mask=valid)
    leading_low, leading_high = _unpacked(leading, dtype)
    trailing_low, trailing_high = _unpacked(trailing, dtype)

    # Weights and factors split by word half: word w's features take
    # weight pair ``first`` or ``second`` and rotate by factor pair w.
    # Unrotated words keep their normalized values, so they read no
    # factors.
    leading_weights = _load_pairs(weights, first, None)
    trailing_weights = _load_pairs(weights, second, None)
    factors = tokens[:, None, None] * (FACTOR_STRIDE // 2) + words
    cosines = _load_pairs(cosine, factors, valid & turned)
    sines = _load_pairs(sine, factors, valid & turned)

    # Eager rounding order: the weighted normalization and the factors
    # round to the source dtype, in which the products and the rotated
    # sum then evaluate, each rounding once.
    leading = _packed(
        (leading_low * inverse) * leading_weights[0],
        (leading_high * inverse) * leading_weights[1],
        dtype,
    )
    trailing = _packed(
        (trailing_low * inverse) * trailing_weights[0],
        (trailing_high * inverse) * trailing_weights[1],
        dtype,
    )
    cosines = _packed(cosines[0], cosines[1], dtype)
    sines = _packed(sines[0], sines[1], dtype)
    leading_turned = _paired(
        _paired(leading, cosines, "mul", dtype),
        _paired(trailing, sines, "mul", dtype),
        "sub",
        dtype,
    )
    trailing_turned = _paired(
        _paired(trailing, cosines, "mul", dtype),
        _paired(leading, sines, "mul", dtype),
        "add",
        dtype,
    )

    output_words = output.to(tl.pointer_type(tl.uint32), bitcast=True)
    output_rows = (
        tokens * (OUTPUT_STRIDES[0] // 2) + heads * (OUTPUT_STRIDES[1] // 2)
    )[:, None, None]
    tl.store(
        output_words + output_rows + first,
        tl.where(turned, leading_turned, leading),
        mask=valid,
    )
    tl.store(
        output_words + output_rows + second,
        tl.where(turned, trailing_turned, trailing),
        mask=valid,
    )


@triton.jit
def _add_rn(a, b):
    # Round-to-nearest FP32 sum the compiler cannot contract with a
    # neighbouring multiply.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def _zero_tile_padding_kernel(
    values,
    variable_block_sizes,
    stride_row,
    stride_head,
    heads: tl.constexpr,
    tile_rows: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
):
    """Store zeros to one tile's row block past the tile's valid size."""
    tile = tl.program_id(0)
    block = tl.program_id(1)
    valid = tl.load(variable_block_sizes + tile)
    # A row block wholly inside the valid rows stores nothing.
    if (block + 1) * block_rows > valid:
        local = block * block_rows + tl.arange(0, block_rows)
        rows = (tile * tile_rows + local).to(tl.int64)
        columns = tl.arange(0, width)
        padding = (local >= valid)[:, None]
        zeros = tl.zeros((block_rows, width), dtype=values.dtype.element_ty)
        for head in tl.static_range(heads):
            offsets = (
                rows[:, None] * stride_row
                + head * stride_head
                + columns[None, :]
            )
            tl.store(values + offsets, zeros, mask=padding)

@triton.jit
def _add_gated_tiles_kernel(
    output,
    compressed,
    gate,
    output_stride_row,
    output_stride_head,
    gate_stride_row,
    gate_stride_head,
    compressed_stride_head,
    compressed_stride_tile,
    tile_rows: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
):
    """Add one row block's gated tile compression for one head."""
    block = tl.program_id(0)
    head = tl.program_id(1)
    rows = (block * block_rows + tl.arange(0, block_rows)).to(tl.int64)
    columns = tl.arange(0, width)
    # A row block lies within one tile.
    tile = (block * block_rows) // tile_rows
    tile_values = tl.load(
        compressed
        + head * compressed_stride_head
        + tile * compressed_stride_tile
        + columns
    ).to(tl.float32)
    gate_values = tl.load(
        gate
        + rows[:, None] * gate_stride_row
        + head * gate_stride_head
        + columns[None, :]
    ).to(tl.float32)
    destination = (
        output
        + rows[:, None] * output_stride_row
        + head * output_stride_head
        + columns[None, :]
    )
    # The product rounds to the storage dtype before the sum, as the two
    # eager operations store it in between.
    product = _mul_rn(tile_values[None, :], gate_values)
    product = product.to(
        output.dtype.element_ty, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    total = _add_rn(tl.load(destination).to(tl.float32), product)
    tl.store(
        destination,
        total.to(output.dtype.element_ty, fp_downcast_rounding="rtne"),
    )

@triton.jit(do_not_specialize=["owner_rows", "start", "count"])
def _compose_gated_rows_kernel(
    output,
    fine,
    compressed,
    gate,
    fine_stride_row,
    fine_stride_head,
    gate_stride_row,
    gate_stride_head,
    compressed_stride_head,
    compressed_stride_tile,
    owner_rows,
    start,
    count,
    heads: tl.constexpr,
    tile_rows: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
    slots: tl.constexpr,
):
    """Compose one row block of one member's interval for one head.

    Row ``i`` of member ``m``'s interval is row ``m * owner_rows + start
    + i`` of ``fine`` and ``gate``; its sum lands in row ``i`` of slot
    ``slots[m]`` of the contiguous ``[members, count, heads, width]``
    payload, with the arithmetic of :func:`_add_gated_tiles_kernel`.
    """
    block = tl.program_id(0)
    head = tl.program_id(1)
    member = tl.program_id(2).to(tl.int64)
    local = (block * block_rows + tl.arange(0, block_rows)).to(tl.int64)
    first = member * owner_rows + start + block * block_rows
    rows = first + tl.arange(0, block_rows)
    columns = tl.arange(0, width)
    # Owner rows, interval starts and counts are whole row blocks, so a
    # row block lies within one tile.
    tile = first // tile_rows
    tile_values = tl.load(
        compressed
        + head * compressed_stride_head
        + tile * compressed_stride_tile
        + columns
    ).to(tl.float32)
    gate_values = tl.load(
        gate
        + rows[:, None] * gate_stride_row
        + head * gate_stride_head
        + columns[None, :]
    ).to(tl.float32)
    fine_values = tl.load(
        fine
        + rows[:, None] * fine_stride_row
        + head * fine_stride_head
        + columns[None, :]
    ).to(tl.float32)
    product = _mul_rn(tile_values[None, :], gate_values)
    product = product.to(
        output.dtype.element_ty, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    total = _add_rn(fine_values, product)

    slot = member
    for index in tl.static_range(len(slots)):
        slot = tl.where(member == index, slots[index], slot)
    destination = (
        output
        + ((slot * count + local)[:, None] * heads + head) * width
        + columns[None, :]
    )
    tl.store(
        destination,
        total.to(output.dtype.element_ty, fp_downcast_rounding="rtne"),
    )

@triton.jit
def _phase_sum(
    rows,
    row_stride,
    columns,
    phase: tl.constexpr,
    width: tl.constexpr,
    tile_rows: tl.constexpr,
):
    """Sum the rows ``phase``, ``phase + 4``, ... of one tile.

    Four accumulators take, in row order, the phase's rows ``phase + 4 *
    i + 16 * m`` for ``i`` = 0, 1, 2 and 3, and their sums then add left
    to right: one thread's share of ``torch.sum``'s reduction below.
    """
    first = tl.zeros((width,), dtype=tl.float32)
    second = tl.zeros((width,), dtype=tl.float32)
    third = tl.zeros((width,), dtype=tl.float32)
    fourth = tl.zeros((width,), dtype=tl.float32)
    for step in tl.static_range(tile_rows // 16):
        base = rows + (16 * step + phase) * row_stride + columns
        first = _add_rn(first, tl.load(base).to(tl.float32))
        second = _add_rn(
            second, tl.load(base + 4 * row_stride).to(tl.float32)
        )
        third = _add_rn(
            third, tl.load(base + 8 * row_stride).to(tl.float32)
        )
        fourth = _add_rn(
            fourth, tl.load(base + 12 * row_stride).to(tl.float32)
        )
    return _add_rn(_add_rn(_add_rn(first, second), third), fourth)

@triton.jit
def _tile_mean(
    rows,
    row_stride,
    columns,
    divisor,
    width: tl.constexpr,
    tile_rows: tl.constexpr,
):
    """Return the FP32 mean of the ``tile_rows`` rows at ``rows``.

    Row ``r`` starts at ``rows + r * row_stride``. The sum follows the
    order in which ``torch.sum(dtype=torch.float32)`` reduces a strided
    row axis of 64 or 128 rows: four threads each take the rows of one
    phase modulo four (:func:`_phase_sum`), and a halving tree adds their
    sums as ``(p0 + p2) + (p1 + p3)``. Every sum and the division round
    to nearest, so the mean equals the eager mean bit for bit.
    """
    first = _phase_sum(rows, row_stride, columns, 0, width, tile_rows)
    second = _phase_sum(rows, row_stride, columns, 1, width, tile_rows)
    third = _phase_sum(rows, row_stride, columns, 2, width, tile_rows)
    fourth = _phase_sum(rows, row_stride, columns, 3, width, tile_rows)
    total = _add_rn(_add_rn(first, third), _add_rn(second, fourth))
    return tl.math.div_rn(total, divisor)

@triton.jit
def _prepare_tiles_kernel(
    query,
    key,
    value,
    query_weight,
    key_weight,
    cosine,
    sine,
    variable_block_sizes,
    pooled_query,
    pooled_key,
    pooled_value,
    QUERY_STRIDES: tl.constexpr,  # noqa: N803
    KEY_STRIDES: tl.constexpr,  # noqa: N803
    VALUE_STRIDES: tl.constexpr,  # noqa: N803
    QUERY_ROWS: tl.constexpr,  # noqa: N803
    KEY_ROWS: tl.constexpr,  # noqa: N803
    FACTOR_STRIDE: tl.constexpr,  # noqa: N803
    HEADS: tl.constexpr,  # noqa: N803
    DIM: tl.constexpr,  # noqa: N803
    ROTATED: tl.constexpr,  # noqa: N803
    EPS: tl.constexpr,  # noqa: N803
    TILE: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    NORM: tl.constexpr,  # noqa: N803
):
    """Prepare one tile of one head: rotate Q/K, zero padding, pool.

    ``*_STRIDES`` are the (row, head) strides of Q, K and V; ``QUERY_ROWS``
    and ``KEY_ROWS`` restate the row strides as the (token, head) strides
    of a single-head row sequence, the form :func:`_norm_rope_halves`
    reads. With ``NORM``, each ``BLOCK``-row block's valid rows of Q and
    K are normalized and rotated in place by that routine on the blocks
    and warps of ``qk_norm_rope``'s stepwise launch, which stores its
    bits. The rows past the tile's valid size of Q, K and V then hold
    zeros, and once every store of the program is visible the tile's
    means of the stored rows are written to the contiguous ``[tiles,
    HEADS, DIM]`` pooled outputs.
    """
    tile = tl.program_id(0)
    head = tl.program_id(1)
    valid = tl.load(variable_block_sizes + tile)
    columns = tl.arange(0, DIM)
    first = tile.to(tl.int64) * TILE
    query_tile = query + first * QUERY_STRIDES[0] + head * QUERY_STRIDES[1]
    key_tile = key + first * KEY_STRIDES[0] + head * KEY_STRIDES[1]
    value_tile = value + first * VALUE_STRIDES[0] + head * VALUE_STRIDES[1]

    for block in tl.static_range(TILE // BLOCK):
        offset = block * BLOCK
        if NORM:
            # Only the block's valid rows are normalized and stored, so
            # the padding stores below touch disjoint rows.
            count = tl.minimum(tl.maximum(valid - offset, 0), BLOCK)
            factors = (first + offset) * FACTOR_STRIDE
            block_query = query_tile + offset * QUERY_STRIDES[0]
            block_key = key_tile + offset * KEY_STRIDES[0]
            _norm_rope_halves(
                block_query,
                query_weight,
                cosine + factors,
                sine + factors,
                block_query,
                0,
                count,
                1,
                QUERY_ROWS,
                QUERY_ROWS,
                DIM,
                ROTATED,
                FACTOR_STRIDE,
                EPS,
                BLOCK,
            )
            _norm_rope_halves(
                block_key,
                key_weight,
                cosine + factors,
                sine + factors,
                block_key,
                0,
                count,
                1,
                KEY_ROWS,
                KEY_ROWS,
                DIM,
                ROTATED,
                FACTOR_STRIDE,
                EPS,
                BLOCK,
            )

        local = offset + tl.arange(0, BLOCK)
        padding = (local >= valid)[:, None]
        zeros = tl.zeros((BLOCK, DIM), dtype=query.dtype.element_ty)
        tl.store(
            query_tile
            + local[:, None] * QUERY_STRIDES[0]
            + columns[None, :],
            zeros,
            mask=padding,
        )
        tl.store(
            key_tile + local[:, None] * KEY_STRIDES[0] + columns[None, :],
            zeros,
            mask=padding,
        )
        tl.store(
            value_tile
            + local[:, None] * VALUE_STRIDES[0]
            + columns[None, :],
            zeros,
            mask=padding,
        )

    # Other threads of the program stored rows the means read.
    tl.debug_barrier()
    divisor = tl.maximum(valid, 1).to(tl.float32)
    pooled = (tile.to(tl.int64) * HEADS + head) * DIM + columns
    tl.store(
        pooled_query + pooled,
        _tile_mean(
            query_tile, QUERY_STRIDES[0], columns, divisor, DIM, TILE
        ),
    )
    tl.store(
        pooled_key + pooled,
        _tile_mean(key_tile, KEY_STRIDES[0], columns, divisor, DIM, TILE),
    )
    tl.store(
        pooled_value + pooled,
        _tile_mean(
            value_tile, VALUE_STRIDES[0], columns, divisor, DIM, TILE
        ),
    )
