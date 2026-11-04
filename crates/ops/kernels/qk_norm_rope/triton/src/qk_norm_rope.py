"""Fused Q/K RMS normalization and rotation: the Triton source of the
``qk_norm_rope`` operator.

One program grid normalizes and rotates Q rows, then K rows: each row's RMS
normalization domains and its rotary axes are compile-time tuples, so programs
own complete token/head rows, accumulate in FP32 and round once into
caller-supplied outputs. Q and K share the domains, axes and factor tables but
keep their own weights, head counts and strides.

The kernel bodies are the ones UniServe normalized and rotated with
(``uniserve_kernels/src/uniserve_kernels/rope.py``), unchanged except that
``pdl_prologue`` lives here, so the file imports nothing but Triton: the shape
and alias checks, the factor tables and the torch types stay with the
operator's Rust side.
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
def _norm_rope_tensor(
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
    EPS: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    ROWS: tl.constexpr,  # noqa: N803
    HALVES: tl.constexpr,  # noqa: N803
    STEPWISE: tl.constexpr,  # noqa: N803
):
    """Process one tensor's rows on the split-half or the row path.

    ``HALVES`` selects :func:`_norm_rope_halves`, which requires the
    stepwise recipe, one domain over the head and one split-half axis;
    both paths store the same bits.
    """
    if HALVES:
        _norm_rope_halves(
            source,
            weights[0],
            cosines[0],
            sines[0],
            output,
            first_row,
            row_count,
            HEADS,
            SOURCE_STRIDES,
            OUTPUT_STRIDES,
            DIM,
            AXIS_ROTATED[0],
            FACTOR_STRIDES[0],
            EPS,
            ROWS,
        )
    else:
        _norm_rope_rows(
            source,
            weights,
            cosines,
            sines,
            output,
            first_row,
            row_count,
            HEADS,
            SOURCE_STRIDES,
            OUTPUT_STRIDES,
            DIM,
            DOMAIN_STARTS,
            DOMAIN_ENDS,
            AXIS_STARTS,
            AXIS_ROTATED,
            FACTOR_STRIDES,
            False,
            EPS,
            BLOCK,
            ROWS,
            STEPWISE,
        )

@triton.jit(do_not_specialize=["q_rows", "k_rows"])
def _qk_norm_rope_kernel(
    q,
    k,
    q_weights,
    k_weights,
    cosines,
    sines,
    q_out,
    k_out,
    q_rows,
    k_rows,
    Q_HEADS: tl.constexpr,  # noqa: N803
    K_HEADS: tl.constexpr,  # noqa: N803
    Q_STRIDES: tl.constexpr,  # noqa: N803
    K_STRIDES: tl.constexpr,  # noqa: N803
    Q_OUT_STRIDES: tl.constexpr,  # noqa: N803
    K_OUT_STRIDES: tl.constexpr,  # noqa: N803
    DIM: tl.constexpr,  # noqa: N803
    DOMAIN_STARTS: tl.constexpr,  # noqa: N803
    DOMAIN_ENDS: tl.constexpr,  # noqa: N803
    AXIS_STARTS: tl.constexpr,  # noqa: N803
    AXIS_ROTATED: tl.constexpr,  # noqa: N803
    FACTOR_STRIDES: tl.constexpr,  # noqa: N803
    EPS: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    ROWS: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
    HALVES: tl.constexpr,  # noqa: N803
    STEPWISE: tl.constexpr,  # noqa: N803
):
    """Normalize and rotate Q rows, then K rows, over one program grid.

    Programs below ``cdiv(q_rows, ROWS)`` process query rows and the
    rest process key rows; Q and K share domains, axes and factor tables
    but keep their own weights, head counts and strides. The branch is
    uniform within each program. ``PDL`` launches run
    :func:`pdl_prologue` before the first load.
    """
    _pdl_prologue(PDL)

    program = tl.program_id(0)
    query_programs = tl.cdiv(q_rows, ROWS)
    if program < query_programs:
        _norm_rope_tensor(
            q,
            q_weights,
            cosines,
            sines,
            q_out,
            program * ROWS,
            q_rows,
            Q_HEADS,
            Q_STRIDES,
            Q_OUT_STRIDES,
            DIM,
            DOMAIN_STARTS,
            DOMAIN_ENDS,
            AXIS_STARTS,
            AXIS_ROTATED,
            FACTOR_STRIDES,
            EPS,
            BLOCK,
            ROWS,
            HALVES,
            STEPWISE,
        )
    else:
        _norm_rope_tensor(
            k,
            k_weights,
            cosines,
            sines,
            k_out,
            (program - query_programs) * ROWS,
            k_rows,
            K_HEADS,
            K_STRIDES,
            K_OUT_STRIDES,
            DIM,
            DOMAIN_STARTS,
            DOMAIN_ENDS,
            AXIS_STARTS,
            AXIS_ROTATED,
            FACTOR_STRIDES,
            EPS,
            BLOCK,
            ROWS,
            HALVES,
            STEPWISE,
        )
