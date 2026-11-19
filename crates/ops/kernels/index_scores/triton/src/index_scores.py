"""The scores of a request's index keys: the Triton source of the
``index_scores`` operator.

One program scores ``BLOCK`` columns of one token's score row::

    s[t, c] = sum_h relu(q[t, h] . key[g]) * w[t, h]

for the group ``g`` the column names, and ``-inf`` for a column of no visible
group. A dense row's column ``c`` is the request's group ``c``, visible below
the token's ``visible`` count; a row restricted to candidate blocks names
``blocks`` blocks of ``CANDIDATES`` groups, and column ``c`` is group
``candidates[t, c // CANDIDATES] * CANDIDATES + c % CANDIDATES``, visible when
the block is named (not negative) and the group is below the count. Group
``g`` lies at slot ``g % GROUPS`` of page ``page_table[rows[t], g // GROUPS]``.

The keys are BF16 rows ``[pages, GROUPS, WIDTH]``, or MXFP4 records whose page
holds every record's ``WIDTH / 2`` value bytes and then every record's
``WIDTH / 32`` UE8M0 scale bytes (``crates/ops/src/record.rs``, the page
layout). A query is read as it is, or rounded to MXFP4 blocks of 32 the way
the record encoding rounds (``QUERY_MXFP4``), as DeepSeek's indexer rounds
its queries before it scores.

Every value either side holds has at most eight significant bits (a BF16
query or key, an E2M1 code times a power of two), which TF32 holds exactly, so
the products run on the tensor cores in TF32 with FP32 accumulation and are
exact before the sum; the rectification, the weighting and the sum over the
heads are FP32. Heads past the query's are
padded with zeros, whose rectified products are zero.
"""

import triton
import triton.language as tl


@triton.jit
def _pow2(exponent):
    """``2^exponent`` for an exponent of a normal FP32 value, from its bits."""
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1(codes):
    """The FP32 value of four-bit E2M1 codes (int32 lanes)."""
    magnitude = codes & 7
    exponent = magnitude >> 1
    mantissa = (magnitude & 1).to(tl.float32)
    # A code of exponent zero is the subnormal 0 or 0.5; the others are
    # (1 + m / 2) * 2^(e - 1), whose power is 1, 2 or 4.
    power = tl.where(exponent == 3, 4.0, tl.where(exponent == 2, 2.0, 1.0))
    value = tl.where(exponent == 0, 0.5 * mantissa, (1.0 + 0.5 * mantissa) * power)
    return tl.where((codes & 8) != 0, -value, value)


@triton.jit
def _e2m1_round(x):
    """``x`` (FP32, within +-6) rounded to the E2M1 grid, nearest even."""
    magnitude = tl.abs(x)
    # The boundaries between neighbouring codes are their midpoints, and a
    # midpoint goes to the even code: the record encoding's rule.
    rounded = tl.where(magnitude > 0.25, 0.5, 0.0)
    rounded = tl.where(magnitude >= 0.75, 1.0, rounded)
    rounded = tl.where(magnitude > 1.25, 1.5, rounded)
    rounded = tl.where(magnitude >= 1.75, 2.0, rounded)
    rounded = tl.where(magnitude > 2.5, 3.0, rounded)
    rounded = tl.where(magnitude >= 3.5, 4.0, rounded)
    rounded = tl.where(magnitude > 5.0, 6.0, rounded)
    return tl.where(x < 0, -rounded, rounded)


@triton.jit
def _mxfp4_round(x, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    """``x``, ``[ROWS, WIDTH]`` FP32, rounded to MXFP4 blocks of 32: the
    values a record of each block holds.
    """  # noqa: D205
    blocks = tl.reshape(x, (ROWS, WIDTH // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(blocks), axis=2), 6.0 * 2.0**-126)
    # ceil(log2(amax * f32(1/6))), read off the bits: the exponent, plus one
    # for a non-zero mantissa.
    bits = (amax * (1.0 / 6.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = _pow2(exponent)[:, :, None]
    quotient = tl.minimum(tl.maximum(blocks / scale, -6.0), 6.0)
    return tl.reshape(_e2m1_round(quotient) * scale, (ROWS, WIDTH))


@triton.jit(do_not_specialize=["tokens", "max_pages", "blocks"])
def _index_scores_kernel(
    query,
    weights,
    keys,
    page_table,
    rows,
    visible,
    candidates,
    out,
    tokens,
    max_pages,
    blocks,
    HEADS: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    KEYS_MXFP4: tl.constexpr,
    WEIGHTED: tl.constexpr,
    QUERY_MXFP4: tl.constexpr,
    CANDIDATES: tl.constexpr,
):
    """Score ``BLOCK`` columns of one token's row. Grid: (column blocks,
    tokens).

    ``query`` is ``[tokens, HEADS, WIDTH]``, ``weights`` ``[tokens, HEADS]``
    FP32 (read when ``WEIGHTED``), ``candidates`` ``[tokens, blocks]`` I32
    (read when ``CANDIDATES`` is not zero) and ``out`` ``[tokens, columns]``
    FP32, ``columns`` being ``max_pages * GROUPS`` or ``blocks * CANDIDATES``.
    """
    token = tl.program_id(1).to(tl.int64)
    column = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)

    row = tl.load(rows + token).to(tl.int64)
    seen = tl.load(visible + token).to(tl.int64)
    groups_of_table = max_pages * GROUPS

    # ---- The group each column scores, and whether it is visible. ----
    if CANDIDATES > 0:
        columns = blocks * CANDIDATES
        live = column < columns
        block = tl.load(
            candidates + token * blocks + column // CANDIDATES, mask=live, other=-1
        )
        named = block.to(tl.int64) * CANDIDATES + column % CANDIDATES
        valid = live & (block >= 0) & (named < seen) & (named < groups_of_table)
        group = tl.where(valid, named, 0)
    else:
        columns = groups_of_table
        live = column < columns
        valid = live & (column < seen)
        group = tl.where(valid, column, 0)

    page = tl.load(page_table + row * max_pages + group // GROUPS, mask=valid, other=0)
    page = page.to(tl.int64)
    slot = group % GROUPS

    # ---- The keys, [BLOCK, WIDTH], as BF16-exact FP32 values. ----
    lane = tl.arange(0, WIDTH)
    if KEYS_MXFP4:
        # A page holds GROUPS records' WIDTH / 2 value bytes, then their
        # WIDTH / 32 scale bytes; element j is the low nibble of byte j / 2
        # when even.
        page_bytes = GROUPS * (WIDTH // 2 + WIDTH // 32)
        base = page * page_bytes
        packed = tl.load(
            keys + base[:, None] + slot[:, None] * (WIDTH // 2) + lane[None, :] // 2,
            mask=valid[:, None],
            other=0,
        ).to(tl.int32)
        codes = tl.where((lane[None, :] % 2) == 0, packed & 0x0F, packed >> 4)
        scale_bytes = tl.load(
            keys
            + base[:, None]
            + GROUPS * (WIDTH // 2)
            + slot[:, None] * (WIDTH // 32)
            + lane[None, :] // 32,
            mask=valid[:, None],
            other=127,
        ).to(tl.int32)
        key = _e2m1(codes) * _pow2(scale_bytes - 127)
    else:
        key = tl.load(
            keys + ((page * GROUPS + slot) * WIDTH)[:, None] + lane[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

    # ---- The token's queries, [HEADS_TILE, WIDTH], zero past its heads. ----
    head = tl.arange(0, HEADS_TILE)
    heads_live = head < HEADS
    q = tl.load(
        query + token * (HEADS * WIDTH) + head[:, None] * WIDTH + lane[None, :],
        mask=heads_live[:, None],
        other=0.0,
    ).to(tl.float32)
    if QUERY_MXFP4:
        q = _mxfp4_round(q, HEADS_TILE, WIDTH)

    # Both sides hold at most eight significant bits, which TF32's eleven hold
    # exactly, so the tensor-core products are exact and only their FP32 sum
    # rounds. (TF32 rather than BF16 operands: the interpreter multiplies a
    # BF16 tile by its bit patterns, and FP32 operands read the same on both.)
    products = tl.dot(q, tl.trans(key), input_precision="tf32", out_dtype=tl.float32)
    rectified = tl.maximum(products, 0.0)
    if WEIGHTED:
        weight = tl.load(weights + token * HEADS + head, mask=heads_live, other=0.0)
        rectified = rectified * weight[:, None]
    score = tl.sum(rectified, axis=0)

    tl.store(
        out + token * columns + column,
        tl.where(valid, score, float("-inf")),
        mask=live,
    )


def derive(fields, parameters, target):
    """The variants a key prints only when they are set, and the heads' tile.

    A key states ``weighted``, ``query`` and ``candidates`` only when they are
    not the unweighted, unrounded, dense scoring; the products need a tile of
    at least sixteen rows, so the heads are padded to a power of two of at
    least sixteen.

    Raises:
        SystemExit: When a rounded query's or a packed key's width is no
            multiple of the MXFP4 block.
    """
    query = fields.get("query", "bf16")
    if (query == "mxfp4" or fields["keys"] == "mxfp4") and fields["width"] % 32:
        raise SystemExit(
            f"MXFP4 blocks of 32 do not divide rows of {fields['width']}"
        )
    return {
        "weighted": fields.get("weighted", False),
        "query": query,
        "candidates": fields.get("candidates", 0),
        "heads_tile": max(16, triton.next_power_of_2(fields["heads"])),
    }
