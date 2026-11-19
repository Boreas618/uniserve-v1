"""The Triton source of the ``latent_attention`` operator.

Attention over one shared latent that is both key and value: one program
attends the queries of ``BLOCK`` heads of one token over the token's entries
(``ops::attention::LatentAttention``)::

    entries  = the window entries at positions max(0, p - WINDOW + 1) ..= p,
               then the compressed entries selection[t, :] names (>= 0)
    s        = scale * q . c                      FP32, c the decoded latent
    m        = max(-1e30, max s)
    o        = sum e^(s - m) c / (sum e^(s - m) + e^(sink - m))

The entries are read a tile of ``TILE`` at a time and the softmax is kept
online: a running maximum floored at ``-1e30`` (so a query that reads nothing
attends its sink alone and is zero, as the reference's ``sparse_attn`` is), a
running denominator and an FP32 accumulator rescaled whenever the maximum
grows. Both products run on the tensor cores over BF16 operands: a decoded
record is a BF16 value exactly in every format (an E4M3 or E2M1 code times a
power of two or an E4M3 scale has at most six significant bits), and each
probability is rounded to BF16 before it multiplies its latent, as the
operator states.

**Records** (``crates/ops/src/record.rs``, per-entry layout: an entry's
values, then its scales): ``bf16`` is ``HEAD_DIM`` BF16 elements; ``mxfp8``
``HEAD_DIM`` E4M3 bytes and a UE8M0 byte per 32; ``nvfp4`` ``HEAD_DIM / 2``
E2M1 bytes (the even element in the low nibble) and an E4M3 byte per 16;
``mxfp4`` the same codes and a UE8M0 byte per 32. Every code is decoded by
its bits, not by a conversion instruction, so the interpreter and a device
decode alike.

**Addressing.** Entry ``e`` of a table lies on page
``table[row, (e // PAGE) % pages]`` at slot ``e % PAGE``, ``PAGE`` being the
window page's entries or the compressed page's (``ENTRIES``), and ``pages``
the table's width: a table as wide as the request addresses its pages
directly, and a narrower window table is a ring of pages.
"""

# Compile-time parameters are spelled in capitals, Triton's convention for
# `tl.constexpr` arguments and the names the launch table binds.
# ruff: noqa: N803

import triton
import triton.language as tl

#: The record codes the derivation compiles in, by the format a key names.
FORMATS = {"bf16": 0, "mxfp8": 1, "nvfp4": 2, "mxfp4": 3}

#: The values one scale byte covers, by format (``Format::block``).
SCALE_BLOCK = {"bf16": None, "mxfp8": 32, "nvfp4": 16, "mxfp4": 32}

#: The rows a tensor-core product needs at least.
MIN_ROWS = 16

#: The entries one tile reads.
ENTRY_TILE = 32


@triton.jit
def _e4m3(codes):
    """The FP32 value of E4M3 bytes: sign, four exponent bits (bias 7) and
    three fraction bits; exponent zero is subnormal, a step of 2^-9.
    """  # noqa: D205
    bits = codes.to(tl.int32)
    exponent = (bits >> 3) & 0xF
    fraction = bits & 0x7
    # A normal code's FP32 bits: the exponent rebiased from 7 to 127, the
    # three fraction bits at the top of the mantissa.
    normal = (((exponent + 120) << 23) | (fraction << 20)).to(
        tl.float32, bitcast=True
    )
    subnormal = fraction.to(tl.float32) * 0.001953125
    magnitude = tl.where(exponent == 0, subnormal, normal)
    return tl.where((bits >> 7) != 0, -magnitude, magnitude)


@triton.jit
def _ue8m0(codes):
    """The FP32 value ``2^(b - 127)`` of UE8M0 scale bytes: the byte is the
    FP32 exponent field itself.
    """  # noqa: D205
    return (codes.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1(nibbles):
    """The FP32 value of E2M1 codes: sign, two exponent bits (bias 1) and one
    fraction bit, the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6.
    """  # noqa: D205
    code = nibbles & 0x7
    exponent = code >> 1
    fraction = (code & 0x1).to(tl.float32)
    # A normal code is (2 + f) * 2^e / 4, the power built from its FP32
    # exponent field; exponent zero is f / 2.
    power = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    normal = (2.0 + fraction) * power * 0.25
    magnitude = tl.where(exponent == 0, 0.5 * fraction, normal)
    return tl.where((nibbles & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _decoded(
    plane,
    record,
    live,
    FORMAT: tl.constexpr,
    RECORD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """The FP32 latents of ``[TILE]`` records of a plane, ``[TILE, HEAD_DIM]``.

    ``record`` is each lane's record index in the plane (page * entries per
    page + slot), int64; a lane that is not ``live`` reads nothing and is
    zero. ``RECORD`` is a record's extent in the plane's elements: BF16
    elements for ``bf16``, bytes otherwise.
    """  # noqa: D205
    channel = tl.arange(0, HEAD_DIM)
    mask = live[:, None]
    base = record[:, None] * RECORD
    if FORMAT == 0:
        latent = tl.load(
            plane + base + channel[None, :], mask=mask, other=0.0
        ).to(tl.float32)
    elif FORMAT == 1:
        codes = tl.load(plane + base + channel[None, :], mask=mask, other=0)
        scales = tl.load(
            plane + base + HEAD_DIM + (channel // 32)[None, :],
            mask=mask,
            other=0,
        )
        latent = _e4m3(codes) * _ue8m0(scales)
    else:
        # Two E2M1 codes a byte, the even channel in the low nibble.
        packed = tl.load(
            plane + base + (channel // 2)[None, :], mask=mask, other=0
        ).to(tl.int32)
        nibbles = tl.where(
            (channel % 2 == 0)[None, :], packed & 0xF, packed >> 4
        )
        if FORMAT == 2:
            scales = tl.load(
                plane + base + HEAD_DIM // 2 + (channel // 16)[None, :],
                mask=mask,
                other=0,
            )
            latent = _e2m1(nibbles) * _e4m3(scales)
        else:
            scales = tl.load(
                plane + base + HEAD_DIM // 2 + (channel // 32)[None, :],
                mask=mask,
                other=0,
            )
            latent = _e2m1(nibbles) * _ue8m0(scales)
    return tl.where(mask, latent, 0.0)


@triton.jit
def _attended(
    query,
    latent,
    live,
    scale,
    peak,
    total,
    acc,
):
    """One tile of the online softmax: the scores of ``query`` against the
    tile's ``latent`` rows, the maximum, the denominator and the accumulator
    rescaled to the new maximum. A lane that is not ``live`` scores ``-inf``
    and weighs nothing.
    """  # noqa: D205
    latent = latent.to(tl.bfloat16)
    scores = tl.dot(query, tl.trans(latent)) * scale
    scores = tl.where(live[None, :], scores, float("-inf"))

    grown = tl.maximum(peak, tl.max(scores, axis=1))
    rescale = tl.exp(peak - grown)
    weights = tl.exp(scores - grown[:, None])
    total = total * rescale + tl.sum(weights, axis=1)
    # The probabilities round to BF16 before they weigh the latents; the
    # denominator keeps their FP32 values.
    acc = acc * rescale[:, None] + tl.dot(weights.to(tl.bfloat16), latent)
    return grown, total, acc


@triton.jit(do_not_specialize=["tokens", "window_pages", "compressed_pages"])
def _latent_attention_kernel(
    query,
    window,
    window_table,
    compressed,
    compressed_table,
    selection,
    rows,
    positions,
    sink,
    out,
    tokens,
    window_pages,
    compressed_pages,
    scale,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    TILE: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW_FORMAT: tl.constexpr,
    WINDOW_RECORD: tl.constexpr,
    COMPRESSED: tl.constexpr,
    COMPRESSED_FORMAT: tl.constexpr,
    COMPRESSED_RECORD: tl.constexpr,
    ENTRIES: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    """Attend ``BLOCK`` heads of one token. Grid: ``(HEADS / BLOCK, tokens)``.

    ``query`` and ``out`` are ``[tokens, HEADS, HEAD_DIM]`` BF16; ``window``
    is ``[pages, PAGE, WINDOW_RECORD]`` and ``compressed`` ``[pages, ENTRIES,
    COMPRESSED_RECORD]`` (the unread operand when not ``COMPRESSED``); the
    tables are ``[table rows, window_pages]`` and ``[table rows,
    compressed_pages]``; ``rows`` and ``positions`` are ``[tokens]``;
    ``selection`` is ``[tokens, COLUMNS]``; ``sink`` is ``[HEADS]`` FP32.
    The head tile ``HEAD_TILE`` covers the block with zero rows past it.
    ``tokens`` is the grid's second extent, which the program reads nowhere.
    """
    block = tl.program_id(0)
    token = tl.program_id(1).to(tl.int64)

    head = tl.arange(0, HEAD_TILE)
    head_live = head < BLOCK
    head_index = block * BLOCK + head
    channel = tl.arange(0, HEAD_DIM)
    # The token's head rows, which the queries and the output share.
    head_rows = (token * HEADS + head_index) * HEAD_DIM
    rows_offset = head_rows[:, None] + channel[None, :]
    q = tl.load(query + rows_offset, mask=head_live[:, None], other=0.0)

    row = tl.load(rows + token).to(tl.int64)
    position = tl.load(positions + token)

    lane = tl.arange(0, TILE)
    peak = tl.full((HEAD_TILE,), -1e30, tl.float32)
    total = tl.zeros((HEAD_TILE,), tl.float32)
    acc = tl.zeros((HEAD_TILE, HEAD_DIM), tl.float32)

    # ---- The window: positions max(0, p - WINDOW + 1) ..= p. A position below
    # zero (a padding token's) reads no entry. ----
    first = tl.maximum(position - WINDOW + 1, 0)
    for start in range(0, WINDOW, TILE):
        offset = start + lane
        entry = first + offset
        live = (offset < WINDOW) & (entry <= position)
        entry = tl.where(live, entry, 0)
        column = (entry // PAGE) % window_pages
        page = tl.load(
            window_table + row * window_pages + column, mask=live, other=0
        )
        record = page.to(tl.int64) * PAGE + entry % PAGE
        latent = _decoded(
            window, record, live, WINDOW_FORMAT, WINDOW_RECORD, HEAD_DIM
        )
        peak, total, acc = _attended(q, latent, live, scale, peak, total, acc)

    # ---- The compressed entries the selection names; a negative one names
    # none. ----
    if COMPRESSED:
        for start in range(0, COLUMNS, TILE):
            offset = start + lane
            named = offset < COLUMNS
            entry = tl.load(
                selection + token * COLUMNS + offset, mask=named, other=-1
            )
            live = named & (entry >= 0)
            entry = tl.where(live, entry, 0)
            column = (entry // ENTRIES) % compressed_pages
            page = tl.load(
                compressed_table + row * compressed_pages + column,
                mask=live,
                other=0,
            )
            record = page.to(tl.int64) * ENTRIES + entry % ENTRIES
            latent = _decoded(
                compressed,
                record,
                live,
                COMPRESSED_FORMAT,
                COMPRESSED_RECORD,
                HEAD_DIM,
            )
            peak, total, acc = _attended(
                q, latent, live, scale, peak, total, acc
            )

    # ---- The sink joins the denominator under the final maximum. ----
    logit = tl.load(sink + head_index, mask=head_live, other=0.0)
    total += tl.exp(logit - peak)
    attended = acc / total[:, None]
    tl.store(
        out + rows_offset,
        attended.to(out.dtype.element_ty),
        mask=head_live[:, None],
    )


def _record(name, head_dim):
    """A record's extent in its plane's elements: BF16 elements for ``bf16``,
    bytes for a packed format (``Format::plane``).
    """  # noqa: D205
    if name == "bf16":
        return head_dim
    values = head_dim if name == "mxfp8" else head_dim // 2
    return values + head_dim // SCALE_BLOCK[name]


def derive(fields, parameters, target):
    """The head tile, the entry tile, both planes' codes and record extents,
    and the warps.

    Raises:
        SystemExit: When the block does not divide the heads, or a format's
            scale block does not divide the latent.
    """  # noqa: D205
    del parameters, target
    heads, block, head_dim = (
        fields["heads"],
        fields["block"],
        fields["head_dim"],
    )
    if block < 1 or heads % block != 0:
        raise SystemExit(f"a block of {block} heads does not divide {heads}")

    compressed = fields["compressed"]
    for name in (fields["window_format"], compressed):
        if (
            name != "none"
            and SCALE_BLOCK[name]
            and head_dim % SCALE_BLOCK[name]
        ):
            raise SystemExit(
                f"{name} records of {head_dim} values: the scale block "
                f"{SCALE_BLOCK[name]} does not divide them"
            )

    side = compressed != "none"
    return {
        "head_tile": max(MIN_ROWS, triton.next_power_of_2(block)),
        "tile": ENTRY_TILE,
        "window_code": FORMATS[fields["window_format"]],
        "window_record": _record(fields["window_format"], head_dim),
        "compressed_side": side,
        # The window-only program never reads the compressed side; its codes
        # are the BF16 ones so the compile-time values stay well formed.
        "compressed_code": FORMATS[compressed] if side else 0,
        "compressed_record": _record(compressed, head_dim)
        if side
        else head_dim,
        "warps": 8 if head_dim >= 512 else 4,
    }
