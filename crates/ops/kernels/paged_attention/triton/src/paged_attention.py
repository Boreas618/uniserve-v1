"""The paged attention's Triton programs, which read a page of any cache type.

The operator, its descriptor and its refusals are
`ops::attention::PagedAttention`'s (`crates/ops/src/attention/mod.rs`). These
programs serve the caches the vendor programs do not: a page of `E4M3` bytes
with **one f32 scale for the whole page** (`QUANTIZED`), and a dense page (BF16,
F16 or F32) of another type than the queries, such as a BF16 model's FP16 cache,
which has no scale and whose scale pointers are null and never read.

Both programs compute the operator's attention over the decoded page,
`x = f32(element) * s` with `s = key_scale[page]` (or the value plane's own) for
a quantized page and `x = f32(element)` for a dense one:
`Q · Kᵀ · softmax_scale`, an online softmax and the `P · V` sum, accumulated in
f32 and rounded once into the output's type. **Nothing is written back**: the
page stays E4M3 and the next step decodes it again. Three things neither program
may do, each of them a wrong number rather than a crash:

* re-quantize a decoded value (a read that changes the cache);
* round a decoded key or value, or a query, through a type that does not hold
  it, which would add an error the operator's tolerance does not claim;
* accumulate in E4M3 or in a 16-bit type — the accumulators are f32 whatever the
  page holds.

**Decode** is one program per (query row, query head): it walks the row's pages
once, carrying a running maximum, a running normaliser and a `[HEAD_DIM]`
accumulator, and decodes each element as `f32(byte) * s` before its product. The
key-value head is the query head divided by the group, so a call whose `q_heads`
is not a whole multiple of `kv_heads` is the launch's refusal, not this
program's rounding.

**Prefill** is one program per tile of `BLOCK_Q` consecutive query tokens of one
batch row and one key-value head, holding the `GROUP` query heads that share it:
a `[BLOCK_Q · GROUP_PAD, HEAD_DIM]` query tile whose row `m` is token
`m // GROUP_PAD` and head `m % GROUP_PAD` of the group (`GROUP_PAD` is the group
rounded up to a power of two; its extra rows are masked). It walks the row's
keys `TILE` at a time up to the last key its last query sees, finds each key's
page in the row's table, and multiplies each key tile once against the whole
query tile, as FlashAttention does. `_paged_attention_prefill_kernel` states how
a tile finds its row and what its products round.

A key a query does not see contributes nothing. A slot past a row's length was
written by an earlier request or never, and an FP16 cache of a BF16 model holds
an infinity where such a value overflowed FP16: decode replaces a hidden lane's
value by zero before the product (`0 · inf` is NaN), and prefill never loads a
key past the last one its tile sees.
"""

import triton
import triton.language as tl

#: The most (query token, query head) pairs one prefill program holds, its
#: `BLOCK_Q · GROUP_PAD`, and the most elements its query tile holds, which
#: lowers the pairs for a wide head (sixty-four at a head of 256, thirty-two at
#: 512) so that the tile, its accumulator and the key tile fit one
#: multiprocessor without spilling. A module records the `BLOCK_Q` they make
#: among its compile-time definitions, from which the operator's launch counts
#: the grid.
TILE_ROWS = 128
TILE_ELEMENTS = 16384

#: The key tile's elements, `TILE · HEAD_DIM`: sixty-four keys of a head of 128,
#: thirty-two of a head of 256, sixteen of 512, so that the tile, its pipelined
#: successor and the tensor-memory accumulators fit one multiprocessor whatever
#: the head.
KEY_TILE_ELEMENTS = 8192

#: The prefill program's compile options: eight warps over its
#: `[TILE_ROWS, head_dim]` query tile, and two software-pipelining stages, the
#: next key tile's loads in flight during this one's products.
PREFILL_OPTIONS = {"num_warps": 8, "num_stages": 2}

#: The (query, cache) element types, by the names a descriptor's fields give
#: them (`ops::abi`'s), whose values one 16-bit type holds exactly, so that the
#: tensor cores multiply them exactly: an E4M3 value has four significant bits
#: and an exponent in `[-9, 8]`, so BF16 and F16 both hold it, and a BF16 cache
#: is the BF16 query's own type. Any other pair is multiplied on the CUDA cores
#: in f32, one key at a time.
TENSOR_CORE_PAIRS = {("bf16", "f8e4m3"), ("bf16", "bf16"), ("f16", "f8e4m3")}


def prefill_tiles(q_heads, kv_heads, head_dim):
    """The prefill program's tile constexprs.

    ``BLOCK_Q`` is the query tokens one program holds: the tile's pairs
    (``TILE_ROWS``, fewer for a wide head, at least sixteen) divided among
    ``GROUP_PAD``, the head group ``q_heads // kv_heads`` rounded up to a power
    of two (a tile's row count must be one), and one token when the group is
    wider. ``TILE`` is the keys one iteration reads.
    """
    group_pad = triton.next_power_of_2(q_heads // kv_heads)
    rows = max(16, min(TILE_ROWS, TILE_ELEMENTS // head_dim))
    return {
        "BLOCK_Q": max(1, rows // group_pad),
        "GROUP_PAD": group_pad,
        "TILE": max(16, KEY_TILE_ELEMENTS // head_dim),
    }


def prefill_products(query, plane):
    """The prefill program's product constexprs for a query and cache type.

    The types are named as the descriptor's fields name them.

    ``TENSOR_CORES`` multiplies on the tensor cores (`TENSOR_CORE_PAIRS`), with
    the scores in F16 for F16 queries (``F16_SCORES``) and in BF16 otherwise.
    """
    return {
        "TENSOR_CORES": (query, plane) in TENSOR_CORE_PAIRS,
        "F16_SCORES": query == "f16",
    }


@triton.jit
def _paged_attention_decode_kernel(
    out_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_ptr,
    page_table_ptr,
    seq_lens_ptr,
    q_stride_tokens,
    q_stride_heads,
    q_stride_dim,
    k_stride_pages,
    k_stride_tokens,
    k_stride_heads,
    k_stride_dim,
    v_stride_pages,
    v_stride_tokens,
    v_stride_heads,
    v_stride_dim,
    o_stride_tokens,
    o_stride_heads,
    o_stride_dim,
    max_pages,
    page_size,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    QUANTIZED: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP
    dims = tl.arange(0, HEAD_DIM)
    lanes = tl.arange(0, PAGE)

    query = tl.load(
        query_ptr
        + row * q_stride_tokens
        + head * q_stride_heads
        + dims * q_stride_dim
    ).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + row)
    pages = tl.cdiv(seq_len, PAGE)
    out_offsets = (
        out_ptr
        + row * o_stride_tokens
        + head * o_stride_heads
        + dims * o_stride_dim
    )
    # A row with no keys is a padded row of a bucket wider than its batch, and
    # it is a **valid token whose output is discarded** - not a row with no
    # route: the capture runs the router over every token the bucket holds, so
    # this row's answer is read by the MoE block whether or not anyone serves
    # it. `0/0` would hand that block a NaN, and a NaN picks an expert from no
    # number and hands the sort an index nothing can serve, which faults one
    # kernel later. The inert value is written instead, so the route the padding
    # produces is a route of the right arity.
    if seq_len == 0:
        padded = (
            tl.zeros((HEAD_DIM,), tl.bfloat16)
            if OUT_BF16
            else tl.zeros((HEAD_DIM,), tl.float32)
        )
        tl.store(out_offsets, padded)
        return

    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)

    for index in range(0, pages):
        page = tl.load(page_table_ptr + row * max_pages + index)
        key_offsets = (
            key_ptr
            + page * k_stride_pages
            + lanes[:, None] * k_stride_tokens
            + kv_head * k_stride_heads
            + dims[None, :] * k_stride_dim
        )
        value_offsets = (
            value_ptr
            + page * v_stride_pages
            + lanes[:, None] * v_stride_tokens
            + kv_head * v_stride_heads
            + dims[None, :] * v_stride_dim
        )
        keys = tl.load(key_offsets)
        values = tl.load(value_offsets)
        if QUANTIZED:
            # One multiply per element, in f32, and the decoded value is never
            # written back.
            key_scale = tl.load(key_scale_ptr + page)
            value_scale = tl.load(value_scale_ptr + page)
            keys = keys.to(tl.float32) * key_scale
            values = values.to(tl.float32) * value_scale
        else:
            keys = keys.to(tl.float32)
            values = values.to(tl.float32)

        scores = tl.sum(keys * query[None, :], axis=1) * softmax_scale
        # The last page is partial: the keys past the row's length are not
        # attended to, and a masked score cannot win the maximum or the
        # normaliser. Their values are zeroed, so a stale non-finite value meets
        # a zero weight as zero rather than as NaN.
        live = lanes < seq_len - index * PAGE
        scores = tl.where(live, scores, float("-inf"))
        values = tl.where(live[:, None], values, 0.0)

        page_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, page_max)
        weights = tl.exp(scores - new_max)
        correction = tl.exp(running_max - new_max)
        running_sum = running_sum * correction + tl.sum(weights, axis=0)
        accumulator = accumulator * correction + tl.sum(
            weights[:, None] * values, axis=0
        )
        running_max = new_max

    result = accumulator / running_sum
    if OUT_BF16:
        tl.store(out_offsets, result.to(tl.bfloat16))
    else:
        tl.store(out_offsets, result)


@triton.jit
def _first_token(cum_seq_lens_q_ptr, row, max_q_len, packed):
    """The first query token of `row`.

    Its packed offset, or `row * max_q_len` in an unpacked batch.
    """
    if packed != 0:
        first = tl.load(cum_seq_lens_q_ptr + row)
    else:
        first = row * max_q_len
    return first


@triton.jit
def _first_tile(
    cum_seq_lens_q_ptr, row, max_q_len, packed, BLOCK_Q: tl.constexpr
):
    """The number of `row`'s first query tile.

    Row `r` starts at tile `first(r) // BLOCK_Q + r`. Between two rows' starts
    lie `first(r + 1) // BLOCK_Q - first(r) // BLOCK_Q + 1` numbers, at least
    the `ceil(length / BLOCK_Q)` tiles row `r` needs (its surplus tiles find no
    query and return), and the starts strictly increase with `r`, so a tile's
    row is found by bisection. Row `rows` is the padding past the last row's
    end, whose tiles write zeros.
    """
    return (
        _first_token(cum_seq_lens_q_ptr, row, max_q_len, packed) // BLOCK_Q
        + row
    )


@triton.jit
def _tile_row(
    cum_seq_lens_q_ptr, tile, rows, max_q_len, packed, BLOCK_Q: tl.constexpr
):
    """The last row in `[0, rows]` whose first tile is at most `tile`, or -1."""
    low: tl.int32 = 0
    high = rows + 1
    while low < high:
        middle = (low + high) // 2
        if (
            _first_tile(cum_seq_lens_q_ptr, middle, max_q_len, packed, BLOCK_Q)
            <= tile
        ):
            low = middle + 1
        else:
            high = middle
    return low - 1


@triton.jit
def _attend(
    query,
    own,
    running_max,
    running_sum,
    accumulator,
    key_start,
    end,
    table,
    key_heads,
    value_heads,
    key_scale_ptr,
    value_scale_ptr,
    k_stride_pages,
    k_stride_tokens,
    v_stride_pages,
    v_stride_tokens,
    softmax_scale,
    dims,
    lanes,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    QUANTIZED: tl.constexpr,
    F16_SCORES: tl.constexpr,
    MASKED: tl.constexpr,
):
    """One key tile of the online softmax, on the tensor cores.

    Answers the running maximum, normaliser and accumulator after the keys
    `key_start .. key_start + TILE`.

    `key_heads` and `value_heads` are the planes' bases at the tile's key-value
    head. An unmasked tile (`MASKED` false) lies wholly before every live pair's
    own key and below `end`, so it needs neither the causal mask nor masked
    loads; a masked tile loads only its keys below `end` and scores a key after
    a pair's own key `-inf`.
    """
    keys = key_start + lanes
    present = keys < end
    if TILE <= PAGE:
        # The tile lies in one page, whose entry is the row's
        # (`key_start < end <= seq_len`): one table entry and one scale per
        # plane.
        page = tl.load(table + key_start // PAGE).to(tl.int64)
    elif MASKED:
        # A tile of several pages past `end` would read table entries past the
        # row's own.
        page = tl.load(table + keys // PAGE, mask=present, other=0).to(tl.int64)
    else:
        page = tl.load(table + keys // PAGE).to(tl.int64)
    slot = keys % PAGE
    key_rows = page * k_stride_pages + slot * k_stride_tokens
    value_rows = page * v_stride_pages + slot * v_stride_tokens
    if MASKED:
        key = tl.load(
            key_heads + key_rows[:, None] + dims[None, :],
            mask=present[:, None],
            other=0.0,
        )
        value = tl.load(
            value_heads + value_rows[:, None] + dims[None, :],
            mask=present[:, None],
            other=0.0,
        )
    else:
        key = tl.load(key_heads + key_rows[:, None] + dims[None, :])
        value = tl.load(value_heads + value_rows[:, None] + dims[None, :])

    # The query times the key bytes, exactly product by product and summed in
    # f32: the decoded key is the byte times its page's scale, which multiplies
    # the sum once below.
    if F16_SCORES:
        scores = tl.dot(query, tl.trans(key.to(tl.float16)))
    else:
        scores = tl.dot(query, tl.trans(key.to(tl.bfloat16)))
    if QUANTIZED:
        key_scale = tl.load(key_scale_ptr + page)
        if TILE <= PAGE:
            scores = scores * (key_scale * softmax_scale)
        else:
            scores = scores * (key_scale * softmax_scale)[None, :]
    else:
        scores = scores * softmax_scale
    if MASKED:
        # A key after a pair's own key is in its future, and a key not loaded is
        # after every live pair's: both score `-inf` and weigh exactly zero.
        scores = tl.where(keys[None, :] <= own[:, None], scores, float("-inf"))

    new_max = tl.maximum(running_max, tl.max(scores, axis=1))
    # A pair that has seen no key yet keeps a `-inf` maximum; shifting by zero
    # instead keeps its weights `exp(-inf) = 0` rather than
    # `exp(-inf - -inf) = NaN`. `tl.exp` is the hardware's `ex2.approx` of
    # `x · log2 e`, within a few f32 roundings of `exp(x)`.
    shift = tl.where(new_max == float("-inf"), 0.0, new_max)
    weights = tl.exp(scores - shift[:, None])
    correction = tl.exp(running_max - shift)
    running_sum = running_sum * correction + tl.sum(weights, axis=1)
    if QUANTIZED:
        # The value's page scale multiplies its weight, so the product reads the
        # value bytes.
        value_scale = tl.load(value_scale_ptr + page)
        if TILE <= PAGE:
            weights = weights * value_scale
        else:
            weights = weights * value_scale[None, :]

    # The f32 weight as two BF16 parts, the weight rounded and its remainder
    # rounded, which hold it to sixteen significant bits; the parts sit side by
    # side along the contraction and the value tile twice, so one product
    # multiplies each part by its value exactly and sums both in f32. (One
    # product, not one per part: each product holds its own tensor-memory
    # accumulator, and one per part would not fit beside the scores' at every
    # declared head.)
    high = weights.to(tl.bfloat16)
    low = (weights - high.to(tl.float32)).to(tl.bfloat16)
    parts = tl.reshape(tl.join(high, low), (weights.shape[0], 2 * TILE))
    decoded = value.to(tl.bfloat16)
    twice = tl.permute(tl.join(decoded, decoded), (0, 2, 1))
    accumulator = accumulator * correction[:, None]
    accumulator = tl.dot(
        parts, tl.reshape(twice, (2 * TILE, value.shape[1])), accumulator
    )
    return new_max, running_sum, accumulator


@triton.jit
def _attend_key(
    query,
    own,
    running_max,
    running_sum,
    accumulator,
    key,
    table,
    key_heads,
    value_heads,
    key_scale_ptr,
    value_scale_ptr,
    k_stride_pages,
    k_stride_tokens,
    v_stride_pages,
    v_stride_tokens,
    softmax_scale,
    dims,
    PAGE: tl.constexpr,
    QUANTIZED: tl.constexpr,
):
    """One key of the online softmax in f32 on the CUDA cores, for every pair.

    The decode program's arithmetic: the key and the value decoded once,
    `f32(element) * s`, every product and sum in f32, and a value the pair does
    not see replaced by zero before its product, so that a live slot holding an
    infinity reaches only the pairs that see it.
    """
    page = tl.load(table + key // PAGE).to(tl.int64)
    slot = key % PAGE
    decoded_key = tl.load(
        key_heads + page * k_stride_pages + slot * k_stride_tokens + dims
    )
    decoded_value = tl.load(
        value_heads + page * v_stride_pages + slot * v_stride_tokens + dims
    )
    decoded_key = decoded_key.to(tl.float32)
    decoded_value = decoded_value.to(tl.float32)
    if QUANTIZED:
        decoded_key = decoded_key * tl.load(key_scale_ptr + page)
        decoded_value = decoded_value * tl.load(value_scale_ptr + page)

    scores = tl.sum(query * decoded_key[None, :], axis=1) * softmax_scale
    visible = key <= own
    scores = tl.where(visible, scores, float("-inf"))

    new_max = tl.maximum(running_max, scores)
    shift = tl.where(new_max == float("-inf"), 0.0, new_max)
    weights = tl.exp(scores - shift)
    correction = tl.exp(running_max - shift)
    running_sum = running_sum * correction + weights
    values = tl.where(visible[:, None], decoded_value[None, :], 0.0)
    accumulator = accumulator * correction[:, None] + weights[:, None] * values
    return new_max, running_sum, accumulator


@triton.jit
def _paged_attention_prefill_kernel(
    out_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_ptr,
    page_table_ptr,
    seq_lens_ptr,
    cum_seq_lens_q_ptr,
    q_stride_tokens,
    q_stride_heads,
    q_stride_dim,
    k_stride_pages,
    k_stride_tokens,
    k_stride_heads,
    k_stride_dim,
    v_stride_pages,
    v_stride_tokens,
    v_stride_heads,
    v_stride_dim,
    o_stride_tokens,
    o_stride_heads,
    o_stride_dim,
    max_pages,
    page_size,
    softmax_scale,
    max_q_len,
    packed,
    tokens,
    rows,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    QUANTIZED: tl.constexpr,
    OUT_BF16: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    TILE: tl.constexpr,
    TENSOR_CORES: tl.constexpr,
    F16_SCORES: tl.constexpr,
):
    """One tile of a batch row's query tokens, over the keys they see, causally.

    **The rows.** A batch row's queries are the last `length` of its `seq_len`
    keys: the cache already holds them from `kv_write`, after any prefix an
    earlier step or the prefix cache left there, so query `position` of the row
    sits at key `seq_len - length + position` and sees the keys up to it. A
    packed batch states each row's first query token in `cum_seq_lens_q`; an
    unpacked one gives every row `max_q_len` tokens, so the row's first token is
    `row * max_q_len`. Either way the output row is `first + position`, which is
    what the operator's `[tokens, q_heads, head_dim]` view means.

    **The grid** is `(cdiv(tokens, BLOCK_Q) + rows, kv_heads)`: axis 0 numbers
    the query tiles of every row (`_first_tile`) and then the padding's, which
    is at least as many as they need (`first(rows) // BLOCK_Q + rows` of the
    rows', `cdiv(tokens - first(rows), BLOCK_Q)` of the padding's, whose sum is
    at most `cdiv(tokens, BLOCK_Q) + rows`), and the numbers past the padding
    return at once. The heaviest tiles of a causal row are its last, so the
    numbers run backwards from the end of the grid and the long tiles start
    first.

    **The padding.** This bucket's graph runs the router over all of its tokens,
    so a token no row owns must be written: left untouched it holds whatever the
    arena held - stale bytes from the previous replay - and the route computed
    from them indexes a table nothing can serve (an illegal address inside the
    MoE gather, one kernel later). The tokens no live row owns are those from
    the last row's end (`first(rows)`; the padded rows of a bucket have length 0
    and start there) to `tokens`, and the padding's tiles write each of them
    once, as zero. A query that sees no key - a row stating more queries than
    keys, as a warm-up's idle columns do - writes zero too, the inert value the
    decode program writes for a row with no keys.

    **The products.** On the tensor cores (`TENSOR_CORE_PAIRS`) the query and
    the key bytes are both values of the product's 16-bit type, so every
    `q_d · byte_d` is exact in f32 and the tensor core sums the score in f32;
    the page's scale then multiplies the score once, `s · Σ q · byte`, which
    equals `Σ q · (byte · s)` in exact arithmetic, the two orders differing by
    roundings of f32 alone. The value plane's scale multiplies the weight,
    `(p_j · s_j) · byte_j`, and the f32 weight enters the product as two BF16
    parts that hold it to sixteen significant bits (a relative `2⁻¹⁶`, below the
    operator's `γₙ` for any cache of 256 keys or more), each multiplied exactly
    and summed in f32. Any other pair (a dense page whose elements the query's
    type does not hold) walks its keys one at a time on the CUDA cores in f32,
    with the decode program's arithmetic, for all the tile's pairs at once.
    (Neither Triton's `ieee` product nor its `tf32x3` serves those pairs: on
    sm_100 the first multiplies in one TF32 product whenever an operand is
    widened from a 16-bit type, rounding the other to eleven bits, and the
    second holds a tensor-memory accumulator per product, more than fit.)

    **Hidden keys.** A key past the last one the tile's last query sees is never
    loaded, so a stale slot past the row's length takes no part, finite or not.
    On the tensor cores a key the tile loads and one of its queries does not see
    yet scores `-inf`, so its weight is exactly zero, and `0 · v` is zero for
    every finite `v`; a live slot holding a non-finite value would make NaN of
    the tile's earlier queries as well as of those that see it, which neither an
    E4M3 page (it holds no infinity) nor a BF16 page of a BF16 model (it cannot
    overflow) holds. The CUDA-core walk replaces a value a pair does not see by
    zero, so an FP16 page's overflow reaches only the pairs that see it.
    """
    BLOCK_M: tl.constexpr = BLOCK_Q * GROUP_PAD
    # The operator's `check` holds every view to a sixteen-byte base and to
    # strides of whole sixteen-byte units, with a unit last stride. Rewriting
    # each stride as a multiple of its unit (which changes no value the check
    # admits) states that to the compiler, whose loads and stores then move
    # whole sixteen-byte vectors along the head dimension; a hint on the
    # argument itself would not reach them. The `*_stride_dim` words are that
    # unit last stride.
    QUERY_UNIT: tl.constexpr = (
        128 // query_ptr.dtype.element_ty.primitive_bitwidth
    )
    KEY_UNIT: tl.constexpr = 128 // key_ptr.dtype.element_ty.primitive_bitwidth
    OUT_UNIT: tl.constexpr = 128 // out_ptr.dtype.element_ty.primitive_bitwidth
    q_stride_tokens = q_stride_tokens // QUERY_UNIT * QUERY_UNIT
    q_stride_heads = q_stride_heads // QUERY_UNIT * QUERY_UNIT
    k_stride_pages = k_stride_pages // KEY_UNIT * KEY_UNIT
    k_stride_tokens = k_stride_tokens // KEY_UNIT * KEY_UNIT
    k_stride_heads = k_stride_heads // KEY_UNIT * KEY_UNIT
    v_stride_pages = v_stride_pages // KEY_UNIT * KEY_UNIT
    v_stride_tokens = v_stride_tokens // KEY_UNIT * KEY_UNIT
    v_stride_heads = v_stride_heads // KEY_UNIT * KEY_UNIT
    o_stride_tokens = o_stride_tokens // OUT_UNIT * OUT_UNIT
    o_stride_heads = o_stride_heads // OUT_UNIT * OUT_UNIT

    tile = tl.num_programs(0) - 1 - tl.program_id(0)
    kv_head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    pairs = tl.arange(0, BLOCK_M)
    # Row `m` of the tile: query token `m // GROUP_PAD` of the tile, query head
    # `m % GROUP_PAD` of the group; the rows past the group of a group that is
    # not a power of two are masked.
    member = pairs % GROUP_PAD
    head = kv_head * GROUP + member
    in_group = member < GROUP
    offset = pairs // GROUP_PAD

    row = _tile_row(cum_seq_lens_q_ptr, tile, rows, max_q_len, packed, BLOCK_Q)
    if row < 0:
        return
    first = _first_token(cum_seq_lens_q_ptr, row, max_q_len, packed)
    start = (tile - (first // BLOCK_Q + row)) * BLOCK_Q

    if row == rows:
        # A tile of the padding: the tokens from `first(rows)` to `tokens`,
        # written as zero.
        token = first + start + offset
        padded = in_group & (token < tokens)
        out_rows = token * o_stride_tokens + head * o_stride_heads
        zeros = tl.zeros(
            (BLOCK_M, HEAD_DIM), tl.bfloat16 if OUT_BF16 else tl.float32
        )
        tl.store(
            out_ptr + out_rows[:, None] + dims[None, :],
            zeros,
            mask=padded[:, None],
        )
        return

    length = (
        _first_token(cum_seq_lens_q_ptr, row + 1, max_q_len, packed) - first
    )
    if start >= length:
        return

    seq_len = tl.load(seq_lens_ptr + row)
    context = seq_len - length
    position = start + offset
    live = in_group & (position < length)
    token = first + position
    # Each pair's own key, the last it sees.
    own = context + position
    query = tl.load(
        query_ptr
        + (token * q_stride_tokens + head * q_stride_heads)[:, None]
        + dims[None, :],
        mask=live[:, None],
        other=0.0,
    )
    if not TENSOR_CORES:
        query = query.to(tl.float32)

    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # The keys up to the one the tile's last live query sees: no later key is
    # loaded.
    end = context + tl.minimum(start + BLOCK_Q, length)
    table = page_table_ptr + row * max_pages
    key_heads = key_ptr + kv_head * k_stride_heads
    value_heads = value_ptr + kv_head * v_stride_heads
    if TENSOR_CORES:
        # The tiles before the first live pair's own key are seen whole by every
        # live pair and run unmasked.
        whole = tl.maximum(context + start + 1, 0) // TILE * TILE
        lanes = tl.arange(0, TILE)
        for key_start in range(0, whole, TILE):
            running_max, running_sum, accumulator = _attend(
                query,
                own,
                running_max,
                running_sum,
                accumulator,
                key_start,
                end,
                table,
                key_heads,
                value_heads,
                key_scale_ptr,
                value_scale_ptr,
                k_stride_pages,
                k_stride_tokens,
                v_stride_pages,
                v_stride_tokens,
                softmax_scale,
                dims,
                lanes,
                PAGE,
                TILE,
                QUANTIZED,
                F16_SCORES,
                False,
            )
        for key_start in range(whole, end, TILE):
            running_max, running_sum, accumulator = _attend(
                query,
                own,
                running_max,
                running_sum,
                accumulator,
                key_start,
                end,
                table,
                key_heads,
                value_heads,
                key_scale_ptr,
                value_scale_ptr,
                k_stride_pages,
                k_stride_tokens,
                v_stride_pages,
                v_stride_tokens,
                softmax_scale,
                dims,
                lanes,
                PAGE,
                TILE,
                QUANTIZED,
                F16_SCORES,
                True,
            )
    else:
        for key in range(0, end):
            running_max, running_sum, accumulator = _attend_key(
                query,
                own,
                running_max,
                running_sum,
                accumulator,
                key,
                table,
                key_heads,
                value_heads,
                key_scale_ptr,
                value_scale_ptr,
                k_stride_pages,
                k_stride_tokens,
                v_stride_pages,
                v_stride_tokens,
                softmax_scale,
                dims,
                PAGE,
                QUANTIZED,
            )

    # A pair that saw no key has a zero accumulator and a zero normaliser: it
    # writes zero.
    normaliser = tl.where(running_sum > 0.0, running_sum, 1.0)
    result = accumulator / normaliser[:, None]
    out_rows = token * o_stride_tokens + head * o_stride_heads
    out_offsets = out_ptr + out_rows[:, None] + dims[None, :]
    if OUT_BF16:
        tl.store(out_offsets, result.to(tl.bfloat16), mask=live[:, None])
    else:
        tl.store(out_offsets, result, mask=live[:, None])


#: The decode program's software-pipelining stages: Triton's default on an
#: NVIDIA target (``CUDAOptions.num_stages``), the decode program stating none
#: of its own.
DECODE_STAGES = 3


def derive(fields, parameters, target):
    """The query group, the prefill tiles and products, and the phase's options.

    ``group`` is the query heads that share a key-value head; a ``q_heads``
    that is no whole multiple of ``kv_heads`` leaves a query head without one
    and is refused. A decode program takes ``DECODE_STAGES`` and eight warps
    for a ``[page, head_dim]`` block of 8192 elements or more, four below. A
    prefill program takes ``PREFILL_OPTIONS``, ``prefill_tiles`` and
    ``prefill_products``.
    """
    q_heads, kv_heads = fields["q_heads"], fields["kv_heads"]
    if q_heads % kv_heads != 0:
        raise SystemExit(
            f"q_heads={q_heads} is not a whole multiple of "
            f"kv_heads={kv_heads}: a query head has no key-value head"
        )
    derived = {"group": q_heads // kv_heads}

    if fields["phase"] == "decode":
        block = fields["page"] * fields["head_dim"]
        derived["warps"] = 8 if block >= 8192 else 4
        derived["stages"] = DECODE_STAGES
        return derived

    tiles = prefill_tiles(q_heads, kv_heads, fields["head_dim"])
    products = prefill_products(fields["dtype"], fields["kv"])
    derived.update(
        warps=PREFILL_OPTIONS["num_warps"],
        stages=PREFILL_OPTIONS["num_stages"],
        block_q=tiles["BLOCK_Q"],
        group_pad=tiles["GROUP_PAD"],
        tile=tiles["TILE"],
        tensor_cores=products["TENSOR_CORES"],
        f16_scores=products["F16_SCORES"],
    )
    return derived
