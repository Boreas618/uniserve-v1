"""The Triton paged attention's prefill against the float64 composition.

The program is launched as `ops::attention::PagedAttention` launches it: a
grid of `cdiv(tokens, BLOCK_Q) + rows` query tiles by `kv_heads`, the tile
constexprs of the package's `prefill_tiles` and `prefill_products`, and the
packed token count as `max_q_len`. Each case is a property this family's
defects would break:

- **the causal mask**: a batch row's queries are the last of its keys, so its
  query `p` of `n` over `s` keys attends keys `0..=s - n + p`; the first query
  of a row that holds no earlier keys answers over one key whatever the rest
  of the cache holds, and a row whose queries follow a prefix already in the
  cache sees that prefix;
- **the tiles**: a row longer than one query tile and one key tile; key tiles
  that span pages (a page of 32 keys), that are one page (64) and that lie
  within one (128); a head group that is not a power of two, whose padded tile
  rows must stay unwritten; and a head of 512;
- **a batch's rows are not all the same length**: a shorter row's tiles leave
  the next row's tokens alone, and the tokens of a padded bucket that no row
  owns read zero;
- **a query that sees no key**: a row stating more queries than keys, as a
  warm-up's idle columns do, writes zero;
- **the served pairs**: BF16 queries over an E4M3 cache into a BF16 output,
  and every dense pair the package declares, with hidden slots that hold
  infinities and NaN.

The reference is the composition the operator states, in float64 from the
same bytes::

    page_k = f64(byte) · key_scale[page]     (a dense page: f64(element))
    scores = (q · page_kᵀ) · softmax_scale   masked by j <= seq_len - n + p
    out    = Σ softmax(scores) · page_v      zero for a query that sees no key

**The bound.** Both sides read the same bytes and scales, so neither the
cache's own quantization nor the decode is part of this comparison (an E4M3
value has four significant bits and an exponent in `[-9, 8]`, so each one is a
BF16 value and the kernel's decode of it is exact). What remains is the
kernel's f32 arithmetic, bounded per query token and head from the case's own
numbers, with `u = 2⁻²⁴`, `D` the head dimension, `n` the keys the query sees,
`σ` the softmax scale, `A = σ · max_j Σ_d |q_d · k_jd|` the largest score
magnitude and `V = max_j,d |v_jd|` over the keys it sees:

1. the score: a pair the tensor cores serve is multiplied exactly, product by
   product, and summed in f32, where an addition may truncate (Fasi, Higham,
   Mikaitis and Pranesh, *Numerical behavior of NVIDIA tensor cores*, PeerJ CS
   7, e330, 2021) and so costs at most `2u`; any other pair is decoded and
   multiplied in f32 on the CUDA cores, a rounding per product and per
   addition. Either way, with the scale's multiplications, a score is within
   `ε = (γ_D(2u) + 3u) · A` of exact, where `γ_k(v) = k·v / (1 − k·v)`;
2. a weight `exp(s_j − m)` takes `ε`, the subtraction and the `log2 e`
   multiplication inside `tl.exp` (`2u · 2A` together), `ex2.approx`'s
   relative `2⁻²² = 4u`, and the online softmax's `c` corrections of earlier
   keys (`c = ⌈n/TILE⌉` key tiles on the tensor cores, `c = n` keys on the CUDA
   cores): a relative error of at most `η = exp(ε + 12u·A + 5u·(c + 1)) − 1`;
3. weights perturbed by a relative `η` move a weighted mean of values within
   `V` of each other by at most `2η/(1 − η) · V`;
4. the weighted sum: on the tensor cores the f32 weight enters as two BF16
   parts that hold it to a relative `e_w = 2⁻¹⁶` (`e_w = 0` on the CUDA cores);
   the normaliser and the accumulator are f32 sums of `N = n + 2c` terms,
   `γ_N(2u)` each, with one rounding of the weight's product and one division:
   `(e_w + 2γ_N(2u) + 2u) · V`;
5. the store rounds once into the output's type: `u_out · |out|` (BF16 `2⁻⁸`
   where a device rounds to nearest and `2⁻⁷` where the interpreter truncates,
   F16 `2⁻¹¹`, F32 `u`).

With an f32 output the allowance is about `2 · 10⁻⁵ · V` for these cases, so a
program that rounded a weight, a dense key or a query to BF16 before its
product (a relative `2⁻⁹`) would exceed it.

Triton's interpreter multiplies a BF16 tile by its bit patterns rather than
its values, so the cases whose pair the package multiplies on the tensor cores
(its `TENSOR_CORE_PAIRS`) need a device, and the interpreter runs the cases of
the f32 walk: it proves the rule, its order and its masking.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit

Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
PAGE = 32
SET = 0.0625

UNIT = 2.0**-24
#: The unit roundoff of an F16 row, which both modes round to nearest.
HALF_ROUNDOFF = 2.0**-11
#: The relative error of a weight held as two BF16 parts, each rounded to
#: nearest: `2⁻⁸ · 2⁻⁸`.
WEIGHT_PARTS_ERROR = 2.0**-16
#: Triton's names of the element types, by which the package states its
#: products.
TRITON_NAMES = {
    torch.float8_e4m3fn: "fp8e4nv",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}


def output_roundoff(dtype, interpreted):
    """The store's rounding into `dtype`: a BF16 store rounds one step where a
    device rounds to nearest and two where the interpreter truncates.
    """  # noqa: D205
    if dtype == torch.bfloat16:
        return 2.0**-7 if interpreted else 2.0**-8
    return {torch.float32: UNIT, torch.float16: HALF_ROUNDOFF}[dtype]


def cache(rows, generator, queries, plane, hidden, page, heads, head_dim):
    """A cache of as many pages as the rows' keys need, each row's own, and
    each row's first token.

    `rows` are the rows' key counts and `queries` their query counts, the last
    of their keys. An E4M3 cache has a scale per page; a `plane` names a dense
    cache's element type, which has none, and `hidden` fills its slots past
    each row's length.
    """  # noqa: D205
    kv_heads = heads[1]
    pages = sum((length + page - 1) // page for length in rows) + 1
    width = max(1, max((length + page - 1) // page for length in rows))
    table = torch.zeros((len(rows), width), dtype=torch.int32)
    first = [0]
    # Page 0 is left to no row, so a table entry read past a row's own names a
    # page whose keys change the answer.
    start = 1
    for index, length in enumerate(rows):
        live = (length + page - 1) // page
        table[index, :live] = torch.arange(
            start, start + live, dtype=torch.int32
        )
        start += live
        first.append(first[-1] + queries[index])
    shape = (pages, page, kv_heads, head_dim)
    key_values = torch.rand(shape, generator=generator) * 2.0 - 1.0
    value_values = torch.rand(shape, generator=generator) * 2.0 - 1.0
    state = {
        "page_table": table,
        "seq_lens": torch.tensor(rows, dtype=torch.int32),
        "cum_seq_lens_q": torch.tensor(first, dtype=torch.int32),
    }
    if plane is not None:
        keys = key_values.to(plane)
        values = value_values.to(plane)
        if hidden is not None:
            for row, length in enumerate(rows):
                if length == 0:
                    continue
                last = int(table[row, (length - 1) // page])
                keys[last, length % page or page :] = hidden
                values[last, length % page or page :] = hidden
        return {
            **state,
            "key": keys,
            "value": values,
            "key_scale": None,
            "value_scale": None,
        }
    cycle = [1.0, 0.5, 2.0, 0.25]
    scales = torch.tensor(
        [cycle[index % len(cycle)] for index in range(pages)],
        dtype=torch.float32,
    ).view(-1, 1, 1, 1)
    return {
        **state,
        "key": (key_values / scales).to(torch.float8_e4m3fn),
        "value": (value_values / scales).to(torch.float8_e4m3fn),
        "key_scale": scales,
        "value_scale": scales.clone(),
    }


def operands(
    rows,
    generator,
    out_dtype=torch.float32,
    padding=0,
    queries=None,
    plane=None,
    hidden=None,
    query_dtype=torch.float32,
    page=PAGE,
    heads=(Q_HEADS, KV_HEADS),
    head_dim=HEAD_DIM,
):
    """The query, the cache and the output buffer, in the layouts the operator
    states.

    `queries` are the rows' query counts, each row's whole key count unless
    stated. `padding` tokens follow the live ones, as a bucket larger than its
    batch holds them. The output starts as NaN everywhere, so a token the
    kernel failed to write cannot pass as a zero.
    """  # noqa: D205
    queries = rows if queries is None else queries
    total = sum(queries) + padding
    tokens = (
        torch.rand((total, heads[0], head_dim), generator=generator) * 2.0 - 1.0
    ).to(query_dtype)
    out = torch.full((total, heads[0], head_dim), float("nan"), dtype=out_dtype)
    # At prefill the operator passes the packed token count as `max_q_len`
    # (`crates/ops/src/attention/mod.rs`).
    return {
        "query": tokens,
        "out": out,
        **cache(rows, generator, queries, plane, hidden, page, heads, head_dim),
        "max_q_len": total,
        "page": page,
        "heads": heads,
        "head_dim": head_dim,
    }


def reference(operands_of):
    """The operator's composition in float64, causally masked per row, with
    what the bound needs.

    Answers the output and, per query token and head, the largest score
    magnitude `A`, the largest value magnitude `V` and the count `n` of the
    keys the query sees.
    """  # noqa: D205
    q_heads, kv_heads = operands_of["heads"]
    page = operands_of["page"]
    head_dim = operands_of["head_dim"]
    query = operands_of["query"].to(torch.float64)
    key = operands_of["key"].to(torch.float64)
    value = operands_of["value"].to(torch.float64)
    if operands_of["key_scale"] is not None:
        key = key * operands_of["key_scale"].to(torch.float64)
        value = value * operands_of["value_scale"].to(torch.float64)
    table = operands_of["page_table"]
    first = operands_of["cum_seq_lens_q"].tolist()
    # A token no row owns is padding, and the kernel writes it as zero; so is a
    # query that sees no key.
    out = torch.zeros_like(query)
    largest_score = torch.zeros(query.shape[:2], dtype=torch.float64)
    largest_value = torch.zeros(query.shape[:2], dtype=torch.float64)
    seen = torch.zeros(query.shape[:2], dtype=torch.int64)
    for row, seq_len in enumerate(operands_of["seq_lens"].tolist()):
        live = (seq_len + page - 1) // page
        pages = table[row, :live].tolist()
        keys = key[pages].reshape(-1, kv_heads, head_dim)
        values = value[pages].reshape(-1, kv_heads, head_dim)
        length = first[row + 1] - first[row]
        for position in range(length):
            # The query's own key: the row's queries are the last of its keys,
            # and the keys it sees are those up to it.
            own = seq_len - length + position
            if own < 0:
                continue
            token = first[row] + position
            for head in range(q_heads):
                kv_head = head // (q_heads // kv_heads)
                visible_keys = keys[: own + 1, kv_head, :]
                visible_values = values[: own + 1, kv_head, :]
                scores = (visible_keys @ query[token, head, :]) * SET
                out[token, head, :] = (
                    torch.softmax(scores, dim=0) @ visible_values
                )
                magnitudes = visible_keys.abs() @ query[token, head, :].abs()
                largest_score[token, head] = SET * magnitudes.max()
                largest_value[token, head] = visible_values.abs().max()
                seen[token, head] = own + 1
    return out, largest_score, largest_value, seen


def gamma(terms, unit):
    """`γₖ = k·u / (1 − k·u)`."""
    product = terms * unit
    return math.inf if product >= 1.0 else product / (1.0 - product)


def allowance(
    largest_score, largest_value, seen, expected, roundoff, tile, tensor_cores
):
    """The module documentation's bound, per element, for a head of
    `expected`'s width and a store of unit roundoff `roundoff`.

    `tensor_cores` states that the pair multiplies on the tensor cores, a key
    tile at a time with the weight as two BF16 parts; otherwise one key at a
    time in f32.
    """  # noqa: D205
    head_dim = expected.shape[2]
    weight_error = WEIGHT_PARTS_ERROR if tensor_cores else 0.0
    corrections = (seen + tile - 1) // tile if tensor_cores else seen
    sums = seen + 2 * corrections
    score_error = (gamma(head_dim, 2.0 * UNIT) + 3.0 * UNIT) * largest_score
    exponent = (
        score_error
        + 12.0 * UNIT * largest_score
        + 5.0 * UNIT * (corrections + 1).to(torch.float64)
    )
    eta = torch.expm1(exponent)
    sum_error = torch.tensor(
        [gamma(int(count), 2.0 * UNIT) for count in sums.flatten().tolist()],
        dtype=torch.float64,
    ).view(sums.shape)
    absolute = (
        2.0 * eta / (1.0 - eta) + weight_error + 2.0 * sum_error + 2.0 * UNIT
    ) * largest_value
    return absolute[:, :, None] + roundoff * expected.abs()


def launch(kernels, operands_of, device, interpreted):
    """One prefill as the operator launches it; answers what it wrote, on the
    host.
    """  # noqa: D205
    q_heads, kv_heads = operands_of["heads"]
    head_dim = operands_of["head_dim"]
    page = operands_of["page"]
    on = {
        name: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
        for name, tensor in operands_of.items()
    }
    query, key, value, out = on["query"], on["key"], on["value"], on["out"]
    tokens = query.shape[0]
    rows = on["page_table"].shape[0]
    tiles = kernels.prefill_tiles(q_heads, kv_heads, head_dim)
    products = kernels.prefill_products(
        TRITON_NAMES[query.dtype], TRITON_NAMES[key.dtype]
    )
    grid = (
        (tokens + tiles["BLOCK_Q"] - 1) // tiles["BLOCK_Q"] + rows,
        kv_heads,
    )
    kernels._paged_attention_prefill_kernel[grid](
        out,
        query,
        key,
        value,
        on["key_scale"],
        on["value_scale"],
        on["page_table"],
        on["seq_lens"],
        on["cum_seq_lens_q"],
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        on["page_table"].shape[1],
        page,
        SET,
        on["max_q_len"],
        1,
        tokens,
        rows,
        HEAD_DIM=head_dim,
        PAGE=page,
        GROUP=q_heads // kv_heads,
        QUANTIZED=on["key_scale"] is not None,
        OUT_BF16=out.dtype == torch.bfloat16,
        **tiles,
        **products,
        # The package's compile options; the interpreter has no warps or
        # stages.
        **({} if interpreted else kernels.PREFILL_OPTIONS),
    )
    return out.cpu()


#: The element types by the names the case ids use.
NAMES = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}

#: A case the package multiplies on the tensor cores, which the interpreter
#: cannot run.
TENSOR_CORES = pytest.mark.gpu

#: The served pair: BF16 queries over the E4M3 cache, into BF16.
SERVED = {"out_dtype": torch.bfloat16, "query_dtype": torch.bfloat16}

#: The cases: the rows' key counts and what `operands` reads.
CASES = [
    pytest.param([40], {}, id="a causal prefill over two pages"),
    pytest.param([70], {}, id="a partial last page under the mask"),
    # The shorter row comes first, so its tiles past its length fall on the
    # second row's live tokens: no row's tile may write another row's tokens.
    pytest.param([24, 40], {}, id="a batch of two rows with different lengths"),
    # A bucket larger than its batch: a padded row of length 0 and tokens no
    # row owns, which the kernel must write as zero.
    pytest.param([24, 16, 0], {"padding": 40}, id="a padded bucket"),
    pytest.param([40], {"out_dtype": torch.bfloat16}, id="the narrowed store"),
    # Rows whose queries follow keys already in the cache: 9 queries after 61
    # keys across a page boundary, and 5 after 30 within the first page,
    # beside a row of no prefix.
    pytest.param(
        [70, 35, 12],
        {"queries": [9, 5, 12]},
        id="queries after a cached prefix",
    ),
    # The served pair over rows of several query tiles and several key tiles,
    # with key tiles of two pages, one page and half a page: a row of no
    # prefix, a row whose queries follow a long prefix (its first tiles before
    # the diagonal are unmasked), and padding.
    *(
        pytest.param(
            [300, 260, 0],
            {"queries": [300, 70, 0], "padding": 13, "page": page, **SERVED},
            id=f"bf16 queries over an e4m3 cache of {page}-key pages, "
            "into bf16",
            marks=TENSOR_CORES,
        )
        for page in (32, 64, 128)
    ),
    # F16 queries over an E4M3 cache: the scores multiply in F16, which holds
    # both exactly.
    pytest.param(
        [200, 90],
        {
            "out_dtype": torch.float16,
            "queries": [200, 30],
            "query_dtype": torch.float16,
            "page": 64,
        },
        id="f16 queries over an e4m3 cache, into f16",
        marks=TENSOR_CORES,
    ),
    # A head of 512, the widest the package declares a model with: four query
    # tokens of the group of four a tile and sixteen keys an iteration, on the
    # tensor cores and in the f32 walk.
    *(
        pytest.param(
            [90, 40],
            {
                "out_dtype": query_dtype,
                "queries": [60, 40],
                "query_dtype": query_dtype,
                "page": 64,
                "head_dim": 512,
            },
            id=f"a head of 512 under {NAMES[query_dtype]} queries over an "
            "e4m3 cache",
            marks=marks,
        )
        for query_dtype, marks in (
            (torch.bfloat16, TENSOR_CORES),
            (torch.float32, ()),
        )
    ),
    # A group of three query heads per key-value head: the tile's rows past
    # the group are padding and must write nothing.
    pytest.param(
        [150, 40],
        {"queries": [90, 40], "heads": (6, 2), **SERVED},
        id="a head group that is not a power of two",
        marks=TENSOR_CORES,
    ),
    # A warm-up's idle columns: the first row states the bucket's every token
    # as its queries over no key, and the other rows are empty. Every token
    # reads zero.
    pytest.param(
        [0, 0, 0],
        {"queries": [48, 0, 0]},
        id="a row stating more queries than keys",
    ),
    pytest.param(
        [20],
        {"queries": [36], **SERVED},
        id="a row stating more queries than it has keys before them",
        marks=TENSOR_CORES,
    ),
    # The served dense pairs: BF16 queries over each dense cache type, and
    # FP16 queries over a BF16 cache, each writing its own type; one row
    # follows a prefix and the last pages are partial.
    *(
        pytest.param(
            [170, 24],
            {
                "out_dtype": query_dtype,
                "queries": [106, 24],
                "plane": plane,
                "query_dtype": query_dtype,
            },
            id=f"a {NAMES[plane]} cache under {NAMES[query_dtype]} queries",
            marks=marks,
        )
        for plane, query_dtype, marks in (
            (torch.float16, torch.bfloat16, ()),
            (torch.bfloat16, torch.bfloat16, TENSOR_CORES),
            (torch.float32, torch.bfloat16, ()),
            (torch.bfloat16, torch.float16, ()),
        )
    ),
    # The same dense pairs into an f32 output, where the store's rounding no
    # longer covers a product rounded through BF16.
    *(
        pytest.param(
            [170, 24],
            {"queries": [106, 24], "plane": plane, "query_dtype": query_dtype},
            id=f"a {NAMES[plane]} cache under {NAMES[query_dtype]} queries "
            "into f32",
        )
        for plane, query_dtype in (
            (torch.float16, torch.bfloat16),
            (torch.float32, torch.float16),
        )
    ),
    # An FP16 cache whose hidden slots hold infinities and NaN: each query's
    # visible keys decide alone.
    *(
        pytest.param(
            [70, 24],
            {
                "out_dtype": torch.bfloat16,
                "queries": [6, 24],
                "plane": torch.float16,
                "hidden": hidden,
                "query_dtype": torch.bfloat16,
            },
            id=f"hidden slots of an f16 cache holding {hidden}",
        )
        for hidden in (float("inf"), float("nan"))
    ),
]


@pytest.mark.parametrize(("rows", "case"), CASES)
def test_a_prefill_matches_the_float64_composition_within_its_bound(
    rows, case, torch_device, interpreted, kernel_source
):
    kernels = kernel_source("paged_attention")
    generator = torch.Generator().manual_seed(23)
    operands_of = operands(rows, generator, **case)
    tensor_cores = kernels.prefill_products(
        TRITON_NAMES[operands_of["query"].dtype],
        TRITON_NAMES[operands_of["key"].dtype],
    )["TENSOR_CORES"]
    assert not (interpreted and tensor_cores), (
        "the package multiplies this pair on the tensor cores, which the "
        "interpreter cannot: the case needs the gpu mark"
    )

    expected, largest_score, largest_value, seen = reference(operands_of)
    got = launch(kernels, operands_of, torch_device, interpreted).to(
        torch.float64
    )

    tile = kernels.prefill_tiles(
        *operands_of["heads"], operands_of["head_dim"]
    )["TILE"]
    bound = allowance(
        largest_score,
        largest_value,
        seen,
        expected,
        output_roundoff(operands_of["out"].dtype, interpreted),
        tile,
        tensor_cores,
    )
    # A NaN difference is a failure: `<=` is false for it.
    close = (got - expected).abs() <= bound
    differing = (~close).nonzero()
    token, head, feature = (
        differing[0].tolist() if len(differing) else (0, 0, 0)
    )
    assert bool(close.all()), (
        f"{len(differing)} of {got.numel()} values differ; first at token "
        f"{token}, head {head}, feature {feature}: "
        f"{got[token, head, feature].item():.9g} against "
        f"{expected[token, head, feature].item():.9g}, allowance "
        f"{bound[token, head, feature].item():.3g}"
    )


def test_the_first_query_answers_over_its_own_key_whatever_follows(
    torch_device, interpreted, kernel_source
):
    """The case a program without a causal mask cannot pass: with the values
    after position 0 replaced by values whose attention would dominate, a
    causal program's first query is unchanged and an unmasked one's is not.
    """  # noqa: D205
    kernels = kernel_source("paged_attention")
    generator = torch.Generator().manual_seed(23)
    operands_of = operands([40], generator)
    before = launch(kernels, operands_of, torch_device, interpreted)

    operands_of["value"] = operands_of["value"].clone()
    operands_of["value"][:, 1:, :, :] = torch.full_like(
        operands_of["value"][:, 1:, :, :], 0.5
    ).to(torch.float8_e4m3fn)
    after = launch(kernels, operands_of, torch_device, interpreted)

    # The later queries may move; the first may not.
    drifted = (after[0] - before[0]).abs().max().item()
    assert drifted == 0.0, (
        f"the first query moved by {drifted:.3g} when later keys changed, so "
        "the causal mask is not applied"
    )
