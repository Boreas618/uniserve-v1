#!/usr/bin/env python3
"""Run the Triton paged attention's prefill kernel and compare what it wrote with the reference.

The operator's interface is `ops::attention::PagedAttention`'s, launched as the
operator launches it: a grid of `cdiv(tokens, BLOCK_Q) + rows` query tiles by
`kv_heads`, the tile constexprs of `prefill_tiles`, and the packed token count
as `max_q_len`. Each case is a property this family's defects would break:

* **the causal mask** — a batch row's queries are the last of its keys, so its
  query `p` of `n` over `s` keys attends keys `0..=s - n + p`: the first query
  of a row that holds no earlier keys answers over one key whatever the rest of
  the cache holds, and a row whose queries follow a prefix already in the cache
  (an earlier chunk, or the prefix cache's pages) sees that prefix;
* **the tiles** — a row longer than one query tile and one key tile, so a tile
  finds its row and its first query, and walks key tiles before its diagonal
  unmasked and the rest masked; key tiles that span pages (a page of 32 keys),
  that are one page (64) and that lie within one (128); a head group that is not
  a power of two, whose padded tile rows must stay unwritten; and a head of 512,
  whose tile holds four tokens and sixteen keys;
* **a batch's rows are not all the same length** — a shorter row's tiles must
  leave the next row's tokens alone, and the tokens of a padded bucket that no
  row owns must read zero;
* **a query that sees no key** — a row stating more queries than keys, as a
  warm-up's idle columns do, writes zero;
* **the served pair** — BF16 queries over an E4M3 cache into a BF16 output, and
  every dense pair the package declares, with hidden slots that hold infinities
  and NaN.

The reference is the composition the operator states, in float64 from the same
bytes:

```text
page_k = f64(byte) · key_scale[page]        (a dense page: f64(element))
scores = (q · page_kᵀ) · softmax_scale      masked by j <= seq_len - n + p
out    = Σ softmax(scores) · page_v         zero for a query that sees no key
```

**The bound.** Both sides read the same bytes and scales, so the cache's own
quantization (the error `kv_write_scaled` makes) is not part of this comparison,
and neither is the decode: an E4M3 value has four significant bits and an
exponent in `[-9, 8]`, so each one is a BF16 value and the kernel's decode of it
is exact. What remains is the kernel's f32 arithmetic, bounded per query token
and head from the case's own numbers, with `u = 2⁻²⁴`, `D` the head dimension,
`n` the keys the query sees, `σ` the softmax scale,
`A = σ · max_j Σ_d |q_d · k_jd|` the largest score magnitude and
`V = max_j,d |v_jd|` over the keys it sees:

1. the score: an E4M3 or BF16 key under a BF16 query (an E4M3 key under an F16
   one) is multiplied on the tensor cores exactly, product by product, and
   summed in f32, where an addition may truncate (Fasi, Higham, Mikaitis and
   Pranesh, *Numerical behavior of NVIDIA tensor cores*, PeerJ CS 7, e330, 2021)
   and so costs at most `2u`; any other pair is decoded and multiplied in f32 on
   the CUDA cores, a rounding per product and per addition. Either way, with the
   scale's multiplications, a score is within `ε = (γ_D(2u) + 3u) · A` of exact,
   where `γ_k(v) = k·v / (1 − k·v)`;
2. a weight `exp(s_j − m)` takes `ε`, the subtraction and the `log2 e`
   multiplication inside `tl.exp` (`2u · 2A` together), `ex2.approx`'s relative
   `2⁻²² = 4u`, and the online softmax's `c` corrections of earlier keys (the
   same three, over a maximum that grows by at most `2A` in all, and an f32
   multiplication each; `c = ⌈n/TILE⌉` key tiles on the tensor cores, `c = n`
   keys on the CUDA cores): a relative error of at most
   `η = exp(ε + 12u·A + 5u·(c + 1)) − 1`; the shift `m` itself cancels between
   the weights and their sum;
3. weights perturbed by a relative `η` move a weighted mean of values within `V`
   of each other by at most `2η/(1 − η) · V`;
4. the weighted sum: on the tensor cores the f32 weight enters as two BF16 parts
   that hold it to a relative `e_w = 2⁻¹⁶`, each multiplied by its value exactly
   (`e_w = 0` on the CUDA cores, whose products the next term counts); the
   normaliser and the accumulator are f32 sums of `N = n + 2c` terms (each key's
   or tile's own sum, its correction and its addition), `γ_N(2u)` each, with one
   rounding of the weight's product and one division:
   `(e_w + 2γ_N(2u) + 2u) · V`;
5. the store rounds once into the output's type: `u_out · |out|` (BF16 `2⁻⁸`
   where a device rounds to nearest and `2⁻⁷` where the interpreter truncates,
   F16 `2⁻¹¹`, F32 `u`).

The allowance is the sum,
`2η/(1 − η) · V + (e_w + 2γ_N(2u) + 2u) · V + u_out · |out|`. With an f32 output
it is about `2 · 10⁻⁵ · V` for these cases, so a program that rounded a weight,
a dense key or a query to BF16 before its product (a relative `2⁻⁹`) would
exceed it. The `2⁻¹⁶` of the two-part weight is below the operator's own `γₙ`
for any cache of 256 keys or more.

**What the interpreter does not prove.** Triton's interpreter multiplies a BF16
tile by its bit patterns rather than its values, so the cases the tensor cores
serve (a BF16 or F16 query over an E4M3 page, a BF16 query over a BF16 page) run
with `--device` alone, and the interpreter runs the cases of the f32 walk. Its
E4M3 *decode* is exact for every byte, but its BF16 store truncates where a
device rounds, as in `decode_check.py`. So the interpreter proves the rule, its
order and its masking; the format, the rounding and the tensor cores' products
are the hardware's.

    python paged_triton_prefill_check.py [--package DIR]            # Triton's
    interpreter python paged_triton_prefill_check.py --device [--package DIR]
    # a device
"""

import argparse
import importlib.util
import math
import os
import pathlib
import sys

#: Triton reads `TRITON_INTERPRET` when it is imported, so the mode is chosen
#: from the command line before the kernel is loaded: `--device` compiles for
#: the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402  (after TRITON_INTERPRET, which Triton reads at import)

#: The kernel packages' root: this check lives in
#: `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "paged_attention" / "triton"

Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
PAGE = 32
SET = 0.0625

UNIT = 2.0**-24
#: The bound of a BF16 store: one step where a device rounds to nearest, two
#: where the interpreter truncates.
NARROWED_ROUNDOFF = 2.0**-8 if DEVICE else 2.0**-7
#: The unit roundoff of an F16 row, which both round to nearest.
HALF_ROUNDOFF = 2.0**-11
#: The store's rounding by output type.
OUTPUT_ROUNDOFF = {
    torch.float32: UNIT,
    torch.bfloat16: NARROWED_ROUNDOFF,
    torch.float16: HALF_ROUNDOFF,
}
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


def load_module(package):
    """The package's program module, imported from its source directory."""
    del (
        package
    )  # the check's own location is the package's, wherever it is run from
    path = PACKAGE / "src" / "paged_attention.py"
    spec = importlib.util.spec_from_file_location(
        "paged_attention_kernel", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache(
    rows,
    generator,
    queries,
    plane=None,
    hidden=None,
    page=PAGE,
    heads=(Q_HEADS, KV_HEADS),
    head_dim=HEAD_DIM,
):
    """A cache of as many pages as the rows' keys need, each row's own, and each row's first token.

    `rows` are the rows' key counts and `queries` their query counts, the last
    of their keys. An `E4M3` cache has a scale per page; a `plane` names a dense
    cache's element type, which has none, and `hidden` fills its slots past each
    row's length.
    """
    kv_heads = heads[1]
    pages = sum((length + page - 1) // page for length in rows) + 1
    width = max(1, max((length + page - 1) // page for length in rows))
    table = torch.zeros((len(rows), width), dtype=torch.int32)
    first = [0]
    # Page 0 is left to no row, so a table entry read past a row's own would
    # name a page whose keys change the answer.
    start = 1
    for index, length in enumerate(rows):
        live = (length + page - 1) // page
        table[index, :live] = torch.arange(
            start, start + live, dtype=torch.int32
        )
        start += live
        first.append(first[-1] + queries[index])
    key_values = (
        torch.rand((pages, page, kv_heads, head_dim), generator=generator) * 2.0
        - 1.0
    )
    value_values = (
        torch.rand((pages, page, kv_heads, head_dim), generator=generator) * 2.0
        - 1.0
    )
    state = {
        "page_table": table,
        "seq_lens": torch.tensor(rows, dtype=torch.int32),
        "cum_seq_lens_q": torch.tensor(first, dtype=torch.int32),
        "page": page,
        "heads": heads,
        "head_dim": head_dim,
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
    )
    view = scales.view(-1, 1, 1, 1)
    return {
        **state,
        "key": (key_values / view).to(torch.float8_e4m3fn),
        "value": (value_values / view).to(torch.float8_e4m3fn),
        "key_scale": view,
        "value_scale": view.clone(),
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
    """The query, the cache and the output buffer, in the layouts the operator states.

    `queries` are the rows' query counts, each row's whole key count unless
    stated. `padding` tokens follow the live ones, as a bucket larger than its
    batch holds them. The output starts as `nan` everywhere, so a token the
    kernel failed to write cannot pass as a zero.
    """
    queries = rows if queries is None else queries
    total = sum(queries) + padding
    tokens = (
        torch.rand((total, heads[0], head_dim), generator=generator) * 2.0 - 1.0
    ).to(query_dtype)
    out = torch.full((total, heads[0], head_dim), float("nan"), dtype=out_dtype)
    state = cache(
        rows, generator, queries, plane, hidden, page, heads, head_dim
    )
    # At prefill the operator passes the packed token count as `max_q_len`
    # (`crates/ops/src/attention/mod.rs`).
    return {"query": tokens, "out": out, "max_q_len": total, **state}


def reference(operands_of):
    """The operator's composition in float64, causally masked per row, with what the bound needs.

    Answers the output and, per query token and head, the largest score
    magnitude `A`, the largest value magnitude `V` and the count `n` of the keys
    the query sees.
    """
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
    lengths = operands_of["seq_lens"].tolist()
    first = operands_of["cum_seq_lens_q"].tolist()
    # A token no row owns is padding, and the kernel writes it as zero; so is a
    # query that sees no key.
    out = torch.zeros_like(query)
    largest_score = torch.zeros(query.shape[:2], dtype=torch.float64)
    largest_value = torch.zeros(query.shape[:2], dtype=torch.float64)
    seen = torch.zeros(query.shape[:2], dtype=torch.int64)
    for row, seq_len in enumerate(lengths):
        live = (seq_len + page - 1) // page
        pages = table[row, :live].tolist()
        keys = key[pages].reshape(-1, kv_heads, head_dim)
        values = value[pages].reshape(-1, kv_heads, head_dim)
        length = first[row + 1] - first[row]
        for position in range(length):
            # The query's own key: the row's queries are the last of its keys,
            # and the keys it sees are those up to it, cut so a slot it does not
            # see takes no part.
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
    largest_score,
    largest_value,
    seen,
    expected,
    out_dtype,
    tile,
    tensor_cores,
    head_dim,
):
    """The module documentation's bound, per element.

    `tensor_cores` states that the pair multiplies on the tensor cores, a key
    tile at a time with the weight as two BF16 parts; otherwise one key at a
    time in f32.
    """
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
    return absolute[:, :, None] + OUTPUT_ROUNDOFF[out_dtype] * expected.abs()


def placed(operands_of):
    """The operands where the launch reads them.

    A compiled launch reads them on the device and the interpreter as they are;
    a missing scale stays `None`.
    """
    return {
        name: tensor.cuda()
        if isinstance(tensor, torch.Tensor) and DEVICE
        else tensor
        for name, tensor in operands_of.items()
    }


def launch(module, operands_of):
    """Launch one prefill as the operator does and answer what it wrote, on the host."""
    q_heads, kv_heads = operands_of["heads"]
    on = placed(operands_of)
    query, key, value, out = on["query"], on["key"], on["value"], on["out"]
    tokens = query.shape[0]
    rows = on["page_table"].shape[0]
    head_dim = operands_of["head_dim"]
    tiles = module.prefill_tiles(q_heads, kv_heads, head_dim)
    products = module.prefill_products(
        TRITON_NAMES[query.dtype], TRITON_NAMES[key.dtype]
    )
    grid = (
        (tokens + tiles["BLOCK_Q"] - 1) // tiles["BLOCK_Q"] + rows,
        kv_heads,
    )
    module._paged_attention_prefill_kernel[grid](
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
        on["page"],
        SET,
        on["max_q_len"],
        1,
        tokens,
        rows,
        HEAD_DIM=head_dim,
        PAGE=on["page"],
        GROUP=q_heads // kv_heads,
        QUANTIZED=on["key_scale"] is not None,
        OUT_BF16=out.dtype == torch.bfloat16,
        **tiles,
        **products,
        # The package's compile options; the interpreter has no warps or stages.
        **(module.PREFILL_OPTIONS if DEVICE else {}),
    )
    return out.cpu()


def compare(name, module, rows, generator, out_dtype=torch.float32, **case):
    """Compare one case against the reference within the module documentation's bound.

    `case` states the case's query counts, padding, cache type, page and heads,
    as [`operands`] reads them.
    """
    operands_of = operands(rows, generator, out_dtype, **case)
    tensor_cores = module.prefill_products(
        TRITON_NAMES[operands_of["query"].dtype],
        TRITON_NAMES[operands_of["key"].dtype],
    )["TENSOR_CORES"]
    if tensor_cores and not DEVICE:
        # The interpreter cannot multiply a BF16 tile (module documentation).
        return []
    expected, largest_score, largest_value, seen = reference(operands_of)
    got = launch(module, operands_of).to(torch.float64)
    head_dim = operands_of["head_dim"]
    tile = module.prefill_tiles(*operands_of["heads"], head_dim)["TILE"]
    bound = allowance(
        largest_score,
        largest_value,
        seen,
        expected,
        out_dtype,
        tile,
        tensor_cores,
        head_dim,
    )
    difference = (got - expected).abs()
    # A NaN difference is a failure: `<=` is false for it.
    close = difference <= bound
    if bool(close.all()):
        return []
    differing = (~close).nonzero()
    token, head, feature = differing[0].tolist()
    return [
        f"{name}: {len(differing)} of {got.numel()} values differ; first at token {token}, head "
        f"{head}, feature {feature}: {got[token, head, feature].item():.9g} against "
        f"{expected[token, head, feature].item():.9g}, allowance "
        f"{bound[token, head, feature].item():.3g}"
    ]


def the_mask_is_applied(name, module, generator):
    """The first query row must answer over its own key alone, whatever the cache holds.

    This is the case a program without a mask cannot pass: with the keys after
    position 0 replaced by values whose attention would dominate, a causal
    program's row 0 is unchanged and an unmasked one is not.
    """
    rows = [40]
    operands_of = operands(rows, generator)
    before = launch(module, operands_of).clone()
    operands_of["value"] = operands_of["value"].clone()
    operands_of["value"][:, 1:, :, :] = torch.full_like(
        operands_of["value"][:, 1:, :, :], 0.5
    ).to(torch.float8_e4m3fn)
    after = launch(module, operands_of)
    drifted = (after[0] - before[0]).abs().max().item()
    # Row 1 is allowed to move; row 0 is not.
    if drifted != 0.0:
        return [
            f"{name}: the first query row moved by {drifted:.3g} when later keys changed, so the "
            "causal mask is not applied"
        ]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        default=".",
        help="the package directory (default: this one)",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="compile the cases for a device instead of interpreting them",
    )
    args = parser.parse_args()
    if args.device and not torch.cuda.is_available():
        print("skipped: --device needs a device")
        return 0
    module = load_module(args.package)
    generator = torch.Generator().manual_seed(23)
    served = {"out_dtype": torch.bfloat16, "query_dtype": torch.bfloat16}

    failures = []
    failures += compare(
        "a causal prefill over two pages", module, [40], generator
    )
    failures += compare(
        "a partial last page under the mask", module, [70], generator
    )
    # The shorter row comes first, so its tiles past its length fall on the
    # second row's live tokens: no row's tile may write another row's tokens.
    failures += compare(
        "a batch of two rows with different lengths",
        module,
        [24, 40],
        generator,
    )
    # A bucket larger than its batch: a padded row of length 0 and tokens no row
    # owns, which the kernel must write as zero rather than leave as whatever
    # the buffer held.
    failures += compare(
        "a padded bucket", module, [24, 16, 0], generator, padding=40
    )
    failures += compare(
        "the narrowed store", module, [40], generator, out_dtype=torch.bfloat16
    )
    failures += the_mask_is_applied("the causal mask", module, generator)
    # Rows whose queries follow keys already in the cache: 9 queries after 61
    # keys across a page boundary, and 5 after 30 within the first page, beside
    # a row of no prefix.
    failures += compare(
        "queries after a cached prefix",
        module,
        [70, 35, 12],
        generator,
        queries=[9, 5, 12],
    )
    # The served pair over rows of several query tiles and several key tiles,
    # with key tiles of two pages, one page and half a page: a row of no prefix,
    # a row whose queries follow a long prefix (its first tiles before the
    # diagonal are unmasked), and padding.
    for page in (32, 64, 128):
        failures += compare(
            f"bf16 queries over an e4m3 cache of {page}-key pages, into bf16",
            module,
            [300, 260, 0],
            generator,
            queries=[300, 70, 0],
            padding=13,
            page=page,
            **served,
        )
    # F16 queries over an E4M3 cache: the scores multiply in F16, which holds
    # both exactly.
    failures += compare(
        "f16 queries over an e4m3 cache, into f16",
        module,
        [200, 90],
        generator,
        out_dtype=torch.float16,
        queries=[200, 30],
        query_dtype=torch.float16,
        page=64,
    )
    # A head of 512, the widest the package declares a model with: four query
    # tokens of the group of four a tile and sixteen keys an iteration, on the
    # tensor cores and in the f32 walk.
    for query_dtype in (torch.bfloat16, torch.float32):
        failures += compare(
            f"a head of 512 under {query_dtype} queries over an e4m3 cache",
            module,
            [90, 40],
            generator,
            out_dtype=query_dtype,
            queries=[60, 40],
            query_dtype=query_dtype,
            page=64,
            head_dim=512,
        )
    # A group of three query heads per key-value head: the tile's rows past the
    # group are padding and must write nothing.
    failures += compare(
        "a head group that is not a power of two",
        module,
        [150, 40],
        generator,
        queries=[90, 40],
        heads=(6, 2),
        **served,
    )
    # A warm-up's idle columns: the first row states the bucket's every token as
    # its queries over no key, and the other rows are empty. Every token reads
    # zero.
    failures += compare(
        "a row stating more queries than keys",
        module,
        [0, 0, 0],
        generator,
        queries=[48, 0, 0],
    )
    failures += compare(
        "a row stating more queries than it has keys before them",
        module,
        [20],
        generator,
        queries=[36],
        **served,
    )
    # The served dense pairs: BF16 queries over each dense cache type, and FP16
    # queries over a BF16 cache, each writing its own type; one row follows a
    # prefix and the last pages are partial.
    for plane, query_dtype in (
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ):
        failures += compare(
            f"a {plane} cache under {query_dtype} queries",
            module,
            [170, 24],
            generator,
            out_dtype=query_dtype,
            queries=[106, 24],
            plane=plane,
            query_dtype=query_dtype,
        )
    # The same dense pairs into an f32 output, where the store's rounding no
    # longer covers a product rounded through BF16.
    for plane, query_dtype in (
        (torch.float16, torch.bfloat16),
        (torch.float32, torch.float16),
    ):
        failures += compare(
            f"a {plane} cache under {query_dtype} queries into f32",
            module,
            [170, 24],
            generator,
            queries=[106, 24],
            plane=plane,
            query_dtype=query_dtype,
        )
    # An FP16 cache whose hidden slots hold infinities and NaN: each query's
    # visible keys decide alone.
    for hidden in (float("inf"), float("nan")):
        failures += compare(
            f"hidden slots of an f16 cache holding {hidden}",
            module,
            [70, 24],
            generator,
            out_dtype=torch.bfloat16,
            queries=[6, 24],
            plane=torch.float16,
            hidden=hidden,
            query_dtype=torch.bfloat16,
        )

    if failures:
        print("failed:")
        for line in failures:
            print("  " + line)
        return 1
    where = "the device" if args.device else "Triton's interpreter"
    print(
        f"ok on {where}: every case matches the float64 reference within the bound of the "
        "kernel's f32 arithmetic — a query attends only its own past, after any prefix its row "
        "holds, across query tiles, key tiles and pages, whatever the hidden slots hold, and the "
        "tokens no row owns and the queries that see no key read zero"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
