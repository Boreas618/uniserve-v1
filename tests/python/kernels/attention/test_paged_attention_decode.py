"""The Triton paged attention's decode against the operator's composition.

The operator is one paged attention (`ops::attention::PagedAttention`) whose
cache this program serves when the vendor programs do not: a page of E4M3
bytes with one f32 scale per page, or a dense page (BF16, F16 or F32) of
another type than the queries, read with no scale. The reference is the
composition the operator states, in torch from the same lines the kernel
computes::

    page_k = f32(byte) · key_scale[page]   once, in f32, before any product
    page_v = f32(byte) · value_scale[page] (a dense page: f32(element))
    scores = (Q · page_kᵀ) · softmax_scale  f32, over the row's keys
    out    = Σ softmax(scores) · page_v     f32, one rounding into the output

The cases, each a property this family's defects would break:

- **a whole table of pages**: several pages per row, a per-page scale that
  differs, and a head group above one, so a program that read the query head
  as the key-value head, or one page's scale as the cache's, answers a
  different matrix;
- **the exact dequantization**: bytes and a power-of-two scale that reproduce
  chosen values exactly, so the answer equals the same attention over those
  values ("decoded once, never re-quantized");
- **a partial last page**: the keys past the row's length are attended to by
  nothing;
- **one boosted page**: a page's scale ten times its neighbours' shows as that
  page's weight, which a program that skipped the scale cannot produce;
- **a dense page of each type**: BF16 queries over a BF16, FP16 and FP32
  cache, and FP16 queries over a BF16 cache, read with no scale (the scale
  pointers are null);
- **hidden lanes that are not finite**: an FP16 cache holds an infinity where
  a value overflowed, and a slot past a row's length holds whatever an earlier
  request left; the answer is the one over the row's own keys, where a program
  that weighed a hidden value by zero would answer NaN.

The interpreter's own E4M3 cast is wrong at a mantissa boundary
(`126.7 → 64.0`) and its BF16 store truncates where a device rounds, and every
case casts through it; so on the interpreter this proves the rule, its order
and its arithmetic, and a narrowed store's bound carries two steps of the
output's type where a device's carries one.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

# A small shape structured like the declared row: a query head group above
# one, and a page that does not divide the row's length.
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
PAGE = 32
#: The softmax scale a device's `1/sqrt(head_dim)` would be, kept exact.
SET = 0.0625

#: The unit roundoff of the F32 accumulation, and the operator's value
#: envelope, over which the page scales' cancellation is an absolute term.
ACCUMULATION_ROUNDOFF = 2.0**-24
VALUE_ENVELOPE = 8.0
#: The output's rounding the bound carries, by output type: one BF16 step for
#: an F32 output (the operator's declared output roundoff), one step of an F16
#: row, and for a BF16 row one step where a device rounds to nearest and two
#: where the interpreter truncates.
OUTPUT_ROUNDOFF = 2.0**-8
HALF_ROUNDOFF = 2.0**-11


def narrowed_roundoff(interpreted):
    """The bound of a BF16 store in the session's mode."""
    return 2.0 * OUTPUT_ROUNDOFF if interpreted else OUTPUT_ROUNDOFF


def gamma(reduction, unit=ACCUMULATION_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def pages_of(rows, generator):
    """A table over its own distinct pages, each row's pages its own, and two
    independent planes of values in `[-1, 1)`: a program that read the value
    plane as the key plane, or swapped them, answers a different matrix.
    """  # noqa: D205
    pages = sum((length + PAGE - 1) // PAGE for length in rows)
    width = max((length + PAGE - 1) // PAGE for length in rows)
    table = torch.zeros((len(rows), width), dtype=torch.int32)
    start = 0
    for index, length in enumerate(rows):
        live = (length + PAGE - 1) // PAGE
        table[index, :live] = torch.arange(
            start, start + live, dtype=torch.int32
        )
        start += live
    shape = (pages, PAGE, KV_HEADS, HEAD_DIM)
    keys = torch.rand(shape, generator=generator) * 2.0 - 1.0
    values = torch.rand(shape, generator=generator) * 2.0 - 1.0
    return table, pages, keys, values


def page_scales(page):
    """A per-page scale that differs page by page, so one scale read for all
    is a different answer.
    """  # noqa: D205
    return [1.0, 0.5, 2.0, 1.0, 0.25][page % 5]


def exact_scales(page):
    """A power of two, so a byte times it is the value it stored."""
    del page
    return 1.0


def boosted(page):
    """Ten times the neighbours' scale on the first page."""
    return 10.0 if page == 0 else 1.0


def quantized(scales_of):
    """An E4M3 cache with the per-page scales `scales_of(page)`."""

    def cache_of(rows, generator):
        table, pages, key_values, value_values = pages_of(rows, generator)
        scales = torch.tensor(
            [scales_of(page) for page in range(pages)], dtype=torch.float32
        ).view(-1, 1, 1, 1)
        keys = (key_values / scales).to(torch.float8_e4m3fn)
        values = (value_values / scales).to(torch.float8_e4m3fn)
        lengths = torch.tensor(rows, dtype=torch.int32)
        return table, lengths, keys, values, scales, scales.clone()

    return cache_of


def dense(plane, hidden=None):
    """A dense cache of `plane` elements with no scales; `hidden` fills every
    slot past a row's length in its last page.
    """  # noqa: D205

    def cache_of(rows, generator):
        table, _, key_values, value_values = pages_of(rows, generator)
        keys = key_values.to(plane)
        values = value_values.to(plane)
        if hidden is not None:
            for row, length in enumerate(rows):
                last = int(table[row, (length - 1) // PAGE])
                keys[last, length % PAGE or PAGE :] = hidden
                values[last, length % PAGE or PAGE :] = hidden
        lengths = torch.tensor(rows, dtype=torch.int32)
        return table, lengths, keys, values, None, None

    return cache_of


def reference(operands):
    """The operator's composition, in torch, over every key the row's length
    shows.
    """  # noqa: D205
    query = operands["query"].to(torch.float32)
    key = operands["key"].to(torch.float32)
    value = operands["value"].to(torch.float32)
    if operands["key_scale"] is not None:
        key = key * operands["key_scale"]
        value = value * operands["value_scale"]
    table = operands["page_table"]
    out = torch.zeros_like(query)
    for row, length in enumerate(operands["seq_lens"].tolist()):
        live = (length + PAGE - 1) // PAGE
        pages = table[row, :live].tolist()
        # `[length, KV_HEADS, HEAD_DIM]`: the pages in the table's own order,
        # cut at the row's length, so a slot past it takes no part whatever it
        # holds.
        keys = key[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:length]
        values = value[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:length]
        for head in range(Q_HEADS):
            kv_head = head * KV_HEADS // Q_HEADS
            scores = (keys[:, kv_head, :] @ query[row, head, :]) * SET
            out[row, head, :] = (
                torch.softmax(scores, dim=0) @ values[:, kv_head]
            )
    return out


def launch(kernels, operands):
    """One decode over the operands, which are where the launch reads them;
    a dense cache's scales are `None`, the null pointers the operator passes.
    """  # noqa: D205
    query, key, value, out = (
        operands[name] for name in ("query", "key", "value", "out")
    )
    page_table = operands["page_table"]
    kernels._paged_attention_decode_kernel[
        (operands["seq_lens"].shape[0], Q_HEADS, 1)
    ](
        out,
        query,
        key,
        value,
        operands["key_scale"],
        operands["value_scale"],
        page_table,
        operands["seq_lens"],
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
        page_table.shape[1],
        PAGE,
        SET,
        HEAD_DIM=HEAD_DIM,
        PAGE=PAGE,
        GROUP=Q_HEADS // KV_HEADS,
        QUANTIZED=operands["key_scale"] is not None,
        OUT_BF16=out.dtype == torch.bfloat16,
    )


#: The element types by the names the case ids use.
NAMES = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}

#: The cases: the cache, the rows' key counts, the output's type and the
#: queries' type.
CASES = {
    # A whole table of pages, each with its own scale: 5 pages for one row, 3
    # for the other.
    "a table of pages with per-page scales": (
        quantized(page_scales), [160, 96], torch.float32, torch.float32
    ),
    # The identity: bytes times a power-of-two scale reproduce the values, so
    # the quantized path must agree with the same attention over them.
    "the exact dequantization": (
        quantized(exact_scales), [128, 64], torch.float32, torch.float32
    ),
    "a partial last page": (
        quantized(page_scales), [PAGE * 2 + 7, PAGE + 1], torch.float32,
        torch.float32,
    ),
    "a cache with a ten-times page": (
        quantized(boosted), [96], torch.float32, torch.float32
    ),
    # The narrowed store: the case proves the arithmetic reaches the store; the
    # rounding itself is the device's.
    "the narrowed store": (
        quantized(page_scales), [96], torch.bfloat16, torch.float32
    ),
    # The served dense pairs: BF16 queries over each dense cache type, and FP16
    # queries over a BF16 cache, each writing its own type, with partial last
    # pages.
    **{
        f"a {NAMES[plane]} cache under {NAMES[query]} queries": (
            dense(plane), [PAGE * 2 + 7, PAGE + 1], query, query
        )
        for plane, query in (
            (torch.float16, torch.bfloat16),
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float16),
        )
    },
    # An FP16 cache whose hidden slots hold infinities and NaN: the row's own
    # keys decide alone.
    **{
        f"hidden slots of an f16 cache holding {hidden}": (
            dense(torch.float16, hidden), [PAGE * 2 + 7, PAGE + 1],
            torch.bfloat16, torch.bfloat16,
        )
        for hidden in (float("inf"), float("nan"))
    },
}  # fmt: skip


@pytest.mark.parametrize("case", CASES)
def test_a_decode_matches_the_operators_composition_within_its_bound(
    case, torch_device, interpreted, kernel_source
):
    cache_of, rows, out_dtype, query_dtype = CASES[case]
    generator = torch.Generator().manual_seed(11)
    table, lengths, keys, values, key_scale, value_scale = cache_of(
        rows, generator
    )
    query = (
        torch.rand((len(rows), Q_HEADS, HEAD_DIM), generator=generator) * 2.0
        - 1.0
    ).to(query_dtype)
    operands = {
        "query": query,
        "key": keys,
        "value": values,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "page_table": table,
        "seq_lens": lengths,
        "out": torch.zeros((len(rows), Q_HEADS, HEAD_DIM), dtype=out_dtype),
    }
    expected = reference(operands)

    placed = {
        name: None if tensor is None else tensor.to(torch_device)
        for name, tensor in operands.items()
    }
    launch(kernel_source("paged_attention"), placed)
    got = placed["out"].cpu().to(torch.float32)

    # `Tolerance::rounding(max_kv_len, F32, output)`, the reduction being the
    # row's key count, and the operator's envelope term
    # `(2γ + u) · VALUE_ENVELOPE` for the page scales' cancellation.
    output_roundoff = {
        torch.float32: OUTPUT_ROUNDOFF,
        torch.bfloat16: narrowed_roundoff(interpreted),
        torch.float16: HALF_ROUNDOFF,
    }[out_dtype]
    reduction = max(rows)
    bound = 2.0 * gamma(reduction) * reduction**0.5 + output_roundoff
    envelope = (2.0 * gamma(reduction) + ACCUMULATION_ROUNDOFF) * VALUE_ENVELOPE
    # A NaN difference is a failure: `<=` is false for it.
    close = (got - expected).abs() <= bound * expected.abs() + envelope
    differing = (~close).nonzero()
    row, head, feature = differing[0].tolist() if len(differing) else (0, 0, 0)
    assert bool(close.all()), (
        f"{case}: {len(differing)} of {got.numel()} values differ; first at "
        f"row {row}, head {head}, feature {feature}: "
        f"{got[row, head, feature].item():.7g} against "
        f"{expected[row, head, feature].item():.7g}, allowance {bound:.3g} "
        f"relative plus {envelope:.3g} absolute"
    )
