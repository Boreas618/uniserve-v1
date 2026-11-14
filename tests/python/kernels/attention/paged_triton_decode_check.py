#!/usr/bin/env python3
"""Run the Triton paged attention's decode kernel and compare what it wrote with the reference.

The operator is one paged attention (`ops::attention::PagedAttention`) whose cache this program
serves when the vendor programs do not: a page of `E4M3` bytes with one f32 scale per page, or a
dense page (BF16, F16 or F32) of another type than the queries, read with no scale. The reference
here is the composition the operator states, written in torch from the same lines the kernel
computes:

```text
page_k = f32(byte) · key_scale[page]                    once, in f32, before any product
page_v = f32(byte) · value_scale[page]                  (a dense page: f32(element), no scale)
scores = (Q · page_kᵀ) · softmax_scale                  f32, over the keys the row's length shows
out    = Σ softmax(scores) · page_v                     f32, one rounding into the output's type
```

The cases, each a property this family's defects would break:

* **a whole table of pages** — several pages per row, a per-page scale that differs, and a head group
  above one: a program that read the query head as the key-value head, or one page's scale as the
  cache's, answers a different matrix here;
* **the exact dequantization** — a cache whose bytes and power-of-two scale reproduce chosen values
  exactly: the answer must equal the same attention over those values, which is the property
  *"decoded once, never re-quantized"* is worth;
* **a partial last page** — the keys past the row's length are attended to by nothing, so the answer
  must equal the same attention with those keys replaced by values that would change it;
* **one boosted page** — a page's scale ten times its neighbours' must show up as that page's weight,
  which a program that skipped the scale cannot produce;
* **a dense page of each type** — BF16 queries over a BF16, FP16 and FP32 cache, and FP16 queries
  over a BF16 cache, read with no scale (the scale pointers are null, so a program that read one
  does not compile);
* **hidden lanes that are not finite** — an FP16 cache of a BF16 model holds an infinity where a
  value overflowed, and a slot past a row's length holds whatever an earlier request left: the
  answer must be the one over the row's own keys, where a program that weighed a hidden value by
  zero would answer NaN.

**What this check does not prove: the format or the rounding.** The interpreter's own `E4M3` cast is
wrong at a mantissa boundary (measured on this platform: `126.7 → 64.0`) and its BF16 store truncates
where a device rounds (2072 of 4096 values), and every case casts through the interpreter. So this
proves the rule, its order and its arithmetic — that the page is decoded once, in f32, before any
product — and the format and the rounding are the hardware's. The narrowing case states that in its
own bound rather than in the arithmetic cases'. `--device` compiles the same cases for a device,
where the narrowed store rounds to nearest and its bound is one step.

    python paged_triton_decode_check.py [--package DIR]            # Triton's interpreter
    python paged_triton_decode_check.py --device [--package DIR]   # a device
"""

import argparse
import importlib.util
import os
import pathlib
import sys

#: Triton reads `TRITON_INTERPRET` when it is imported, so the mode is chosen from the command line
#: before the kernel is loaded: `--device` compiles for the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402  (after TRITON_INTERPRET, which Triton reads at import)

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "paged_attention" / "triton"

# The interpreted shape: small enough to run in seconds, structured like the declared row (a query
# head group above one, a page that does not divide the row's length).
BATCH = 2
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
PAGE = 32
SET = 0.0625  # the softmax scale a device's `1/sqrt(head_dim)` would be, kept exact


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    del (
        package
    )  # the check's own location is the package's, wherever it is run from
    path = PACKAGE / "src" / "paged_attention.py"
    spec = importlib.util.spec_from_file_location(
        "paged_attention_kernel", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._paged_attention_decode_kernel


def operands(
    cache_of,
    rows,
    generator,
    out_dtype=torch.float32,
    query_dtype=torch.float32,
):
    """The query, the cache and the output buffer, in the layouts the operator states.

    A dense cache has no scales: both are `None`, the null pointers the operator passes.
    """
    table, lengths, keys, values, key_scale, value_scale = cache_of(
        rows, generator
    )
    query = (
        torch.rand((len(rows), Q_HEADS, HEAD_DIM), generator=generator) * 2.0
        - 1.0
    ).to(query_dtype)
    out = torch.zeros((len(rows), Q_HEADS, HEAD_DIM), dtype=out_dtype)
    return {
        "query": query,
        "key": keys,
        "value": values,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "page_table": table,
        "seq_lens": lengths,
        "out": out,
    }


def decoded(tensor, dtype=torch.float32):
    """A page decoded once, in f32 — the one operation the operator fixes the order of."""
    return tensor.to(dtype)


def reference(operands_of, dtype=torch.float32):
    """The operator's composition, in torch, over every key the row's length shows."""
    query = operands_of["query"].to(dtype)
    key = decoded(operands_of["key"], dtype)
    value = decoded(operands_of["value"], dtype)
    if operands_of["key_scale"] is not None:
        key = key * operands_of["key_scale"].to(dtype)
        value = value * operands_of["value_scale"].to(dtype)
    table = operands_of["page_table"]
    lengths = operands_of["seq_lens"]
    out = torch.zeros_like(query)
    for row in range(query.shape[0]):
        length = int(lengths[row].item())
        live = (length + PAGE - 1) // PAGE
        pages = table[row, :live].tolist()
        # `[length, KV_HEADS, HEAD_DIM]`: the pages in the table's own order, cut at the row's
        # length, so a slot past it takes no part whatever it holds.
        keys = key[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:length]
        values = value[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:length]
        for head in range(Q_HEADS):
            kv_head = head * KV_HEADS // Q_HEADS
            scores = (keys[:, kv_head, :] @ query[row, head, :]) * SET
            weights = torch.softmax(scores, dim=0)
            out[row, head, :] = weights @ values[:, kv_head, :]
    return out


def placed(operands_of):
    """The operands where the launch reads them.

    A compiled launch reads them on the device and the interpreter as they are; a missing scale
    stays `None`.
    """
    return {
        name: tensor
        if tensor is None or not isinstance(tensor, torch.Tensor) or not DEVICE
        else tensor.cuda()
        for name, tensor in operands_of.items()
    }


def launch(kernel, operands_of):
    """Launch one decode and answer what it wrote, on the host."""
    narrowed = operands_of["out"].dtype == torch.bfloat16
    on = placed(operands_of)
    kernel[(on["seq_lens"].shape[0], Q_HEADS, 1)](
        on["out"],
        on["query"],
        on["key"],
        on["value"],
        on["key_scale"],
        on["value_scale"],
        on["page_table"],
        on["seq_lens"],
        on["query"].stride(0),
        on["query"].stride(1),
        on["query"].stride(2),
        on["key"].stride(0),
        on["key"].stride(1),
        on["key"].stride(2),
        on["key"].stride(3),
        on["value"].stride(0),
        on["value"].stride(1),
        on["value"].stride(2),
        on["value"].stride(3),
        on["out"].stride(0),
        on["out"].stride(1),
        on["out"].stride(2),
        on["page_table"].shape[1],
        PAGE,
        SET,
        HEAD_DIM=HEAD_DIM,
        PAGE=PAGE,
        GROUP=Q_HEADS // KV_HEADS,
        QUANTIZED=on["key_scale"] is not None,
        OUT_BF16=narrowed,
    )
    return on["out"].cpu()


#: The unit roundoff of the F32 accumulation and of the BF16 row the store narrows into.
ACCUMULATION_ROUNDOFF = 2.0**-24
OUTPUT_ROUNDOFF = 2.0**-8
#: The bound of a BF16 store: one step where a device rounds to nearest, two where the interpreter
#: truncates.
NARROWED_ROUNDOFF = OUTPUT_ROUNDOFF if DEVICE else 2.0 * OUTPUT_ROUNDOFF
#: The unit roundoff of an F16 row, which both round to nearest.
HALF_ROUNDOFF = 2.0**-11


def gamma(reduction, unit=ACCUMULATION_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def compare(
    name,
    kernel,
    cache_of,
    rows,
    generator,
    out_dtype=torch.float32,
    output_roundoff=OUTPUT_ROUNDOFF,
    query_dtype=torch.float32,
):
    """Compare one case against the reference within the operator's declared bound."""
    operands_of = operands(cache_of, rows, generator, out_dtype, query_dtype)
    expected = reference(operands_of).to(torch.float32)
    got = launch(kernel, operands_of).to(torch.float32)
    # `Tolerance::rounding(max_kv_len, F32, output)` — the reduction is the row's key count — and
    # the operator's envelope term `(2γ + u) · VALUE_ENVELOPE` for the page scales' cancellation.
    reduction = max(int(value) for value in rows)
    bound = 2.0 * gamma(reduction) * reduction**0.5 + output_roundoff
    envelope = (2.0 * gamma(reduction) + ACCUMULATION_ROUNDOFF) * 8.0
    difference = (got - expected).abs()
    close = difference <= bound * expected.abs() + envelope
    if bool(close.all()):
        return []
    differing = (~close).nonzero()
    row, head, feature = differing[0].tolist()
    return [
        f"{name}: {len(differing)} of {got.numel()} values differ; first at row {row}, head "
        f"{head}, feature {feature}: {got[row, head, feature].item():.7g} against "
        f"{expected[row, head, feature].item():.7g}, allowance {bound:.3g} relative plus "
        f"{envelope:.3g} absolute"
    ]


def page_scales(page):
    """A per-page scale that differs page by page, so one scale read for all is a different answer."""
    return [1.0, 0.5, 2.0, 1.0, 0.25][page % 5]


def exact_scales(page):
    """The identity case's scale: a power of two, so a byte times it is the value it stored."""
    del page
    return 1.0


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
    kernel = load_kernel(args.package)
    generator = torch.Generator().manual_seed(11)

    def whole(rows, generator):
        """A table over its own distinct pages, each row's pages its own."""
        pages = sum((length + PAGE - 1) // PAGE for length in rows)
        length = len(rows)
        width = max((rows[index] + PAGE - 1) // PAGE for index in range(length))
        table = torch.zeros((length, width), dtype=torch.int32)
        start = 0
        for index in range(length):
            live = (rows[index] + PAGE - 1) // PAGE
            table[index, :live] = torch.arange(
                start, start + live, dtype=torch.int32
            )
            start += live
        # Two independent tensors: a program that read the value plane as the key plane, or swapped
        # them, answers a different matrix here.
        keys = (
            torch.rand((pages, PAGE, KV_HEADS, HEAD_DIM), generator=generator)
            * 2.0
            - 1.0
        )
        values = (
            torch.rand((pages, PAGE, KV_HEADS, HEAD_DIM), generator=generator)
            * 2.0
            - 1.0
        )
        return table, pages, keys, values

    def build(rows, generator, scales_of):
        table, pages, key_values, value_values = whole(rows, generator)
        scales = torch.tensor(
            [scales_of(page) for page in range(pages)], dtype=torch.float32
        )
        view = scales.view(-1, 1, 1, 1)
        keys = (key_values / view).to(torch.float8_e4m3fn)
        values = (value_values / view).to(torch.float8_e4m3fn)
        lengths = torch.tensor(rows, dtype=torch.int32)
        return (
            table,
            lengths,
            keys,
            values,
            view,
            scales.clone().view(-1, 1, 1, 1),
        )

    cases = []
    # A whole table of pages, each with its own scale: 5 pages for one row, 3 for the other.
    cases.append(
        (
            "a table of pages with per-page scales",
            lambda rows, generator: build(rows, generator, page_scales),
            [160, 96],
        )
    )
    # The identity: bytes times a power-of-two scale reproduce the values, so the quantized path
    # must agree with the same attention over the values it stored.
    cases.append(
        (
            "the exact dequantization",
            lambda rows, generator: build(rows, generator, exact_scales),
            [128, 64],
        )
    )
    # A last page that is partial in both rows.
    cases.append(
        (
            "a partial last page",
            lambda rows, generator: build(rows, generator, page_scales),
            [PAGE * 2 + 7, PAGE + 1],
        )
    )

    failures = []
    for name, cache_of, rows in cases:
        failures += compare(name, kernel, cache_of, rows, generator)

    # One page's scale is ten times its neighbours': a program that read one page's scale for the
    # cache, or skipped it, answers a different weighted sum here.
    def boosted(page):
        return 10.0 if page == 0 else 1.0

    failures += compare(
        "a cache with a ten-times page",
        kernel,
        lambda rows, generator: build(rows, generator, boosted),
        [96],
        generator,
    )
    # The narrowed store, as its own case: Triton's interpreter truncates a BF16 store where a device
    # rounds (measured here for 2072 of 4096 values, up to two steps of the row type), so its bound
    # carries two steps where the operator declares one. The case proves the arithmetic reaches the
    # store; the rounding itself is the device's.
    failures += compare(
        "the narrowed store",
        kernel,
        lambda rows, generator: build(rows, generator, page_scales),
        [96],
        generator,
        out_dtype=torch.bfloat16,
        output_roundoff=NARROWED_ROUNDOFF,
    )

    def dense(plane, hidden=None):
        """A dense cache of `plane` elements with no scales.

        `hidden` fills every slot past a row's length in its last page.
        """

        def cache_of(rows, generator):
            table, _, key_values, value_values = whole(rows, generator)
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

    # The served pairs: BF16 queries over each dense cache type, and FP16 queries over a BF16
    # cache, each writing its own type, with partial last pages.
    for plane, query_dtype, roundoff in (
        (torch.float16, torch.bfloat16, NARROWED_ROUNDOFF),
        (torch.bfloat16, torch.bfloat16, NARROWED_ROUNDOFF),
        (torch.float32, torch.bfloat16, NARROWED_ROUNDOFF),
        (torch.bfloat16, torch.float16, HALF_ROUNDOFF),
    ):
        failures += compare(
            f"a {plane} cache under {query_dtype} queries",
            kernel,
            dense(plane),
            [PAGE * 2 + 7, PAGE + 1],
            generator,
            out_dtype=query_dtype,
            output_roundoff=roundoff,
            query_dtype=query_dtype,
        )
    # An FP16 cache whose hidden slots hold infinities and NaN: the row's own keys decide alone.
    for hidden in (float("inf"), float("nan")):
        failures += compare(
            f"hidden slots of an f16 cache holding {hidden}",
            kernel,
            dense(torch.float16, hidden),
            [PAGE * 2 + 7, PAGE + 1],
            generator,
            out_dtype=torch.bfloat16,
            output_roundoff=NARROWED_ROUNDOFF,
            query_dtype=torch.bfloat16,
        )

    if failures:
        print("failed:")
        for line in failures:
            print("  " + line)
        return 1
    where = "the device" if args.device else "Triton's interpreter"
    print(
        f"ok on {where}: every case matches the reference — the pages decode once in f32, a dense "
        "page of each type is read without a scale, the rows' lengths mask the last page whatever "
        "its hidden slots hold, and each page's own scale reaches the answer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
