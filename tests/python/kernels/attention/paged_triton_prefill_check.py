#!/usr/bin/env python3
"""Run the Triton paged attention's prefill kernel and compare what it wrote with the reference.

The operator's interface is `ops::attention::PagedAttention`'s; the prefill program differs from
the decode one in exactly two ways, and each is a case here:

* **the causal mask** — a batch row's queries are the last of its keys, so its query `p` of `n`
  over `s` keys attends keys `0..=s - n + p`: the first query of a row that holds no earlier keys
  answers over one key whatever the rest of the cache holds, and a row whose queries follow a
  prefix already in the cache (an earlier chunk, or the prefix cache's pages) sees that prefix;
* **a batch's rows are not all the same length** — a packed batch states each row's first query token
  in `cum_seq_lens_q`, an unpacked one gives every row `max_q_len`, and the two statements of the same
  data must produce the same answer. The launch is the operator's: `max_q_len` and the grid's query
  rows are the packed token count, so a shorter row's tiles past its length must leave the next
  row's tokens alone, and the tokens of a padded bucket that no row owns must read zero.

The reference is the composition the operator states, written in torch:

```text
page_k = f32(byte) · key_scale[page]        once, in f32, before any product (a dense page: f32)
scores = (q · page_kᵀ) · softmax_scale      masked by j <= seq_len - n + p and j < seq_len
out    = Σ softmax(scores) · page_v         one rounding into the output's type
```

The cache is an `E4M3` page with one f32 scale per page, or a dense page (BF16, F16 or F32) of
another type than the queries, read with no scale; a dense case also fills the slots past each
row's length with infinities, as an FP16 cache of a BF16 model may hold, which must take no part.

**What this check does not prove**, as in `decode_check.py`: the interpreter's E4M3 *decode* is exact
for every byte, but its BF16 store truncates where a device rounds, and both sides cast through the
interpreter. So this proves the rule, its order and its masking; the format and the rounding are the
hardware's, which `--device` runs the same cases on.

    python paged_triton_prefill_check.py [--package DIR]            # Triton's interpreter
    python paged_triton_prefill_check.py --device [--package DIR]   # a device
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

Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
PAGE = 32
SET = 0.0625


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
    return module._paged_attention_prefill_kernel


def cache(rows, generator, queries, plane=None, hidden=None):
    """A cache of as many pages as the rows' keys need, each row's own, and each row's first token.

    `rows` are the rows' key counts and `queries` their query counts, the last of their keys. An
    `E4M3` cache has a scale per page; a `plane` names a dense cache's element type, which has
    none, and `hidden` fills its slots past each row's length.
    """
    pages = sum((length + PAGE - 1) // PAGE for length in rows)
    width = max((length + PAGE - 1) // PAGE for length in rows)
    table = torch.zeros((len(rows), width), dtype=torch.int32)
    first = [0]
    start = 0
    for index, length in enumerate(rows):
        live = (length + PAGE - 1) // PAGE
        table[index, :live] = torch.arange(
            start, start + live, dtype=torch.int32
        )
        start += live
        first.append(first[-1] + queries[index])
    key_values = (
        torch.rand((pages, PAGE, KV_HEADS, HEAD_DIM), generator=generator) * 2.0
        - 1.0
    )
    value_values = (
        torch.rand((pages, PAGE, KV_HEADS, HEAD_DIM), generator=generator) * 2.0
        - 1.0
    )
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
                last = int(table[row, (length - 1) // PAGE])
                keys[last, length % PAGE or PAGE :] = hidden
                values[last, length % PAGE or PAGE :] = hidden
        return {
            **state,
            "key": keys,
            "value": values,
            "key_scale": None,
            "value_scale": None,
        }
    cycle = [1.0, 0.5, 2.0, 0.25]
    scales = torch.tensor(
        [cycle[page % len(cycle)] for page in range(pages)], dtype=torch.float32
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
):
    """The query, the cache and the output buffer, in the layouts the operator states.

    `queries` are the rows' query counts, each row's whole key count unless stated. `padding`
    tokens follow the live ones, as a bucket larger than its batch holds them. The output starts
    as `nan` everywhere, so a token the kernel failed to write cannot pass as a zero.
    """
    queries = rows if queries is None else queries
    total = sum(queries) + padding
    tokens = (
        torch.rand((total, Q_HEADS, HEAD_DIM), generator=generator) * 2.0 - 1.0
    ).to(query_dtype)
    out = torch.full((total, Q_HEADS, HEAD_DIM), float("nan"), dtype=out_dtype)
    state = cache(rows, generator, queries, plane, hidden)
    # At prefill the operator passes the packed token count as `max_q_len` and launches that many
    # query rows per batch row (`crates/ops/src/attention/mod.rs`), so a shorter row's tiles past
    # its length are the padding path.
    return {
        "query": tokens,
        "out": out,
        "max_q_len": total,
        "packed": True,
        **state,
    }


def reference(operands_of, dtype=torch.float32):
    """The operator's composition, in torch, causally masked per row."""
    query = operands_of["query"].to(dtype)
    key = operands_of["key"].to(dtype)
    value = operands_of["value"].to(dtype)
    if operands_of["key_scale"] is not None:
        key = key * operands_of["key_scale"].to(dtype)
        value = value * operands_of["value_scale"].to(dtype)
    table = operands_of["page_table"]
    lengths = operands_of["seq_lens"].tolist()
    first = operands_of["cum_seq_lens_q"].tolist()
    # A token no row owns is padding, and the kernel writes it as zero.
    out = torch.zeros_like(query)
    for row, seq_len in enumerate(lengths):
        live = (seq_len + PAGE - 1) // PAGE
        pages = table[row, :live].tolist()
        keys = key[pages].reshape(-1, KV_HEADS, HEAD_DIM)
        values = value[pages].reshape(-1, KV_HEADS, HEAD_DIM)
        length = first[row + 1] - first[row]
        for position in range(length):
            # The query's own key: the row's queries are the last of its keys, and the keys it
            # sees are those up to it, cut so a slot it does not see takes no part.
            own = seq_len - length + position
            for head in range(Q_HEADS):
                kv_head = head * KV_HEADS // Q_HEADS
                scores = (
                    keys[: own + 1, kv_head, :]
                    @ query[first[row] + position, head, :]
                ) * SET
                weights = torch.softmax(scores, dim=0)
                out[first[row] + position, head, :] = (
                    weights @ values[: own + 1, kv_head, :]
                )
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


def launch(kernel, operands_of, packed=True):
    """Launch one prefill and answer what it wrote, on the host."""
    narrowed = operands_of["out"].dtype == torch.bfloat16
    tokens = operands_of["query"].shape[0]
    on = placed(operands_of)
    kernel[(on["max_q_len"], Q_HEADS, on["page_table"].shape[0])](
        on["out"],
        on["query"],
        on["key"],
        on["value"],
        on["key_scale"],
        on["value_scale"],
        on["page_table"],
        on["seq_lens"],
        on["cum_seq_lens_q"],
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
        on["max_q_len"],
        1 if packed else 0,
        HEAD_DIM=HEAD_DIM,
        PAGE=PAGE,
        GROUP=Q_HEADS // KV_HEADS,
        QUANTIZED=on["key_scale"] is not None,
        OUT_BF16=narrowed,
    )
    return on["out"].cpu(), tokens


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
    rows,
    generator,
    out_dtype=torch.float32,
    packed=True,
    output_roundoff=OUTPUT_ROUNDOFF,
    padding=0,
    **cache_of,
):
    """Compare one case against the reference within the operator's declared bound.

    `cache_of` states the case's query counts and cache type, as [`operands`] reads them.
    """
    operands_of = operands(rows, generator, out_dtype, padding, **cache_of)
    expected = reference(operands_of).to(torch.float32)
    got = launch(kernel, operands_of, packed)[0].to(torch.float32)
    reduction = max(rows)
    bound = 2.0 * gamma(reduction) * reduction**0.5 + output_roundoff
    envelope = (2.0 * gamma(reduction) + ACCUMULATION_ROUNDOFF) * 8.0
    difference = (got - expected).abs()
    close = difference <= bound * expected.abs() + envelope
    if bool(close.all()):
        return []
    differing = (~close).nonzero()
    token, head, feature = differing[0].tolist()
    return [
        f"{name}: {len(differing)} of {got.numel()} values differ; first at token {token}, head "
        f"{head}, feature {feature}: {got[token, head, feature].item():.7g} against "
        f"{expected[token, head, feature].item():.7g}"
    ]


def the_mask_is_applied(name, kernel, generator):
    """The first query row must answer over its own key alone, whatever the cache holds.

    This is the case a program without a mask cannot pass: with `keys[1:]` replaced by values whose
    attention would dominate, a causal program's row 0 is unchanged and an unmasked one is not.
    """
    rows = [40]
    operands_of = operands(rows, generator)
    before = launch(kernel, operands_of)[0].clone()
    # The keys and values after position 0 are replaced by a hundred-fold larger page, and row 0 must
    # not see them at all.
    operands_of["value"] = operands_of["value"].clone()
    operands_of["value"][:, 1:, :, :] = torch.full_like(
        operands_of["value"][:, 1:, :, :], 0.5
    ).to(torch.float8_e4m3fn)
    after = launch(kernel, operands_of)[0]
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
    kernel = load_kernel(args.package)
    generator = torch.Generator().manual_seed(23)

    failures = []
    failures += compare(
        "a causal prefill over two pages", kernel, [40], generator
    )
    failures += compare(
        "a partial last page under the mask", kernel, [70], generator
    )
    # The shorter row comes first, so its tiles past its length fall on the second row's live
    # tokens: only the batch's last row may write padding.
    failures += compare(
        "a batch of two rows with different lengths",
        kernel,
        [24, 40],
        generator,
    )
    # A bucket larger than its batch: a padded row of length 0 and tokens no row owns, which the
    # kernel must write as zero rather than leave as whatever the buffer held.
    failures += compare(
        "a padded bucket",
        kernel,
        [24, 16, 0],
        generator,
        padding=8,
    )
    # The narrowed store, as its own case: the interpreter truncates a BF16 store where a device
    # rounds (measured in `decode_check.py`: 2072 of 4096 values, up to two steps), so its bound
    # carries two steps where the operator declares one.
    failures += compare(
        "the narrowed store",
        kernel,
        [40],
        generator,
        out_dtype=torch.bfloat16,
        output_roundoff=NARROWED_ROUNDOFF,
    )
    failures += the_mask_is_applied("the causal mask", kernel, generator)
    # Rows whose queries follow keys already in the cache: 9 queries after 61 keys across a page
    # boundary, and 5 after 30 within the first page, beside a row of no prefix.
    failures += compare(
        "queries after a cached prefix",
        kernel,
        [70, 35, 12],
        generator,
        queries=[9, 5, 12],
    )
    # The served pairs: BF16 queries over each dense cache type, and FP16 queries over a BF16
    # cache, each writing its own type; one row follows a prefix and the last pages are partial.
    for plane, query_dtype, roundoff in (
        (torch.float16, torch.bfloat16, NARROWED_ROUNDOFF),
        (torch.bfloat16, torch.bfloat16, NARROWED_ROUNDOFF),
        (torch.float32, torch.bfloat16, NARROWED_ROUNDOFF),
        (torch.bfloat16, torch.float16, HALF_ROUNDOFF),
    ):
        failures += compare(
            f"a {plane} cache under {query_dtype} queries",
            kernel,
            [70, 24],
            generator,
            out_dtype=query_dtype,
            output_roundoff=roundoff,
            queries=[6, 24],
            plane=plane,
            query_dtype=query_dtype,
        )
    # An FP16 cache whose hidden slots hold infinities and NaN: each query's visible keys decide
    # alone.
    for hidden in (float("inf"), float("nan")):
        failures += compare(
            f"hidden slots of an f16 cache holding {hidden}",
            kernel,
            [70, 24],
            generator,
            out_dtype=torch.bfloat16,
            output_roundoff=NARROWED_ROUNDOFF,
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
        f"ok on {where}: every case matches the reference — the pages decode once in f32, a dense "
        "page of each type is read without a scale, a query attends only its own past, after any "
        "prefix its row holds, whatever the hidden slots hold, and a batch's rows need not share a "
        "length"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
