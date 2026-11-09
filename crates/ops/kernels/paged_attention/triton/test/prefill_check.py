#!/usr/bin/env python3
"""Run the E4M3 paged attention's prefill kernel under Triton's interpreter and compare what it wrote.

The operator's contract is `ops::attention::PagedAttention`'s; the prefill program differs from
the decode one in exactly two ways, and each is a case here:

* **the causal mask** — a query row at position `p` attends keys `0..=p` of its own batch row, so the
  first row's answer is the answer over one key whatever the rest of the cache holds;
* **a batch's rows are not all the same length** — a packed batch states each row's first query token
  in `cum_seq_lens_q`, an unpacked one gives every row `max_q_len`, and the two statements of the same
  data must produce the same answer.

The reference is the composition the contract states, written in torch:

```text
page_k = f32(byte) · key_scale[page]        once, in f32, before any product
scores = (q · page_kᵀ) · softmax_scale      masked by j <= position and j < seq_len
out    = Σ softmax(scores) · page_v         one rounding into the output's type
```

**What this check does not prove**, as in `decode_check.py`: the interpreter's E4M3 *decode* is exact
for every byte, but its BF16 store truncates where a device rounds, and both sides cast through the
interpreter. So this proves the rule, its order and its masking; the format and the rounding are the
hardware's.

    TRITON_INTERPRET=1 python prefill_check.py [--package DIR]
"""

import argparse
import importlib.util
import pathlib

import torch

Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
PAGE = 32
SET = 0.0625


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    del package  # the check's own location is the package's, wherever it is run from
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "paged_attention.py"
    spec = importlib.util.spec_from_file_location("paged_attention_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._paged_attention_prefill_kernel


def cache(rows, generator):
    """A cache of as many pages as the rows need, each row's own, and each row's first token."""
    pages = sum((length + PAGE - 1) // PAGE for length in rows)
    width = max((length + PAGE - 1) // PAGE for length in rows)
    table = torch.zeros((len(rows), width), dtype=torch.int32)
    first = [0]
    start = 0
    for index, length in enumerate(rows):
        live = (length + PAGE - 1) // PAGE
        table[index, :live] = torch.arange(start, start + live, dtype=torch.int32)
        start += live
        first.append(first[-1] + length)
    key_values = torch.rand((pages, PAGE, KV_HEADS, HEAD_DIM), generator=generator) * 2.0 - 1.0
    value_values = torch.rand((pages, PAGE, KV_HEADS, HEAD_DIM), generator=generator) * 2.0 - 1.0
    cycle = [1.0, 0.5, 2.0, 0.25]
    scales = torch.tensor([cycle[page % len(cycle)] for page in range(pages)], dtype=torch.float32)
    view = scales.view(-1, 1, 1, 1)
    return {
        "page_table": table,
        "seq_lens": torch.tensor(rows, dtype=torch.int32),
        "cum_seq_lens_q": torch.tensor(first, dtype=torch.int32),
        "key": (key_values / view).to(torch.float8_e4m3fn),
        "value": (value_values / view).to(torch.float8_e4m3fn),
        "key_scale": view,
        "value_scale": view.clone(),
    }


def operands(rows, generator, out_dtype=torch.float32):
    """The query, the cache and the output buffer, in the layouts the operator states."""
    tokens = torch.rand((sum(rows), Q_HEADS, HEAD_DIM), generator=generator) * 2.0 - 1.0
    out = torch.zeros((sum(rows), Q_HEADS, HEAD_DIM), dtype=out_dtype)
    state = cache(rows, generator)
    return {
        "query": tokens,
        "out": out,
        "max_q_len": max(rows),
        "packed": True,
        **state,
    }


def reference(operands_of, dtype=torch.float32):
    """The contract's composition, in torch, causally masked per row."""
    query = operands_of["query"].to(dtype)
    key = operands_of["key"].to(dtype) * operands_of["key_scale"].to(dtype)
    value = operands_of["value"].to(dtype) * operands_of["value_scale"].to(dtype)
    table = operands_of["page_table"]
    lengths = operands_of["seq_lens"].tolist()
    first = operands_of["cum_seq_lens_q"].tolist()
    out = torch.zeros_like(query)
    for row, seq_len in enumerate(lengths):
        live = (seq_len + PAGE - 1) // PAGE
        pages = table[row, :live].tolist()
        keys = key[pages].reshape(-1, KV_HEADS, HEAD_DIM)
        values = value[pages].reshape(-1, KV_HEADS, HEAD_DIM)
        for position in range(first[row + 1] - first[row]):
            for head in range(Q_HEADS):
                kv_head = head * KV_HEADS // Q_HEADS
                scores = (keys[:, kv_head, :] @ query[first[row] + position, head, :]) * SET
                shown = torch.arange(keys.shape[0])
                visible = (shown <= position) & (shown < seq_len)
                scores = torch.where(visible, scores, torch.full_like(scores, float("-inf")))
                weights = torch.softmax(scores, dim=0)
                out[first[row] + position, head, :] = weights @ values[:, kv_head, :]
    return out


def launch(kernel, operands_of, packed=True):
    """Launch one prefill and answer what it wrote."""
    narrowed = operands_of["out"].dtype == torch.bfloat16
    tokens = operands_of["query"].shape[0]
    kernel[(operands_of["max_q_len"], Q_HEADS, operands_of["page_table"].shape[0])](
        operands_of["out"],
        operands_of["query"],
        operands_of["key"],
        operands_of["value"],
        operands_of["key_scale"],
        operands_of["value_scale"],
        operands_of["page_table"],
        operands_of["seq_lens"],
        operands_of["cum_seq_lens_q"],
        operands_of["query"].stride(0),
        operands_of["query"].stride(1),
        operands_of["query"].stride(2),
        operands_of["key"].stride(0),
        operands_of["key"].stride(1),
        operands_of["key"].stride(2),
        operands_of["key"].stride(3),
        operands_of["value"].stride(0),
        operands_of["value"].stride(1),
        operands_of["value"].stride(2),
        operands_of["value"].stride(3),
        operands_of["out"].stride(0),
        operands_of["out"].stride(1),
        operands_of["out"].stride(2),
        operands_of["page_table"].shape[1],
        PAGE,
        SET,
        operands_of["max_q_len"],
        1 if packed else 0,
        HEAD_DIM=HEAD_DIM,
        PAGE=PAGE,
        GROUP=Q_HEADS // KV_HEADS,
        QUANTIZED=True,
        OUT_BF16=narrowed,
    )
    return operands_of["out"], tokens


ACCUMULATION_ROUNDOFF = 2.0**-24
OUTPUT_ROUNDOFF = 2.0**-8


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
):
    """Compare one case against the reference within the operator's declared bound."""
    operands_of = operands(rows, generator, out_dtype)
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
    parser.add_argument("--package", default=".", help="the package directory (default: this one)")
    args = parser.parse_args()
    kernel = load_kernel(args.package)
    generator = torch.Generator().manual_seed(23)

    failures = []
    failures += compare("a causal prefill over two pages", kernel, [40], generator)
    failures += compare("a partial last page under the mask", kernel, [70], generator)
    failures += compare(
        "a batch of two rows with different lengths", kernel, [24, 40], generator
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
        output_roundoff=2.0 * OUTPUT_ROUNDOFF,
    )
    failures += the_mask_is_applied("the causal mask", kernel, generator)

    if failures:
        print("failed:")
        for line in failures:
            print("  " + line)
        return 1
    print(
        "ok: every case matches the reference — the pages decode once in f32, a query row attends "
        "only its own past, and a batch's rows need not share a length"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
