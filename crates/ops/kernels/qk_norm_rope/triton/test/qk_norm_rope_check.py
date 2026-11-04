#!/usr/bin/env python3
"""Run `qk_norm_rope`'s kernel and check per-head normalization plus rotation, on the CPU.

    crates/program-server/python/.venv/bin/python qk_norm_rope_check.py [--package DIR]

One program normalizes and rotates the query heads and then the key heads of a block of tokens:

```text
row       = x[t, h, :]                                  the whole head, DIM channels
inverse   = rsqrt(mean(row²) + ε)                       statistics over the whole head
row       = row · inverse · weight                      one weight per channel, every head
out[t,h,k] = row[k]                                     k ≥ ROTATED: normalized, not turned
out[t,h,a] = row[a]·cos − row[b]·sin                    the turned pairs, a/b by the rotation kind
out[t,h,b] = row[b]·cos + row[a]·sin
```

with one `cos`/`sin` pair per turned feature of the token — a **unit** pair, as a rotary embedding
has it — and `half` pairing channel `i` with `i + ROTATED/2`, `interleaved` pairing `2i` with `2i+1`.

**How a kernel with a subtractive rotation is checked.** The operator's bound is
`rounding(head_dim, f32, output)`, which is **relative to the result**. That is the wrong denominator
for a rotation: `row[a]·cos − row[b]·sin` can cancel to a thousandth of its terms, and a bound on the
result then fails on an element whose absolute error is one ulp of the terms. So the comparison here
uses the same relative factor `2γ(head_dim) + u` and applies it to the **sum of the magnitudes of the
terms that produced the element** — the textbook bound for a cancelling sum, and a strictly tighter
statement of the operator's own claim wherever the element does not cancel:

    |got − reference|  ≤  (2γ(head_dim) + u) · (|first·cos| + |second·sin|)

**And the store is separated from the arithmetic.** Two launches per case: the same kernel with an
`f32` output is compared against the reference at that bound (the arithmetic), and then the narrow
output is required to be **bit for bit** the truncation of the wide one (the store) — the interpreter
narrows a store by truncation, which this check measures for its own case rather than assuming:
`14336 of 14336` elements of the fixture cell, where nearest rounding agrees on 7130.

The cases, each named for the mistake it catches:

- **the fixture's own cell** — `bf16` in and out, 56 query and 56 key heads, `interleaved`, 96 of
  128 channels turned;
- **`half` against `interleaved`**, on a head whose channels all differ, so a kernel using the other
  pairing cannot agree with both;
- **a rotation narrower than the head** (96 of 128) with the statistic over the whole head: a kernel
  that normalized only the turned channels reads red here and nowhere else;
- **a full-width rotation** (`ROTATED == DIM`), where the kernel skips its first store;
- **a zero row and a tiny row with a large ε**, where the epsilon alone keeps the reciprocal finite
  and a kernel that dropped it writes `nan`;
- **a token count that does not fill the last block of rows**, and **the unweighted entry** (no
  learned scale), which is a separate kernel rather than one with a null pointer.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import struct
import sys

# This is an interpreter check: the arithmetic is compared against a CPU reference, so the kernel runs
# on Triton's interpreter even where a device exists. The flag is set before any kernel module is
# imported, because Triton reads it when it launches.
if os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    print("running on Triton's interpreter (no device needed)")

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

ENTRIES = {
    (True, False): "_qk_norm_rope_kernel",
    (True, True): "_qk_norm_rope_narrowed_kernel",
    (False, False): "_qk_norm_rope_unweighted_kernel",
    (False, True): "_qk_norm_rope_unweighted_narrowed_kernel",
}


#: Which store model the interpreter followed, per output format, as it is measured case by case.
STORES = set()


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def relative(head_dim, element):
    """`flooring(head_dim, f32, element)`'s relative factor: the head's reduction, one rounding out."""
    unit = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11, torch.float32: 2.0**-24}[element]
    return 2.0 * gamma(head_dim, 2.0**-24) + unit


def store_models(element, values):
    """The two ways a narrow store can leave a value: rounded to nearest, and truncated.

    Which one Triton's interpreter does is **measured** rather than assumed, and it is not the same
    for both formats: `bf16` stores truncate (14336 of 14336 elements of the fixture cell) and `f16`
    stores round to nearest (199 of 384 elements differ from the truncation, and none from the
    nearest), which is the kind of difference that reads as a kernel defect if a check guesses.
    """
    nearest = values.to(element)
    keep = 8 if element == torch.bfloat16 else 11
    bits = values.contiguous().view(torch.int64)
    truncated = (bits & ~((1 << (53 - keep)) - 1)).view(torch.float64).to(element)
    return {"rounded to nearest": nearest, "truncated": truncated}


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "qk_norm_rope.py"
    spec = importlib.util.spec_from_file_location("qk_norm_rope_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(side, weight, cosine, sine, *, head_dim, rotated, interleaved, epsilon, affine):
    """One side's expected values, and the scale each element's error is measured against.

    Returns `(expected, scale)`: `expected` is the operator's arithmetic in FP64, and `scale` is the
    sum of the magnitudes of the terms that produced each element — the denominator the operator's
    relative bound needs when a rotation cancels.
    """
    values = side.to(torch.float64)
    inverse = 1.0 / torch.sqrt((values * values).mean(dim=-1, keepdim=True) + epsilon)
    normalized = values * inverse
    if affine:
        normalized = normalized * weight.to(torch.float64)

    expected = normalized.clone()
    scale = normalized.abs().clone()
    half = rotated // 2
    if interleaved:
        a = torch.arange(0, half) * 2
        b = a + 1
    else:
        a = torch.arange(0, half)
        b = a + half
    first, second = normalized[..., a], normalized[..., b]
    # One unit pair per turned feature of the token, shared by every head of that token.
    cos = cosine.to(torch.float64).unsqueeze(1)
    sin = sine.to(torch.float64).unsqueeze(1)
    expected[..., a] = first * cos - second * sin
    expected[..., b] = second * cos + first * sin
    scale[..., a] = (first * cos).abs() + (second * sin).abs()
    scale[..., b] = (second * cos).abs() + (first * sin).abs()
    return expected, scale


def build(*, dtype, head_dim, q_heads, k_heads, rotated, rows, tokens, epsilon, affine=True,
          zeros=False, tiny=False, seed=19):
    """The case's inputs: distinct in every element, so a row written elsewhere cannot pass."""
    element = DTYPES[dtype]
    torch.manual_seed(seed)
    query = torch.randn(tokens, q_heads, head_dim, dtype=torch.float32)
    key = torch.randn(tokens, k_heads, head_dim, dtype=torch.float32)
    if zeros:
        query[0, 0] = 0.0
        key[0, 0] = 0.0
    if tiny:
        query[1 % tokens, 0] *= 1e-6
    half = rotated // 2
    angle = torch.rand(tokens, half, dtype=torch.float32) * 2.0 * math.pi
    return {
        "query": query.to(element),
        "key": key.to(element),
        "q_weight": torch.rand(head_dim, dtype=torch.float32) + 0.5,
        "k_weight": torch.rand(head_dim, dtype=torch.float32) + 0.5,
        "cosine": torch.cos(angle),
        "sine": torch.sin(angle),
        "tokens": tokens,
        "epsilon": epsilon,
        "head_dim": head_dim,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "rotated": rotated,
        "rows": rows,
        "affine": affine,
    }


def launch(module, case, output, interleaved):
    """One launch of the entry the descriptor selects, returning the query and key results."""
    out_element = DTYPES[output]
    query, key = case["query"], case["key"]
    head_dim, rotated = case["head_dim"], case["rotated"]
    affine = case["affine"]
    q_out = torch.full((case["tokens"], case["q_heads"], head_dim), float("nan"),
                       dtype=out_element)
    k_out = torch.full((case["tokens"], case["k_heads"], head_dim), float("nan"),
                       dtype=out_element)
    words = [query, key]
    if affine:
        words += [case["q_weight"], case["k_weight"]]
    words += [case["cosine"], case["sine"], q_out, k_out, case["tokens"], case["epsilon"], 0]
    if rotated != head_dim:
        words += [rotated]
    # `build.py`'s own lane count: a lane count is a power of two, and 96 turned channels are 48
    # pairs, so the block is 64 lanes wide and the last 16 are masked off.
    entry = getattr(module, ENTRIES[(affine, rotated != head_dim)])
    entry[((case["tokens"] + case["rows"] - 1) // case["rows"], 1, 1)](
        *words,
        Q_HEADS=case["q_heads"], K_HEADS=case["k_heads"], DIM=head_dim, ROTATED=rotated,
        PAIRS=1 << (rotated // 2 - 1).bit_length(), ROWS=case["rows"],
        INTERLEAVED=interleaved, AFFINE=affine, PDL=False,
    )
    return q_out, k_out


def check(module, what, *, output, interleaved, **case_args):
    """One case: the arithmetic on an `f32` launch, then the store bit for bit."""
    case = build(**case_args)
    head_dim, rotated = case["head_dim"], case["rotated"]
    expected_q, scale_q = reference(case["query"], case["q_weight"], case["cosine"], case["sine"],
                                    head_dim=head_dim, rotated=rotated, interleaved=interleaved,
                                    epsilon=case["epsilon"], affine=case["affine"])
    expected_k, scale_k = reference(case["key"], case["k_weight"], case["cosine"], case["sine"],
                                    head_dim=head_dim, rotated=rotated, interleaved=interleaved,
                                    epsilon=case["epsilon"], affine=case["affine"])
    where = (f"{what}: {output} out of {case_args['dtype']} in, {case_args['tokens']} tokens, "
             f"{case['q_heads']}+{case['k_heads']} heads of {head_dim}, "
             f"{rotated} turned {'interleaved' if interleaved else 'half'}")

    wide_q, wide_k = launch(module, case, "f32", interleaved)
    factor = relative(head_dim, torch.float32)
    for name, got, expected, scale in (("query", wide_q, expected_q, scale_q),
                                       ("key", wide_k, expected_k, scale_k)):
        values = got.to(torch.float64)
        allowed = factor * scale
        error = (values - expected).abs()
        violations = int((~(error <= allowed)).sum())
        if violations:
            position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
            raise SystemExit(
                f"{where}: the {name}'s arithmetic is outside the bound at {violations} of "
                f"{got.numel()} elements; first at {position}: "
                f"{float(got.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}, "
                f"terms {float(scale.reshape(-1)[position])}, factor {factor:.3e}"
            )

    if DTYPES[output] == torch.float32:
        return wide_q.numel() + wide_k.numel()

    narrow_q, narrow_k = launch(module, case, output, interleaved)
    stored = 0
    for name, narrow, wide in (("query", narrow_q, wide_q), ("key", narrow_k, wide_k)):
        models = store_models(DTYPES[output], wide.to(torch.float64))
        agreeing = [kind for kind, model in models.items() if bool((narrow == model).all())]
        if not agreeing:
            kind, model = next(iter(models.items()))
            differing = int((narrow != model).sum())
            position = int((narrow != models["truncated"]).reshape(-1).nonzero()[0])
            raise SystemExit(
                f"{where}: the stored {name} is neither a rounded nor a truncated copy of the wide "
                f"result ({differing} of {narrow.numel()} differ from the nearest rounding); first "
                f"at {position}: {float(narrow.reshape(-1)[position])} vs nearest "
                f"{float(models['rounded to nearest'].reshape(-1)[position])} and truncated "
                f"{float(models['truncated'].reshape(-1)[position])}"
            )
        STORES.add(agreeing[0])
        stored += narrow.numel()
    return wide_q.numel() + wide_k.numel() + stored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/qk_norm_rope.py",
    )
    options = parser.parse_args()
    module = load_kernel(options.package)
    fixture_epsilon = struct.unpack("<f", bytes.fromhex("acc52737"))[0]

    checked = 0
    # The fixture's cell: bf16 both ways, 56 + 56 heads, interleaved, 96 of 128 turned.
    checked += check(module, "the fixture's cell", output="bf16", interleaved=True,
                     dtype="bf16", head_dim=128, q_heads=56, k_heads=56, rotated=96, rows=1,
                     tokens=2, epsilon=fixture_epsilon)
    # The two pairings on the same data: a kernel that used the other one cannot agree with both.
    for interleaved in (False, True):
        checked += check(module, "the pairings", output="f32", interleaved=interleaved,
                         dtype="bf16", head_dim=128, q_heads=4, k_heads=2, rotated=64, rows=2,
                         tokens=3, epsilon=fixture_epsilon)
    # A full-width rotation: the narrowing branch is off, so the kernel's first store is skipped.
    checked += check(module, "a full-width rotation", output="f32", interleaved=True,
                     dtype="f32", head_dim=64, q_heads=2, k_heads=1, rotated=64, rows=4,
                     tokens=6, epsilon=1e-6)
    # The epsilon: a zero row and a tiny one, with the statistic still over the whole head.
    checked += check(module, "the epsilon", output="bf16", interleaved=False,
                     dtype="bf16", head_dim=64, q_heads=3, k_heads=3, rotated=32, rows=2,
                     tokens=2, epsilon=1e-1, zeros=True, tiny=True)
    # The unweighted entry: no learned scale, half of the head turned, f16 out.
    checked += check(module, "the unweighted entry", output="f16", interleaved=False,
                     dtype="f16", head_dim=64, q_heads=2, k_heads=2, rotated=32, rows=1,
                     tokens=3, epsilon=1e-5, affine=False)
    print(f"ok: {checked} normalized and rotated values over six cells — the arithmetic at "
          f"rounding(head_dim)'s factor on its terms, and the narrow store bit for bit "
          f"({', '.join(sorted(STORES))}, as the interpreter does it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
