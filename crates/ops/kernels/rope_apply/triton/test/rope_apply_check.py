#!/usr/bin/env python3
"""Run `rope_apply`'s kernel and check the rotation alone, on the CPU.

    crates/program-server/python/.venv/bin/python rope_apply_check.py [--package DIR]

The tables arrive already evaluated — one unit pair per turned feature and token — so the call costs
only the rotation of each head's leading `ROTATED` channels:

```text
out[t,h,low]  = x[t,h,low]·cos − x[t,h,high]·sin      half: low = j, high = j + ROTATED/2
out[t,h,high] = x[t,h,high]·cos + x[t,h,low]·sin      interleaved: low = 2j, high = 2j + 1
out[t,h,k≥ROTATED] = x[t,h,k]                        a partial rotation copies the rest
```

A head need not be a power of two — the fixture's is 72, which is where a kernel that rounds its tile
up and forgets the mask writes past the row.

The cases, each named for the mistake it catches:

- **the fixture's cell** — `bf16` in and out, a 72-channel head, 16 heads, `half`, every channel
  turned, and a token count that does not fill the last program's rows;
- **`half` against `interleaved`** on the same data: a kernel using the other pairing cannot agree
  with both;
- **a partial rotation** (32 of 64, and 96 of 128): the channels past the rotation must be the input,
  so the output is pre-filled with a sentinel and every untouched channel asserted equal to `x`;
- **`f32` and `f16` outputs**, and one token with the full eight rows of a program's block idle.

As in `qk_norm_rope`, the subtraction `x[low]·cos − x[high]·sin` cancels, so the operator's relative
factor `2γ(2) + u` is applied to the **terms** (`|low·cos| + |high·sin|`) rather than to the result —
the forward bound for a cancelling sum, and stronger than the operator's where the element does not
cancel. The store is separated from the arithmetic: the kernel is launched twice, with an `f32`
output for the arithmetic and with the case's output for the store, and the narrow result must be bit
for bit a rounded or truncated copy of the wide one, whichever the interpreter does for that format.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import sys

# This is an interpreter check: the arithmetic is compared against a CPU reference, so the kernel runs
# on Triton's interpreter even where a device exists. The flag is set before any kernel module is
# imported, because Triton reads it when it launches.
if os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    print("running on Triton's interpreter (no device needed)")

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: Which store model the interpreter followed for each format, as measured case by case.
STORES = set()


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def factor(output):
    """`Tolerance::rounding(2, f32, output)`'s relative factor: two products, one rounding out."""
    unit = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11, torch.float32: 2.0**-24}[output]
    return 2.0 * gamma(2, 2.0**-24) + unit


def store_models(element, values):
    """The two ways a narrow store can leave a value: rounded to nearest, and truncated."""
    nearest = values.to(element)
    keep = 8 if element == torch.bfloat16 else 11
    bits = values.contiguous().view(torch.int64)
    truncated = (bits & ~((1 << (53 - keep)) - 1)).view(torch.float64).to(element)
    return {"rounded to nearest": nearest, "truncated": truncated}


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "rope_apply.py"
    spec = importlib.util.spec_from_file_location("rope_apply_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(*, dtype, head_dim, heads, rotated, rows, block, tokens, interleaved, seed=23):
    """The case's inputs, distinct in every element and with distinct tables per token."""
    element = DTYPES[dtype]
    torch.manual_seed(seed)
    values = torch.randn(tokens, heads, head_dim, dtype=torch.float32).to(element)
    half = rotated // 2
    angle = torch.rand(tokens, half, dtype=torch.float32) * 2.0 * math.pi
    return {
        "x": values,
        "cosine": torch.cos(angle),
        "sine": torch.sin(angle),
        "tokens": tokens,
        "head_dim": head_dim,
        "heads": heads,
        "rotated": rotated,
        "rows": rows,
        "block": block,
        "interleaved": interleaved,
    }


def reference(case):
    """The expected output, and the terms each element's error is measured against."""
    x = case["x"].to(torch.float64)
    half = case["rotated"] // 2
    if case["interleaved"]:
        low_column, high_column = 2 * torch.arange(0, half), 2 * torch.arange(0, half) + 1
    else:
        low_column = torch.arange(0, half)
        high_column = low_column + half
    cos = case["cosine"].to(torch.float64).unsqueeze(1)
    sin = case["sine"].to(torch.float64).unsqueeze(1)
    expected = x.clone()
    scale = x.abs().clone()
    low, high = x[..., low_column], x[..., high_column]
    expected[..., low_column] = low * cos - high * sin
    expected[..., high_column] = high * cos + low * sin
    scale[..., low_column] = (low * cos).abs() + (high * sin).abs()
    scale[..., high_column] = (high * cos).abs() + (low * sin).abs()
    return expected, scale


def launch(module, case, output):
    """One launch of the entry the descriptor selects."""
    out_element = DTYPES[output]
    narrowed = case["rotated"] != case["head_dim"]
    out = torch.full((case["tokens"], case["heads"], case["head_dim"]), float("nan"),
                     dtype=out_element)
    words = [case["x"], case["cosine"], case["sine"], out, case["tokens"], 0]
    if narrowed:
        words += [case["rotated"]]
    entry = module._rope_apply_narrowed_kernel if narrowed else module._rope_apply_full_kernel
    entry[((case["tokens"] + case["rows"] - 1) // case["rows"], 1, 1)](
        *words,
        ROTATED=case["rotated"], PAIRS=case["rotated"] // 2,
        PAIRS_TILE=1 << (case["rotated"] // 2 - 1).bit_length(),
        HEAD_DIM=case["head_dim"], HEADS=case["heads"], ROWS=case["rows"],
        HEADS_TILE=case["block"], INTERLEAVED=case["interleaved"], PDL=False,
    )
    return out


def check(module, what, *, output, interleaved, **case_args):
    """One case: the arithmetic on an `f32` launch, then the store bit for bit."""
    case = build(interleaved=interleaved, **case_args)
    where = (f"{what}: {output} out of {case_args['dtype']} in, {case['tokens']} tokens, "
             f"{case['heads']} heads of {case['head_dim']}, {case['rotated']} turned "
             f"{'interleaved' if interleaved else 'half'}")

    expected, scale = reference(case)
    wide = launch(module, case, "f32")
    relative = factor(torch.float32)
    allowed = relative * scale
    error = (wide.to(torch.float64) - expected).abs()
    violations = int((~(error <= allowed)).sum())
    if violations:
        position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{where}: the arithmetic is outside the bound at {violations} of {wide.numel()} "
            f"elements; first at {position}: {float(wide.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}, terms "
            f"{float(scale.reshape(-1)[position])}, factor {relative:.3e}"
        )

    if DTYPES[output] == torch.float32:
        return wide.numel()

    narrow = launch(module, case, output)
    models = store_models(DTYPES[output], wide.to(torch.float64))
    agreeing = [kind for kind, model in models.items() if bool((narrow == model).all())]
    if not agreeing:
        position = int((narrow != models["truncated"]).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{where}: the stored value is neither a rounded nor a truncated copy of the wide "
            f"result; first at {position}: {float(narrow.reshape(-1)[position])} vs nearest "
            f"{float(models['rounded to nearest'].reshape(-1)[position])} and truncated "
            f"{float(models['truncated'].reshape(-1)[position])}"
        )
    STORES.add(f"{output} {agreeing[0]}")
    return wide.numel() + narrow.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/rope_apply.py",
    )
    options = parser.parse_args()
    module = load_kernel(options.package)

    checked = 0
    # The fixture's cell: a 72-channel head, every channel turned, rows that leave a tail.
    checked += check(module, "the fixture's cell", output="bf16", interleaved=False,
                     dtype="bf16", head_dim=72, heads=16, rotated=72, rows=8, block=128,
                     tokens=11)
    # The two pairings on the same data.
    for interleaved in (False, True):
        checked += check(module, "the pairings", output="f32", interleaved=interleaved,
                         dtype="bf16", head_dim=64, heads=8, rotated=64, rows=4, block=64,
                         tokens=6)
    # Partial rotations: the channels past the turn must be the input, untouched.
    checked += check(module, "a partial rotation", output="bf16", interleaved=False,
                     dtype="f32", head_dim=128, heads=32, rotated=96, rows=2, block=64,
                     tokens=5)
    checked += check(module, "a quarter rotation", output="f16", interleaved=True,
                     dtype="bf16", head_dim=64, heads=4, rotated=32, rows=1, block=64,
                     tokens=3)
    # One token against a program's whole block of rows.
    checked += check(module, "one token", output="f32", interleaved=False,
                     dtype="f32", head_dim=96, heads=2, rotated=96, rows=8, block=64,
                     tokens=1)
    print(f"ok: {checked} rotated values over five cells — the arithmetic at rounding(2)'s factor "
          f"on the terms, and the store bit for bit ({', '.join(sorted(STORES))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
