#!/usr/bin/env python3
"""Run `weighted_rms_norm`'s two programs and check them against the operator's own formula.

    crates/program-server/python/.venv/bin/python weighted_rms_norm_check.py [--package DIR]

One program normalizes one contiguous row and applies its learned scale, with the mean square
reduced in FP32 and one rounding into the output type:

```text
inverse  = rsqrt(mean(h²) + ε)          the statistic over the whole row
out[r,k] = h[r,k] · inverse · w[k]      one weight per channel, every row
```

With `absmax` the program also stores, in `partials[r]`, the largest magnitude of the row **it
stored** — measured after the rounding, over the whole row and nothing outside it. The caller
reduces those per-row values to one scalar in a separate launch; this check holds the program to the
per-row magnitudes, which is what the reduction can read.

**What is compared, and against what.** The reference is the formula above evaluated in FP64, and the
bound is the operator's (`WeightedRmsNorm::tolerance`):

    |got − reference|  ≤  (2γ(columns, u_f32) + u_out) · |reference|

which is the FP32 reduction of `columns` terms plus the single rounding into the output type.

Triton's interpreter narrows a store by **truncation**, not to nearest — a full ulp, which at the
bottom of a binade is twice the operator's `u_out`. So each case is launched **twice**: once with an
`f32` output, where nothing narrows and the comparison is the arithmetic at the bound; and once with
the case's own output type, where the narrow store is required to be **bit for bit** the truncation of
the wide result. Neither half hides the other: a program that computes the wrong value fails the first,
and one that stores something other than its own value fails the second.

The per-row magnitude is held **bit for bit** as well, because a maximum of stored values is exact:
anything else means the program measured a different quantity — before the rounding, without the
weight, or across a lane it does not store.

The kernel reads at most `BLOCK` lanes of a row, so `tile ≥ columns` is the operator's precondition
(`check_descriptor` refuses the other order); the cases below stay inside it, and one of them puts the
row *below* the tile so the masking of the lanes past `columns` is exercised.

The cases, each named for the mistake it catches:

- **the fixture's cell** — `bf16` in and out, 5376 columns in a 8192-lane program, two rows;
- **`f32` out** — the arithmetic alone, with no store rounding to hide behind;
- **a row narrower than the tile** — 100 columns in a 256-lane program, where a kernel that forgot its
  mask reads a tail that is not there;
- **the `absmax` program** — three rows, the per-row magnitudes exact, so a program that measured a
  different quantity is red;
- **a zero row and a tiny one under a large ε** — the epsilon alone keeps the reciprocal finite, and a
  program that dropped it divides by zero;
- **a weight with mixed signs and a zero** — a program that used `|w|`, or skipped the weight, cannot
  agree.
"""

import argparse
import importlib.util
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

#: `DType::unit_roundoff`, per element type.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "weighted_rms_norm.py"
    spec = importlib.util.spec_from_file_location("weighted_rms_norm_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(columns, output):
    """The operator's relative factor: the FP32 reduction, then one rounding into `output`."""
    return 2.0 * gamma(columns, UNIT_ROUNDOFF["f32"]) + UNIT_ROUNDOFF[output]


def reference(hidden, weight, epsilon):
    """The formula in FP64: the row's RMS, its learned scale, one product per element."""
    hidden = hidden.double()
    weight = weight.double()
    mean_square = (hidden * hidden).mean(dim=1, keepdim=True)
    inverse = torch.rsqrt(mean_square + epsilon)
    return hidden * inverse * weight


def compare(label, got, expected, factor):
    """Count the elements outside the operator's bound, and report the largest miss."""
    difference = (got.double() - expected).abs()
    allowed = factor * expected.abs()
    # A difference that is not a number is never inside a bound: `>` says otherwise.
    outside = (~(difference <= allowed)).sum().item()
    largest = (difference / expected.abs().clamp(min=1e-300)).max().item()
    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {label} — max relative error {largest:.3e}, "
        f"bound {factor:.3e}, {outside} of {expected.numel()} outside"
    )
    return outside


def truncate(value, element):
    """The interpreter's store: the mantissa below `element`'s precision cut off, not rounded."""
    wide = value.to(torch.float32)
    bits = wide.view(torch.int32)
    drop = {"bf16": 16, "f16": 13, "f32": 0}[element]
    if drop:
        bits = bits & ~((1 << drop) - 1)
    return bits.view(torch.float32).to(DTYPES[element])


def stored(label, narrow, wide):
    """Hold the narrow store to the truncation of the wide result, element for element."""
    expected = truncate(wide, "bf16" if narrow.dtype == torch.bfloat16 else
                        "f16" if narrow.dtype == torch.float16 else "f32")
    agreeing = torch.equal(narrow, expected)
    print(
        f"{'exact' if agreeing else 'FAIL'}: {label} — the narrow store is "
        f"{'bit for bit' if agreeing else 'not'} the truncation of the wide result"
    )
    return 0 if agreeing else narrow.numel()


def magnitudes(label, got, stored):
    """Hold the per-row magnitudes to the largest stored value of their own row."""
    expected = stored.abs().amax(dim=1)
    exact = torch.equal(got, expected)
    print(
        f"{'exact' if exact else 'FAIL'}: {label} — per-row magnitudes "
        f"{'agree with' if exact else 'differ from'} the stored rows' largest magnitude"
    )
    return 0 if exact else got.numel()


def check(module, label, *, columns, tile, rows, dtype, eps, absmax=False, zeros=False,
          tiny=False, signed_weight=False, seed=1):
    """Build one case, launch it, and compare both what it stores and what it measures."""
    element = DTYPES[dtype]
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, columns, generator=generator).to(element)
    weight = torch.randn(columns, generator=generator).to(element)
    if signed_weight:
        # A negative half and one exact zero: a program using |w| or skipping the weight is red.
        weight[::2] = -weight[::2].abs()
        weight[columns // 2] = 0.0
    if zeros:
        hidden[0, :] = 0.0
    if tiny:
        hidden[-1, :] *= 1e-30
    expected = reference(hidden, weight, eps)

    # The wide launch: an f32 output, so the comparison is the arithmetic at the bound and nothing
    # narrows between the reduction and the answer.
    wide = torch.full((rows, columns), float("nan"), dtype=torch.float32)
    narrow = torch.full((rows, columns), float("nan"), dtype=element)
    # One magnitude buffer per launch: the magnitudes describe what that launch stored, so the
    # comparison reads the one the narrow launch wrote.
    wide_partials = torch.full((rows,), float("nan"), dtype=torch.float32)
    partials = torch.full((rows,), float("nan"), dtype=torch.float32)
    launch = (
        module._weighted_rms_norm_absmax_kernel if absmax else module._weighted_rms_norm_kernel
    )
    if absmax:
        launch[(rows,)](hidden, weight, wide, wide_partials, eps, WIDTH=columns, BLOCK=tile)
        launch[(rows,)](hidden, weight, narrow, partials, eps, WIDTH=columns, BLOCK=tile)
    else:
        launch[(rows,)](hidden, weight, wide, eps, WIDTH=columns, BLOCK=tile)
        launch[(rows,)](hidden, weight, narrow, eps, WIDTH=columns, BLOCK=tile)

    cell = f"{label}: {rows} x {columns}, {dtype}, tile {tile}, ε {eps:g}"
    outside = compare(f"{cell}, f32 out", wide.reshape(rows, columns), expected,
                      bound(columns, "f32"))
    outside += stored(f"{cell}, {dtype} out", narrow.reshape(rows, columns),
                      wide.reshape(rows, columns))
    if absmax:
        # The magnitudes describe the values the program stored, so they are read after the fact.
        outside += magnitudes(f"{label}: the magnitudes", partials, narrow.reshape(rows, columns))
    return rows * columns, outside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/weighted_rms_norm.py",
    )
    options = parser.parse_args()
    module = load_kernel(options.package)

    cases = [
        # The fixture's cell: bf16 both ways, 5376 columns in an 8192-lane program.
        dict(label="the fixture's cell", columns=5376, tile=8192, rows=2, dtype="bf16",
             eps=1e-6),
        # The arithmetic with no store rounding to hide behind.
        dict(label="an f32 store", columns=512, tile=1024, rows=2, dtype="f32", eps=1e-6,
             seed=2),
        # A row narrower than the tile: the lanes past `columns` must be masked.
        dict(label="a row narrower than the tile", columns=100, tile=256, rows=3, dtype="bf16",
             eps=1e-6, seed=3),
        # The absmax program: the per-row magnitudes must describe what it stored.
        dict(label="the absmax program", columns=96, tile=256, rows=3, dtype="bf16", eps=1e-6,
             absmax=True, seed=4),
        # The epsilon: a zero row and a tiny one, where the reciprocal is the epsilon's.
        dict(label="the epsilon", columns=64, tile=64, rows=2, dtype="f32", eps=1e-2,
             zeros=True, tiny=True, seed=5),
        # Mixed-sign weights and one exact zero: the weight multiplies, it does not scale magnitudes.
        dict(label="the weight's sign", columns=64, tile=64, rows=2, dtype="bf16", eps=1e-6,
             signed_weight=True, seed=6),
    ]
    checked, failed = 0, 0
    for case in cases:
        values, outside = check(module, **case)
        checked += values
        failed += outside
    if failed:
        print(f"FAIL: {failed} of {checked} values are outside their bound")
        return 1
    print(
        f"ok: {checked} normalized values over {len(cases)} cells — the arithmetic at the "
        f"operator's own factor, and the absmax program's per-row magnitudes bit for bit"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
