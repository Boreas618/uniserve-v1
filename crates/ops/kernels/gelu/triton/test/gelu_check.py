#!/usr/bin/env python3
"""Run `gelu`'s kernel and check both of PyTorch's GELU forms, on the CPU.

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python gelu_check.py [--package DIR]

```text
exact:  0.5 · x · (1 + erf(x / √2))
tanh:   x · sigmoid(√(8/π) · (x + 0.044715 · x³))
```

The two are separate compilations of one operator, and the operator's bound is
`rounding(1, f32, dtype)` with `8 · 8 · u` of absolute room for the `erf`'s or the `exp`'s last
place. Each form is checked against **its own** formula, and the cases are where the two formulas
part far enough that a kernel evaluating the wrong one cannot hide:

- `x = ±2, ±3, ±6`: the forms differ by about `1e-4` here, three orders past the bound, so a module
  compiled `exact` that evaluates `tanh` (or the other way round) reads red;
- `x = 0` and `x = ±1e-3`: the neighbourhood of the identity, where both forms are `0.5x` and a
  kernel missing the `0.5`, or the `√2` in `erf`, shows;
- `x = ±10` and `x = ±40`: `erf` and the logistic both saturate, where a kernel that clamps its
  argument, or that computes `1 + tanh(u)` in a way that loses the far tail, leaves the function but
  not the bound;
- a row that does not fill the last block, and every dtype the manifest declares.

**The `exact` form cannot run here.** It evaluates `libdevice.erf`, and the interpreter has no
implementation of the libdevice calls — it raises `Did you forget to add @triton.jit?` from
`_semantic` before any arithmetic happens. The script probes that form, reports the gap and its exact
error, and checks what it can; the fixture's two `gelu` descriptors are both `approximate_tanh`, so
what it checks is what the store serves — but the `exact` form's module is unproved until a device run
takes it.

The reference makes the kernel's rounds, in the kernel's order, at FP32 — `x³`, the `0.044715`
product, `√(8/π)`'s product, the exponential, the logistic — and takes only `erf` and `exp`
themselves exactly, in FP64, before rounding them back. See
`notes/programs/kernel-arithmetic-checks.md`: a reference that keeps the whole formula in FP64
measures the distance between two precisions instead.
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

#: `gelu.rs`: one rounding into the output plus the transcendental's last place over `|x| ≤ 8`.
ROUNDINGS = 1
VALUE_ENVELOPE = 8.0
SQRT_8_OVER_PI = 1.5957691216057308
SQRT_HALF = 0.7071067811865476


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(element):
    """`Tolerance::rounding(1, f32, dtype).with_atol(8 · envelope · u)`."""
    unit = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11, torch.float32: 2.0**-24}[element]
    return 2.0 * gamma(ROUNDINGS, 2.0**-24) + unit, 8.0 * VALUE_ENVELOPE * unit


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "gelu.py"
    spec = importlib.util.spec_from_file_location("gelu_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values, approximate):
    """The chosen form at FP32 step by step, with `erf`/`exp` exact and rounded back in."""
    x = values.to(torch.float32)
    if approximate == "tanh":
        inner = (torch.full_like(x, SQRT_8_OVER_PI)
                 * (x + 0.044715 * (x * x) * x))
        exponential = torch.exp((-inner).to(torch.float64)).to(torch.float32)
        return (x / (1.0 + exponential)).to(torch.float64)
    error = torch.erf((x * SQRT_HALF).to(torch.float64)).to(torch.float32)
    return (0.5 * x * (1.0 + error)).to(torch.float64)


def run(module, *, approximate, dtype, values, columns, block, seed=11):
    """One launch over `[rows, columns]`, with the grid the operator computes."""
    element = DTYPES[dtype]
    if values is None:
        torch.manual_seed(seed)
        values = torch.randn(3, columns, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, columns).to(element)
    elements = values.numel()
    out = torch.full_like(values, float("nan"))

    module._gelu_kernel[((elements + block - 1) // block, 1, 1)](
        values, out, elements, columns, 0,
        APPROXIMATE=approximate, BLOCK=block, PDL=False,
    )

    expected = reference(values, approximate)
    got = out.to(torch.float64)
    relative, absolute = bound(element)
    allowed = absolute + relative * expected.abs()
    error = (got - expected).abs()
    violations = int((~(error <= allowed)).sum())
    if violations:
        position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{approximate} {dtype} {values.shape} block {block}: {violations} of {elements} "
            f"elements outside the bound; first at {position}: "
            f"{float(out.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])} "
            f"for x = {float(values.reshape(-1)[position])}, "
            f"rtol {relative:.3e} atol {absolute:.3e}"
        )
    return elements


#: Which form ran, and why any other could not.
FORMS = ["tanh"]
GAP = ""


def main():
    global GAP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/gelu.py",
    )
    options = parser.parse_args()
    module = load_kernel(options.package)

    # The exact form's first step is `libdevice.erf`, which the interpreter cannot evaluate: probe
    # it, keep the error as the reason, and go on with the form that runs.
    try:
        run(module, approximate="exact", dtype="f32", values=[0.0], columns=1, block=256)
    except Exception as error:  # noqa: BLE001 - any failure is the tool's, and its words are the proof
        GAP = (f"the exact form, because the interpreter has no libdevice: "
               f"{type(error).__name__}: {str(error).splitlines()[0]}")
    else:
        FORMS.append("exact")

    # Where the two formulas part: whatever a module was compiled for, this is the case that says it
    # evaluated that one and not the other.
    parting = [0.0, 1e-3, -1e-3, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 6.0, -6.0]
    # Both ends of the saturation, where the transcendental's own argument range matters. At
    # `x = -40` the exponent overflows to infinity and the logistic to zero, which is the form's
    # limit and the right answer; the interpreter warns about the overflow on the way.
    saturated = [10.0, -10.0, 40.0, -40.0, 1e-8, -1e-8]

    checked = 0
    for approximate in FORMS:
        grid = parting + saturated
        for dtype in ("f32", "bf16", "f16"):
            checked += run(module, approximate=approximate, dtype=dtype,
                           values=grid, columns=len(grid), block=256)
        # A wide row with a tail that does not fill the block, so the mask is exercised.
        checked += run(module, approximate=approximate, dtype="bf16",
                       values=None, columns=130, block=256)
        checked += run(module, approximate=approximate, dtype="f32",
                       values=None, columns=1024, block=512)
    if GAP:
        print(f"not checked here: {GAP}")
    print(f"ok: {checked} activated values over {', '.join(FORMS)} at the manifest's dtypes and "
          f"widths, within rounding(1)'s bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
