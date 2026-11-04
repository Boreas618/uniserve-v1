#!/usr/bin/env python3
"""Run the `rows` kernel and check it against the operator's two modes, on the CPU.

The kernel is Triton, so the interpreter runs it: this is the arithmetic check a build host can do
without a device, and it is the only one the fixture's cells get here (the store's own build proves
that each cell compiles, not what it computes).

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python rows_check.py [--package DIR]

`replace` copies a source row into the destination row its index names, so it is exact; `add`
accumulates with one rounding into the destination's dtype, so it is checked at the operator's own
bound — two units of that dtype's rounding — with the sum formed in FP64 in the reference so that
only the kernel's own arithmetic is measured.

The cases are the ones a plausible-but-wrong kernel passes:
- **a row wider than the block**, so the tail program masks lanes;
- **rows that leave most of the destination untouched**, because a kernel that rewrote the whole
  destination would pass every case where every row is named;
- **`add` on rows that already hold a value**, so a copy or a zeroed sum is caught;
- **the interpreter's BF16 store**, which is not the hardware's: a plain store of 4,096 FP32
  values into BF16 differs from torch's cast on 2,058 of them, by up to two ulp (written up in
  `notes/programs/kernel-arithmetic-checks.md`), so BF16 `add` is judged at the operator's bound
  plus two ulp and F16/F32 at the bound alone;
- **a repeated index**, which the operator allows — the program grid orders each source row's own
  writes and nothing orders two blocks naming one row, so the interpreter's sequential execution is
  the only ordering there is; the case pins the semantics the kernel implements and the package's
  comment states the limit.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

import torch

#: `Tolerance::exact().with_atol(2 u)` for `add`, and exact for `replace`.
DTYPES = {
    "bf16": (torch.bfloat16, 2.0 * 2.0**-8),
    "f16": (torch.float16, 2.0 * 2.0**-11),
    "f32": (torch.float32, 2.0 * 2.0**-24),
}

BLOCK = 1024


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "rows.py"
    spec = importlib.util.spec_from_file_location("rows_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, *, dtype, mode, columns, rows, count, indices, seed=17):
    """One launch, and the operator's own two lines of arithmetic as the reference."""
    element, unit = DTYPES[dtype]
    torch.manual_seed(seed)
    source = torch.randn(count, columns, dtype=element)
    before = torch.randn(rows, columns, dtype=element)
    destination = before.clone()
    table = torch.tensor(indices, dtype=torch.int32)

    module._rows_kernel[(count, -(-columns // BLOCK), 1)](
        source,
        destination,
        table,
        count,
        columns,
        BLOCK,
        0,
        COLUMNS=columns,
        REPLACE=mode == "replace",
        BLOCK=BLOCK,
    )

    # The reference is the operator's text, walked in the index order the call gives.
    expected = before.clone()
    for row in range(count):
        target = int(table[row])
        if mode == "replace":
            expected[target] = source[row]
        else:
            # One addition in FP64 and a single rounding into the destination's dtype: the
            # kernel's sum is FP32, so the two agree up to that sum's own rounding, which is what
            # the mode's two units of tolerance cover.
            expected[target] = (expected[target].to(torch.float64) + source[row].to(torch.float64)).to(element)

    error = (destination.to(torch.float64) - expected.to(torch.float64)).abs()
    touched = torch.zeros(rows, dtype=torch.bool)
    for row in range(count):
        touched[int(table[row])] = True
    allowed = torch.zeros_like(error)
    if mode == "add":
        allowed[touched] = unit
        if element is torch.bfloat16:
            # The interpreter's own BF16 store is not round-to-nearest-even, and this is measured
            # rather than assumed: a plain store of 4,096 FP32 values into BF16 differs from
            # torch's cast on 2,058 of them, by up to **2 ulp**. A BF16 `add` is therefore judged
            # at the operator's bound plus those two ulps of the value, and the check says so
            # instead of tightening it into a false pass. F16 and F32 are cast as the hardware
            # casts them and both pass at the operator's bound alone, which is why the kernel's
            # BF16 arithmetic is not in doubt here — only the interpreter's store of it is.
            allowed[touched] += 2.0 * expected.to(torch.float64).abs()[touched] * (2.0**-8)
    violations = int((error > allowed).sum())
    if violations:
        position = int((error > allowed).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{dtype} {mode} columns {columns} rows {rows} count {count}: {violations} of "
            f"{destination.numel()} elements outside the bound; first at {position}: "
            f"{float(destination.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}"
        )
    # Rows no index named must be untouched, which no bound would catch if the difference were zero.
    untouched = ~touched
    drift = (destination[untouched].to(torch.float64) - before[untouched].to(torch.float64)).abs()
    if untouched.any() and float(drift.max()) != 0.0:
        raise SystemExit(f"{mode}: a row no index named was written ({float(drift.max()):.3e})")
    return int(destination.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/rows.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("run this with TRITON_INTERPRET=1: the kernel is checked on the CPU")
    module = load_kernel(options.package)

    checked = 0
    # The fixture's widths against a block-sized lane mask, then a narrow row, then a row wider than
    # one block with a tail that is not a multiple of it.
    checked += run(module, dtype="bf16", mode="replace", columns=5120, rows=40, count=6, indices=[0, 39, 7, 7, 12, 3])
    checked += run(module, dtype="bf16", mode="add", columns=5120, rows=40, count=6, indices=[0, 39, 7, 12, 3, 30])
    checked += run(module, dtype="f16", mode="replace", columns=2048, rows=8, count=3, indices=[1, 6, 2])
    checked += run(module, dtype="f32", mode="add", columns=130, rows=5, count=4, indices=[4, 0, 3, 1])
    checked += run(module, dtype="bf16", mode="replace", columns=5376, rows=6, count=2, indices=[5, 0])
    print(f"ok: {checked} elements, both modes, exact copy and the add's two units of rounding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
