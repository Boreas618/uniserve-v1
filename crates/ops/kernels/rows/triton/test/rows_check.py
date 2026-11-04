#!/usr/bin/env python3
"""Run the `rows` kernel and check it against the operator's two modes, on the CPU.

The kernel is Triton, so the interpreter runs it: this is the arithmetic check a build host can do
without a device, and it is the only one the fixture's cells get here (the store's own build proves
that each cell compiles, not what it computes).

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python rows_check.py [--package DIR]

`replace` copies a source row into the destination row its index names, so it is exact; `add`
accumulates with one rounding into the destination's dtype. Both are compared **bit for bit**: the
reference is the operator's own two lines, the sum formed in FP64 (exact for two BF16 or F16 values
and correct for two FP32 ones) and `stored` reproduces the cast the interpreter itself performs.

The cases are the ones a plausible-but-wrong kernel passes:
- **a row wider than the block**, so the tail program masks lanes, and a row narrower than it;
- **rows that leave most of the destination untouched**, because a kernel that rewrote the whole
  destination would pass every case where every row is named;
- **`add` on rows that already hold a value**, so a copy or a zeroed sum is caught;
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

#: The three element types the fixture's cells store in.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

BLOCK = 1024


def stored(element, exact):
    """What the interpreter's store leaves in ``element``, from the exact FP64 value.

    The interpreter casts F16 and F32 to nearest as the hardware does, and **truncates a BF16
    store**: it drops the low 16 bits of the FP32 mantissa instead of rounding them. That is
    measured, not assumed — a plain store of 4,096 FP32 values into BF16 matches the truncation of
    all 4,096 and torch's nearest rounding of 2,038 of them
    (`notes/programs/kernel-arithmetic-checks.md`). Reproducing the tool's own cast makes the
    comparison exact instead of leaving a two-ulp allowance, which would judge the interpreter
    rather than the kernel.
    """
    exact = exact.to(torch.float64)
    if element is torch.bfloat16:
        # Clear the low 45 bits of the **FP64** mantissa — BF16 keeps seven explicit ones of the
        # fifty-two — and never a narrower type in between: rounding to FP32 first carries a value
        # sitting just below a BF16 boundary up onto it, which puts the reference one ulp above a
        # correct kernel. The cast that follows is exact, because the value already sits on the
        # BF16 grid.
        truncated = exact.contiguous().view(torch.int64) & -(1 << 45)
        return truncated.view(torch.float64).to(torch.bfloat16)
    return exact.to(element)


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "rows.py"
    spec = importlib.util.spec_from_file_location("rows_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, *, dtype, mode, columns, rows, count, indices, seed=17):
    """One launch, and the operator's own two lines of arithmetic as the reference."""
    element = DTYPES[dtype]
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
            expected[target] = stored(
                element,
                expected[target].to(torch.float64) + source[row].to(torch.float64),
            )

    if not torch.equal(destination, expected):
        difference = (destination.to(torch.float64) - expected.to(torch.float64)).abs()
        position = int(difference.reshape(-1).argmax())
        raise SystemExit(
            f"{dtype} {mode} columns {columns} rows {rows} count {count}: "
            f"{int((difference > 0).sum())} of {destination.numel()} elements differ; worst at "
            f"{position}: {float(destination.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}"
        )

    # Rows no index named must be untouched, which no difference bound would catch if the
    # difference were zero.
    touched = torch.zeros(rows, dtype=torch.bool)
    for row in range(count):
        touched[int(table[row])] = True
    if touched.all():
        raise SystemExit(f"{mode}: every row was named, so nothing tests the untouched rows")
    if not torch.equal(destination[~touched], before[~touched]):
        raise SystemExit(f"{mode}: a row no index named was written")
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
    # one block whose tail is not a multiple of it.
    checked += run(
        module, dtype="bf16", mode="replace", columns=5120, rows=40, count=6,
        indices=[0, 39, 7, 7, 12, 3],
    )
    checked += run(
        module, dtype="bf16", mode="add", columns=5120, rows=40, count=6,
        indices=[0, 39, 7, 12, 3, 30],
    )
    checked += run(module, dtype="f16", mode="replace", columns=2048, rows=8, count=3, indices=[1, 6, 2])
    checked += run(module, dtype="f32", mode="add", columns=130, rows=5, count=4, indices=[4, 0, 3, 1])
    checked += run(module, dtype="bf16", mode="replace", columns=5376, rows=6, count=2, indices=[5, 0])
    checked += run(module, dtype="f16", mode="add", columns=5376, rows=6, count=3, indices=[2, 5, 1])
    print(f"ok: {checked} elements, both modes, bit for bit, with rows no index names untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
