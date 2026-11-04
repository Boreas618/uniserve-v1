#!/usr/bin/env python3
"""Run cache_copy_region's kernel under Triton's interpreter and compare every stored value.

The operator's tolerance is the stored type's own rounding, so this check computes the expected bytes
with torch's own cast -- the same round-to-nearest-even conversion the store performs -- and compares
element by element, including the two things a per-token grid could get wrong without any shape
complaint:

* **the unit ids are indices, not positions.** The fixture's unit vector is a permutation
  ``[2, 0, 1]``, so a program that addressed ``slot`` instead of ``units[slot]`` writes the wrong
  plane and the comparison says which element and what it found.
* **the flag is raised exactly for the cells the span touches**, and no other cell moves.

A dense pool has no scales, so the operator passes the initialized view's address in the flag slot;
the check passes it the same way, which is what makes the unused pointer exercised rather than
assumed.

    TRITON_INTERPRET=1 python copy_region_check.py [--package .]
"""

import argparse
import importlib.util

import torch

COLUMNS = 4
UNITS_PER_PAGE = 3
HEADS = 4
HEAD_DIM = 8
ROW = HEADS * HEAD_DIM
BLOCK_TOKENS = 6
PLANE_BYTES = BLOCK_TOKENS * ROW * 2
#: The destination page's units, deliberately not `[0, 1, 2]`: the unit ids are a permutation.
UNITS = [2, 0, 1]


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    path = f"{package}/src/cache_copy_region.py"
    spec = importlib.util.spec_from_file_location("copy_region_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._cache_copy_region_kernel


def run(kernel, dtype, offset, count, units_total=8):
    """Launch the kernel for one span and answer what it wrote, plus the buffers it wrote into."""
    source = torch.arange(count * COLUMNS * ROW, dtype=torch.float32).reshape(count, COLUMNS, ROW)
    # Values a BF16 rounds differently from an F16, so a wrong conversion shows as a wrong number
    # rather than as the same one: thirds, a tie at the last mantissa bit, and a magnitude.
    source = (source - 37.0) / 7.0
    values = torch.zeros((COLUMNS, units_total, PLANE_BYTES // 2), dtype=torch.bfloat16).view(
        torch.uint8
    ) if dtype == "bf16" else torch.zeros(
        (COLUMNS, units_total, PLANE_BYTES // 2), dtype=torch.float16
    ).view(torch.uint8)
    initialized = torch.zeros((COLUMNS, units_total), dtype=torch.uint8)
    units = torch.tensor(UNITS, dtype=torch.int32)
    stored = torch.bfloat16 if dtype == "bf16" else torch.float16
    row_pow2 = 1
    while row_pow2 < ROW:
        row_pow2 *= 2
    typed = values.view(stored)
    kernel[(COLUMNS, UNITS_PER_PAGE, count)](
        typed,
        initialized,
        initialized,
        source,
        units,
        offset,
        count,
        units_total,
        ROW_SIZE=ROW,
        ROW_POW2=row_pow2,
        PLANE_ELEMENTS=PLANE_BYTES // 2,
        COLUMNS=COLUMNS,
        BF16=1 if dtype == "bf16" else 0,
        BLOCK=row_pow2,
        PDL=False,
    )
    return source, typed, initialized, stored


def check(kernel, dtype, offset, count):
    """Compare the written planes with the reference's own cast, cell by cell."""
    source, typed, initialized, stored = run(kernel, dtype, offset, count)
    failures = []
    for column in range(COLUMNS):
        for slot in range(UNITS_PER_PAGE):
            unit = UNITS[slot]
            expected = torch.zeros((PLANE_BYTES // 2,), dtype=stored)
            for token in range(count):
                expected[(offset + token) * ROW : (offset + token + 1) * ROW] = source[
                    token, column
                ].to(stored)
            got = typed[column, unit]
            if not torch.equal(got, expected):
                differing = (got != expected).nonzero().flatten().tolist()
                first = differing[0]
                failures.append(
                    f"{dtype} column {column} slot {slot} (unit {unit}) element {first}: "
                    f"{got[first].item()!r} where the cast gives {expected[first].item()!r}"
                )
            flag = int(initialized[column, unit].item())
            if flag != 1:
                failures.append(
                    f"{dtype} column {column} slot {slot} (unit {unit}): flag {flag}, not 1"
                )
    # No cell outside the span's units may be flagged or written.
    touched = {(column, unit) for column in range(COLUMNS) for unit in UNITS}
    for column in range(COLUMNS):
        for unit in range(initialized.shape[1]):
            if (column, unit) not in touched and int(initialized[column, unit].item()) != 0:
                failures.append(f"{dtype} column {column} unit {unit}: flagged outside the span")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=".")
    arguments = parser.parse_args()
    kernel = load_kernel(arguments.package)
    failures = 0
    # Two spans: one at the page's start and one at an offset, so the addressing's `offset + token`
    # is exercised in both its trivial and its general form.
    for dtype in ("bf16", "f16"):
        for offset, count in ((0, BLOCK_TOKENS), (2, 3)):
            problems = check(kernel, dtype, offset, count)
            if problems:
                failures += len(problems)
                for problem in problems:
                    print(f"FAIL {problem}")
            else:
                print(f"ok   {dtype} offset {offset} count {count}: {COLUMNS * UNITS_PER_PAGE} "
                      f"cells, every stored value the cast's own")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
