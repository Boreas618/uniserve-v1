#!/usr/bin/env python3
"""Check the attention_gather_rows kernel on this host, with no GPU.

    TRITON_INTERPRET=1 python interp_check.py

Triton's interpreter runs the kernel's own Python, so this checks the program the
store's module is compiled from — the section offsets, the three program kinds, the
window arithmetic and the address of each token.

The reference is written here from the operator's documentation rather than from
this kernel, in the plainest form the semantics allow: for each table, each row's
first page, each cell of its block table, and each token's append address. Every
output is an index or a copy of one, so the comparison is bit-exact — a gather that
is off by one stride still produces plausible numbers, which is why a tolerance would
be the wrong instrument.

Three scenarios, each for what it can break:

* a dense table, `window` unset, four tokens with `block` four — a section boundary
  exactly at a row's edge;
* a table whose storage is **not contiguous** (rows longer than the width, tables
  further apart than the rows need) and whose slots installed fewer units than
  `request_width`, so a cell past the selected pages and a unit the slot never
  installed are both exercised;
* a **windowed** table, where a row's block table starts at
  `max(0, prefix - window) // block_size` rather than at page zero, so the slot's
  installed start page is subtracted rather than assumed zero.
"""

import os
import pathlib
import sys

if os.environ.get("TRITON_INTERPRET") != "1":
    print(
        "this check runs the kernel through Triton's interpreter: "
        "TRITON_INTERPRET=1 python interp_check.py",
        file=sys.stderr,
    )
    raise SystemExit(2)

kernel_source = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(kernel_source))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenario import Scenario  # noqa: E402


def report(name, scenario):
    """Compare one scenario's outputs against the reference, bit for bit."""
    expected = scenario.reference()
    got = scenario.run()
    failures = []
    for output in sorted(expected):
        mask = scenario.written(output, expected[output].size)
        differences = (got[output] != expected[output]) & mask
        if differences.any():
            index = tuple(int(value) for value in np.argwhere(differences)[0])
            failures.append(
                f"{output} at {index}: got {int(got[output][index])}, "
                f"expected {int(expected[output][index])}"
            )
    print(
        f"{name}: {scenario.tables} table(s), {scenario.rows} rows, "
        f"{scenario.tokens} tokens, width {scenario.width}, block {scenario.block}, "
        f"window {scenario.window}, {'exact' if not failures else 'DIFFERS'}"
    )
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    return failures


def main() -> int:
    scenarios = [
        (
            "a dense table, a token section that ends at a row's edge",
            Scenario(rows=3, tokens=4, tables=1, width=4, block=4, seed=1),
        ),
        (
            "non-contiguous storage and slots that installed fewer units",
            Scenario(
                rows=4,
                tokens=9,
                tables=2,
                width=6,
                block=8,
                request_width=3,
                row_padding=2,
                table_padding=5,
                seed=2,
            ),
        ),
        (
            "a windowed table",
            Scenario(
                rows=2,
                tokens=3,
                tables=1,
                width=4,
                block=4,
                block_size=2,
                window=2,
                seed=3,
            ),
        ),
    ]
    failures = 0
    for name, scenario in scenarios:
        failures += len(report(name, scenario))
        for missed in scenario.untouched():
            print(f"  wrote {missed}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
