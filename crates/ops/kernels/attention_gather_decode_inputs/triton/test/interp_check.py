#!/usr/bin/env python3
"""Check the attention_gather_decode_inputs program's semantics on the CPU.

    TRITON_INTERPRET=1 python interp_check.py

Triton's interpreter runs the kernel's own Python, so this proves the arithmetic, the
masks and the addresses on any host with no device: which page a row's block table
starts at, which unit a token appends to, what an uninstalled unit reads as, and the
exclusive offsets. Every output is an index or a copy of one, so it is compared bit for
bit, over exactly the cells a call must write — a block table's cells past the selected
columns and the rows past the call's belong to the storage, not to the contract.

The prefix offsets are the exception, and it is Triton's interpreter rather than the
program: `tl.cumsum` over a row block returns uninitialized lanes there, differently on
each run with the same inputs, so `SCANNED` names it as not compared here and the device
check compares it instead.
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

import attention_gather_decode_inputs as kernels  # noqa: E402,F401
import numpy as np  # noqa: E402
from scenario import Scenario  # noqa: E402

#: The one output the CPU cannot settle: it is the scan `tl.cumsum` produces, and
#: Triton's interpreter returns uninitialized lanes for it — the same call with the same
#: inputs gives a different tail on different runs, so no comparison over it is
#: evidence either way. The device check compares it; here it is named rather than
#: quietly dropped.
SCANNED = {"prefix_offsets"}


def report(name, scenario):
    """Compare one scenario's outputs against the reference, bit for bit."""
    expected = scenario.reference()
    got = scenario.run()
    failures = []
    for output in sorted(expected):
        if output in SCANNED:
            continue
        mask = scenario.written(output, expected[output].size)
        differences = (got[output] != expected[output]) & mask
        if differences.any():
            index = tuple(int(value) for value in np.argwhere(differences)[0])
            failures.append(
                f"{output} at {index}: got {int(got[output][index])}, "
                f"expected {int(expected[output][index])}"
            )
    print(
        f"{'exact' if not failures else 'DIFFERS'}: {name} — "
        f"{scenario.tables} table(s), {scenario.rows} rows of "
        f"{scenario.max_rows}, {scenario.columns} columns, "
        f"{scenario.position_axes} axis(es), window "
        f"{scenario.shapes[0][2] if scenario.tables else -1}"
    )
    print(
        "  not compared here: the prefix offsets, the one scan; the device check"
        " compares them"
    )
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    return len(failures)


def main():
    scenarios = [
        (
            "a padded batch whose slots hold short and empty caches",
            Scenario(rows=4, max_rows=4, columns=4, seed=1),
        ),
        (
            "non-contiguous storage, three position axes, caches past the request width",
            Scenario(
                rows=5,
                max_rows=8,
                columns=6,
                tables=2,
                request_width=4,
                position_axes=3,
                cache=[9, 0, 17, 3, 2, 0, 1, 12],
                row_padding=2,
                table_padding=5,
                page_padding=3,
                address_padding=2,
                block=16,
                seed=2,
            ),
        ),
        (
            "a windowed table whose slots installed a start page",
            Scenario(
                rows=3,
                max_rows=3,
                columns=3,
                request_width=5,
                window=6,
                installed=[2, 1, 0, 1, 0, 0, 0, 0, 1],
                seed=3,
            ),
        ),
    ]
    failures = sum(report(name, scenario) for name, scenario in scenarios)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
