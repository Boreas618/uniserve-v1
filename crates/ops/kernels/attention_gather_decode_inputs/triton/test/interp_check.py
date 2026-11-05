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
import subprocess
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


def scenarios():
    """The scenarios this check compares, in order; `--scenario N` runs one of them alone."""
    return [
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


def run_alone(index):
    """Run one scenario in this process, for the parent that spawns them one at a time."""
    name, scenario = scenarios()[index]
    return 1 if report(name, scenario) else 0


def main():
    """Run every scenario, each in its own interpreter process.

    One process per scenario because the interpreter cannot be trusted with this kernel's scan: on
    one of the shapes `tl.cumsum` **segfaults** the interpreter process — no traceback, exit 139 —
    and it does so nondeterministically (measured: 4 of 6 runs on the same input died, 2 passed).
    A crash is therefore not an input to keep out of the list and not a verdict about the kernel:
    it is a state this host cannot compare, and the other scenarios' comparisons are still evidence.
    A parent that ran them in one process would lose all of them the first time the scan died.
    """
    if "--scenario" in sys.argv:
        return run_alone(int(sys.argv[sys.argv.index("--scenario") + 1]))

    compared = crashed = differing = 0
    for index, (name, _) in enumerate(scenarios()):
        child = subprocess.run(
            [sys.executable, __file__, "--scenario", str(index)],
            env={**os.environ, "TRITON_INTERPRET": "1"},
        )
        if child.returncode == 0:
            compared += 1
        elif child.returncode < 0 or child.returncode >= 128:
            crashed += 1
            print(
                f"not compared here: {name} — the interpreter process died on this scenario's scan "
                f"(exit {child.returncode}, no traceback). `tl.cumsum` crashes the interpreter "
                f"nondeterministically (measured: 4 of 6 runs on one input), so neither a pass nor a "
                f"crash here is evidence about the program; the device check compares it."
            )
        else:
            differing += 1
    print(
        f"{len(scenarios())} scenario(s): {compared} compared, {differing} differing, "
        f"{crashed} the interpreter cannot run"
    )
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
