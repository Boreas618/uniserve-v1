#!/usr/bin/env python3
"""Run the attention_gather_decode_inputs program on a GPU and check every output's bits.

    python gather_check.py
    python gather_check.py --declared

The package's Triton source is launched through Triton's own launcher, which is the
code the store's module is compiled from, so a run exercises the compiled program over
a different build path. Every output is an index, a copy of one, or an exclusive offset
sum of them: there is nothing to tolerate, and a gather off by one stride still produces
plausible numbers, so the comparison is exact over exactly the cells a call must write
(the scenario says which).

Three scenarios, each for what it can break: a padded batch whose slots hold short and
empty caches; non-contiguous storage with three position axes and caches past the
request width; and a windowed table whose slots installed a start page. `--declared`
adds the package's own decode shape — a 256-row bucket, 256 selected columns of a
257-wide row, 512 slots — which is the shape the store builds a module for, for both
position layouts.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

here = pathlib.Path(__file__).resolve().parent
# The scenario module imports the package's kernel source, so both are on the path.
sys.path.insert(0, str(here.parent / "src"))
sys.path.insert(0, str(here))


def scenarios(declared):
    """The scenarios one run checks, in the order they break things."""
    from scenario import Scenario

    found = [
        (
            "a padded batch whose slots hold short and empty caches",
            Scenario(rows=4, max_rows=4, columns=4, seed=1),
        ),
        (
            "non-contiguous storage, three axes, caches past the request width",
            Scenario(
                rows=5,
                max_rows=8,
                columns=6,
                tables=2,
                request_width=4,
                position_axes=3,
                cache=[9, 0, 17, 3, 2, 0, 1, 12],
                row_padding=2,
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
    if declared:
        for axes in (1, 3):
            found.append(
                (
                    f"the declared geometry, {axes} position axis(es)",
                    Scenario(
                        rows=256,
                        max_rows=256,
                        columns=256,
                        request_width=256,
                        position_axes=axes,
                        cache=np.tile(np.arange(1, 33), 8)[:256],
                        block=256,
                        seed=4 + axes,
                    ),
                )
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declared",
        action="store_true",
        help="also run the package's declared decode shape",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"
    elif not torch.cuda.is_available():
        print(
            "skipped: needs a device — this check runs the kernel itself and has no "
            "interpreter path",
            file=sys.stderr,
        )
        return 0

    device = "cpu" if arguments.interpreter else "cuda"
    failures = 0
    for label, scenario in scenarios(arguments.declared):
        expected = scenario.reference()
        got = scenario.run(device)
        wrong = []
        for output in sorted(expected):
            mask = scenario.written(output, expected[output].size)
            differing = (got[output] != expected[output]) & mask
            if differing.any():
                index = tuple(int(value) for value in np.argwhere(differing)[0])
                wrong.append(
                    f"{output} at {index}: got {int(got[output][index])}, "
                    f"expected {int(expected[output][index])}"
                )
        print(
            f"{'exact' if not wrong else 'DIFFERS'}: {label} — "
            f"{scenario.tables} table(s), {scenario.rows} rows of "
            f"{scenario.max_rows}, {scenario.columns} columns, "
            f"{scenario.position_axes} axis(es), window "
            f"{scenario.shapes[0][2] if scenario.tables else -1}"
        )
        for one in wrong:
            print(f"  {one}", file=sys.stderr)
        failures += len(wrong)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
