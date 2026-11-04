#!/usr/bin/env python3
"""Run the attention_gather_rows program on a GPU and check every output bit for bit.

    python gather_check.py
    python gather_check.py --declared

The package's Triton source is launched through Triton's own launcher, which is the
code the store's module is compiled from, so a run exercises the compiled program and
a different build path. Every output is an index or a copy of one: there is no
arithmetic to tolerate, and a gather that is off by one stride still produces
plausible numbers, so the comparison is exact over exactly the cells a call must
write (the scenario says which — a table's rows past the call's, a block table's cells
past the selected width and a mapping's tokens past the call's belong to the storage,
not to the contract).

Three scenarios, each for what it can break: a dense table whose token section ends
at a row's edge; a storage that is not contiguous, with slots that installed fewer
units than the request width; and a windowed table, where a row's block table starts
at `max(0, prefix - window) // block_size`. `--declared` adds the package's own
descriptor geometry — 256 units per slot, 512 slots, a 257-element row stride — which
is the shape the store builds a module for.
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
            "a dense table, tokens ending at a row's edge",
            Scenario(rows=3, tokens=4, tables=1, width=4, block=4, seed=1),
        ),
        (
            "non-contiguous storage, slots with fewer units",
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
    if declared:
        found.append(
            (
                "the declared descriptor's geometry",
                Scenario(
                    rows=64,
                    tokens=256,
                    tables=1,
                    width=256,
                    block=256,
                    request_width=256,
                    slots=512,
                    row_padding=0,
                    seed=4,
                ),
            ),
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declared",
        action="store_true",
        help="also run the package's declared descriptor geometry",
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
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

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
            f"{scenario.tables} table(s), {scenario.rows} rows, "
            f"{scenario.tokens} tokens, width {scenario.width}, "
            f"block {scenario.block}, window {scenario.window}"
        )
        for one in wrong:
            print(f"  {one}", file=sys.stderr)
        failures += len(wrong)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
