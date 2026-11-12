#!/usr/bin/env python3
"""Run the rows program in each of its modes and check every destination row.

    python rows_check.py                    # on a GPU
    python rows_check.py --interpreter      # the same on Triton's interpreter

The operator places or accumulates the rows a call picks out by index:

    replace: destination[indices[j]] = source[j]
    add:     destination[indices[j]] += source[j]        one rounding into the destination's type
    place:   destination[indices[j]] = source[j]         for indices[j] >= 0, nothing otherwise

The cases are what a row scatter can get wrong: a row narrower than the program's block of
columns and one of several blocks, rows written out of order, rows no index names (which keep
their values), and, for `place`, negative indices among the rows, whose source rows must reach
no destination at all. The indices of a case are distinct: the operator orders nothing between
two source rows that name one destination.

`replace` and `place` copy and are exact. `add` sums in FP32 and rounds once on the store, so it
is held to two units of the destination type's rounding, as `Rows::tolerance` states; a narrow
store under Triton's interpreter truncates where a device rounds, which the bound's second unit
covers.
"""

import argparse
import os
import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
sys.path.insert(0, str(KERNELS / "rows" / "triton" / "src"))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpreter", action="store_true", help="run on Triton's interpreter"
    )
    return parser.parse_args()


FLAGS = parse()
# The family suite's interpreter run states it in the environment, with no device visible.
FLAGS.interpreter = (
    FLAGS.interpreter or os.environ.get("TRITON_INTERPRET") == "1"
)
if FLAGS.interpreter:
    os.environ["TRITON_INTERPRET"] = "1"

import rows as kernels  # noqa: E402
import torch  # noqa: E402

UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float32: 2.0**-24}


def check(mode, dtype, rows, columns, block, indices, device, seed):
    torch.manual_seed(seed)
    count = len(indices)
    integer = dtype == torch.int32
    draw = (
        (lambda *shape: torch.randint(-999, 999, shape))
        if integer
        else torch.randn
    )
    source = draw(count, columns).to(dtype)
    before = draw(rows, columns).to(dtype)
    destination = before.clone().to(device)

    kernels._rows_kernel[(count, -(-columns // block))](
        source.to(device),
        destination,
        torch.tensor(indices, dtype=torch.int32, device=device),
        count,
        columns,
        block,
        0,
        COLUMNS=columns,
        REPLACE=mode != "add",
        BLOCK=block,
        PDL=False,
        SKIP=mode == "place",
        num_warps=4,
    )

    expected = before.double()
    for row, index in enumerate(indices):
        if index < 0:
            continue
        if mode == "add":
            expected[index] += source[row].double()
        else:
            expected[index] = source[row].double()

    error = (destination.cpu().double() - expected).abs()
    bound = (
        2 * UNIT_ROUNDOFF[dtype] * expected.abs()
        if mode == "add"
        else torch.zeros_like(expected)
    )
    outside = int((~(error <= bound)).sum())
    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {mode}, {count} rows into {rows} x {columns} of "
        f"block {block}, {dtype} — max abs {float(error.max()):.3e}, {outside} outside"
    )
    return outside


def main():
    device = "cpu" if FLAGS.interpreter else "cuda"
    outside = 0
    outside += check(
        "replace", torch.bfloat16, 9, 100, 128, [4, 0, 7], device, 1
    )
    outside += check("replace", torch.float32, 6, 300, 128, [5, 2], device, 2)
    outside += check(
        "add", torch.bfloat16, 9, 100, 128, [1, 8, 3, 0], device, 3
    )
    # A negative index names no row: its source row reaches no destination.
    outside += check(
        "place", torch.bfloat16, 9, 128, 128, [4, -1, 7, -1, 0], device, 4
    )
    outside += check("place", torch.bfloat16, 5, 300, 128, [-1, -1], device, 5)
    outside += check("place", torch.int32, 8, 3, 4, [6, -1, 1], device, 6)
    print(
        f"{'ok' if outside == 0 else 'FAIL'}: {outside} values outside the bound"
    )
    return 0 if outside == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
