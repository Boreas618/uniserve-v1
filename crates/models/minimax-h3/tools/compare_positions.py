#!/usr/bin/env python3
"""Compare one case's rotary coordinates with the reference's, axis by axis.

The model's rotation reads one table, `tables.position_ids`: a
`(time, height, width)` triple for every packed row. Our side computes it in
`packing::dense_tables`, the reference computes it in
`uniserve/models/minimax_h3/packing.py`. A swapped height and width - or a
pixel index where a latent index belongs - produces factors of the right shape,
type and count, so every other check this branch owns stays green while every
rotation turns the wrong way. This one names the axis, the row and the numbers.

The two sides write different arrangements, and the tool states which it read
rather than either side hiding the difference:

* ours is the operator's own layout, `[3, rows]` (``ops::rope``: "A sectioned
  rotation's positions are `[axes, tokens]`"), written by
  ``examples/positions.rs``;
* the reference's is `[rows, 3]` (`packing.py`: "position_ids:
  [padded_tokens, 3] FP64 rotary coordinates"), written by
  ``tools/dump_tables.py`` as ``dense.position_ids.npy``.

Both are integers in practice, so the comparison is an integer equality and not
a tolerance: the reference carries FP64 for the cosine's sake, and a coordinate
that is not a whole number would itself be a defect worth seeing.

    compare_positions.py --ours <dir>/positions.h3t \
        --ref <dir>/dense.position_ids.npy
    compare_positions.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

# The container's dtype codes, shared with every dump this crate writes.
CODES = {0: np.float64, 1: np.float32, 2: np.int64, 3: np.int32}

AXES = ("time", "height", "width")


def load(path: Path) -> np.ndarray:
    """Read one `H3D1` tensor: magic, dtype code, rank, extents, the payload."""
    data = path.read_bytes()
    if data[:4] != b"H3D1":
        raise ValueError(f"{path}: not an H3D1 container")
    code, rank = data[4], data[5]
    if code not in CODES:
        raise ValueError(
            f"{path}: dtype code {code} is not one of {sorted(CODES)}"
        )
    dtype = CODES[code]
    header = 6 + rank * 8
    shape = tuple(
        int.from_bytes(data[6 + index * 8 : 14 + index * 8], "little")
        for index in range(rank)
    )
    count = int(np.prod(shape))
    payload = data[header:]
    if len(payload) != count * np.dtype(dtype).itemsize:
        raise ValueError(
            f"{path}: a payload that does not match its shape {shape}"
        )
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def to_rows_axes(values: np.ndarray, layout: str) -> np.ndarray:
    """One tensor as `[rows, 3]`, from the arrangement its writer used."""
    return values.T if layout == "axis-major" else values


def compare(
    ours: np.ndarray, ref: np.ndarray, mapping: tuple[int, ...]
) -> dict:
    """Per-axis differences between two `[rows, 3]` tables.

    ``mapping`` says which of the reference's columns is our axis 0, 1 and 2:
    the reference's own order is (time, height, width), and stating it is what
    makes a reordering visible rather than silently correct.
    """
    if ours.shape != ref.shape:
        raise ValueError(
            f"{ours.shape} and {ref.shape} hold different coordinates"
        )
    rows = ours.shape[0]
    report = {"rows": rows, "axes": {}}
    for axis, name in enumerate(AXES):
        column = mapping[axis]
        difference = ours[:, axis].astype(np.int64) - ref[:, column].astype(
            np.int64
        )
        wrong = np.nonzero(difference != 0)[0]
        worst = int(np.argmax(np.abs(difference))) if rows else 0
        report["axes"][name] = {
            "column": column,
            "worst": int(np.abs(difference).max()) if rows else 0,
            "wrong": int(wrong.size),
            "first_wrong": int(wrong[0]) if wrong.size else None,
            "worst_row": worst,
            "ours": tuple(int(value) for value in ours[worst]) if rows else (),
            "ref": tuple(int(value) for value in ref[worst]) if rows else (),
        }
    # A swap is the failure this tool exists for, and an aggregate cannot see
    # it: if our axis `a` equals the reference's *other* column on every row,
    # the two axes hold the same integers in the wrong order.
    swaps = []
    for axis, name in enumerate(AXES):
        for other in range(3):
            if other == mapping[axis] or rows == 0:
                continue
            taken = np.array_equal(
                ours[:, axis].astype(np.int64), ref[:, other].astype(np.int64)
            )
            if taken:
                named = [
                    key for key, column in enumerate(mapping) if column == other
                ][0]
                swaps.append((name, AXES[named]))
    report["swaps"] = swaps
    report["equal"] = all(
        item["worst"] == 0 for item in report["axes"].values()
    )
    return report


def show(report: dict, ours_layout: str, ref_layout: str) -> None:
    """Print the report: a line per axis, the worst row, the verdict."""
    print(f"ours:      {ours_layout}, {report['rows']} rows")
    print(f"reference: {ref_layout}, {report['rows']} rows")
    for name, item in report["axes"].items():
        print(
            f"  {name:<7} max |difference| {item['worst']}, "
            f"{item['wrong']} of {report['rows']} rows differ"
            + (
                f"; first at row {item['first_wrong']}"
                if item["first_wrong"] is not None
                else ""
            )
        )
        if item["worst"]:
            print(
                f"          worst at row {item['worst_row']}: "
                f"ours {item['ours']}, reference {item['ref']}"
            )
    for ours_name, ref_name in report["swaps"]:
        print(
            f"  exchange: our {ours_name} axis equals the reference's "
            f"{ref_name} axis on every row"
        )
    print(f"verdict: {'equal' if report['equal'] else 'differ'}")


def self_test() -> int:
    """Prove the comparison on tables this script makes, with no reference."""
    rows = 12
    reference = np.stack(
        [np.arange(rows), np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3)],
        axis=1,
    ).astype(np.float64)
    failures = 0

    def check(name, ours, expect_equal, expect_swap=False):
        nonlocal failures
        report = compare(ours.astype(np.int64), reference, (0, 1, 2))
        if report["equal"] != expect_equal:
            print(f"FAIL {name}: verdict {report['equal']}")
            failures += 1
        elif expect_swap and not report["swaps"]:
            print(f"FAIL {name}: the swap was not named")
            failures += 1
        else:
            print(f"ok   {name}")
        return report

    check("an identical table is equal", reference.copy(), True)
    check(
        "a swapped height and width differ",
        reference[:, [0, 2, 1]].copy(),
        False,
        expect_swap=True,
    )
    off_by_one = reference.copy()
    off_by_one[5, 0] += 1
    report = check("one row off by one differs", off_by_one, False)
    if report["axes"]["time"]["first_wrong"] != 5:
        print("FAIL the off-by-one row was not named")
        failures += 1
    else:
        print("ok   the off-by-one row is named")
    check(
        "a misordered mapping differs",
        reference[:, [1, 0, 2]].copy(),
        False,
        expect_swap=True,
    )

    # The container round trip, so the reader is exercised and not only the
    # arithmetic.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "positions.h3t"
        payload = np.ascontiguousarray(reference.T.astype(np.int32)).tobytes()
        path.write_bytes(
            b"H3D1"
            + bytes([3, 2])
            + (3).to_bytes(8, "little")
            + (rows).to_bytes(8, "little")
            + payload
        )
        read = load(path)
        if read.shape != (3, rows) or not np.array_equal(
            read.T, reference.astype(np.int32)
        ):
            print(f"FAIL the container round trip gave {read.shape}")
            failures += 1
        else:
            print("ok   the container reads back as [3, rows]")
    return 1 if failures else 0


def main() -> int:
    """Compare the two files, or prove the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ours", type=Path, help="positions.h3t from examples/positions.rs"
    )
    parser.add_argument(
        "--ref", type=Path, help="dense.position_ids.npy from dump_tables.py"
    )
    parser.add_argument(
        "--ours-layout",
        choices=("axis-major", "row-major"),
        default="axis-major",
        help="the arrangement our file holds; the probe writes [3, rows]",
    )
    parser.add_argument(
        "--ref-layout",
        choices=("row-major", "axis-major"),
        default="row-major",
        help="the reference's arrangement; dump_tables writes [rows, 3]",
    )
    parser.add_argument(
        "--mapping",
        default="time,height,width",
        help="the reference's column order, by our axis names",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="prove the comparison only"
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    if arguments.ours is None or arguments.ref is None:
        parser.error(
            "--ours and --ref name the tables, or --self-test proves the tool"
        )

    orders = {name: index for index, name in enumerate(AXES)}
    try:
        mapping = tuple(
            orders[name.strip()] for name in arguments.mapping.split(",")
        )
    except KeyError as error:
        parser.error(f"{error} is not one of {AXES}")
    if sorted(mapping) != [0, 1, 2]:
        parser.error("--mapping must name each axis once")

    ours = to_rows_axes(load(arguments.ours), arguments.ours_layout)
    ref = to_rows_axes(np.load(arguments.ref), arguments.ref_layout)
    report = compare(ours, ref, mapping)
    show(report, arguments.ours_layout, arguments.ref_layout)
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
