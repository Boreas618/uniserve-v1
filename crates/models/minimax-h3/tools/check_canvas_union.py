#!/usr/bin/env python3
"""Check that the deployment list is exactly the union of its per-canvas runs.

The list's canvas-class rows - `frame_pad`'s tile, the encoder levels' `scaled_add`
widths - are a function of the raster the video encoder serves, so one run of the
generator covers every canvas the deployment offers. That makes one question worth
answering on its own: **is the multi-canvas list the union of the single-canvas ones?**
A `rasters` vector a later edit shadows, a default that replaces the flag, or a call
that reads one list while the header prints another, all pass a "the file regenerates"
check and all fail this one.

    python crates/models/minimax-h3/tools/check_canvas_union.py

The canvases come from the generator's own header line, never from a copy kept here: the
header is what the generator says it served, and a tool carrying its own copy could agree
with itself while disagreeing with the file. One build per canvas, no device.

Exits non-zero, naming the rows that differ, on a mismatch.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]

# The generator's header states the rasters it served as Rust tuples of height and width.
RASTERS = re.compile(r"rasters \(height, width\) \[(?P<list>[^\]]*)\]")
PAIR = re.compile(r"\((\d+),\s*(\d+)\)")


def generate(*arguments: str) -> tuple[str, list[str]]:
    """One run of the generator: its header lines, and its data lines only."""
    command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "minimax-h3",
        "--example",
        "minimax_h3_keys",
        "--",
        *arguments,
    ]
    result = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    return (
        # The header is the generator's own statement of what it served: the `#` lines.
        "\n".join(line for line in lines if line.startswith("#")),
        [line for line in lines if "\t" in line],
    )


def rasters(header: str) -> list[tuple[int, int]]:
    """The canvases the header says it served, as height and width pairs."""
    found = RASTERS.search(header)
    if not found:
        raise SystemExit(
            "the generator's header states no rasters: this check reads the canvases"
            " from there rather than restating them, so a header without them is"
            " refused"
        )
    return [(int(height), int(width)) for height, width in PAIR.findall(found["list"])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the comparison on row sets it makes itself, with no build",
    )
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        return self_test()

    header, rows = generate()
    if not rows:
        raise SystemExit(
            "the generator produced no rows: an empty list is a build failure,"
            " not a union"
        )
    canvases = rasters(header)
    if len(canvases) < 2:
        raise SystemExit(
            f"the generator served {len(canvases)} raster(s): a union of one has"
            " nothing to check, and the list is meant to cover every canvas the"
            " deployment serves"
        )

    whole = set(rows)
    singles: set[str] = set()
    for height, width in canvases:
        raster, produced = generate("--rasters", f"{height}x{width}")
        if not produced:
            raise SystemExit(f"{height}x{width}: the generator produced no rows")
        if rasters(raster) != [(height, width)]:
            raise SystemExit(
                f"--rasters {height}x{width} was asked for and the header"
                f" states {rasters(raster)}: the flag does not reach the rows"
            )
        singles.update(produced)

    print(f"{len(canvases)} canvases, {len(whole)} rows in the union run")
    return report(whole, singles)


def report(whole: set[str], singles: set[str]) -> int:
    """The comparison itself, so the self-test can drive it without a build."""
    missing = sorted(whole - singles)
    extra = sorted(singles - whole)
    if not missing and not extra:
        print(
            "the list is exactly the union of its per-canvas runs:"
            f" {len(singles)} rows, no row missing, no row extra"
        )
        return 0
    for label, rows in (
        ("the union run has and no single run does", missing),
        ("a single run has and the union run does not", extra),
    ):
        if rows:
            print(f"{len(rows)} row(s) {label}:")
            for row in rows:
                print(f"  {row}")
    print(
        "a single-canvas run that disagrees with the multi-canvas one is one list"
        " shadowing another: the flag, the default, or the call the rows come from"
    )
    return 1


def self_test() -> int:
    """Prove the comparison: a matching union passes, a mismatch in either way fails."""
    base = {"a\t1", "b\t2", "c\t3"}
    assert report(base, set(base)) == 0
    assert report(base, {"a\t1", "b\t2"}) == 1
    assert report(base, {"a\t1", "b\t2", "c\t3", "d\t4"}) == 1
    # The header's parse: two canvases, in the order the generator prints them.
    stated = "# derived: rasters (height, width) [(672, 1536), (768, 1344)]."
    assert rasters(stated) == [(672, 1536), (768, 1344)], rasters(stated)
    print("ok   the union comparison, both mismatch directions, the header's parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
