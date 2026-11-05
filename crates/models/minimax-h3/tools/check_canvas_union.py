#!/usr/bin/env python3
"""Check that the deployment list is exactly the union of its per-canvas runs.

The list's canvas-class rows - `frame_pad`'s tile, the encoder levels'
`scaled_add` widths - are a function of the raster the video encoder serves, so
one run of the generator covers every canvas the deployment offers. That makes
one question worth answering on its own: **is the multi-canvas list the union
of the single-canvas ones?** A `rasters` vector a later edit shadows, a default
that replaces the flag, or a call that reads one list while the header prints
another, all pass a "the file regenerates" check and all fail this one.

    python crates/models/minimax-h3/tools/check_canvas_union.py

The canvases come from the generator's own header line, never from a copy kept
here: the header is what the generator says it served, and a tool carrying its
own copy could agree with itself while disagreeing with the file. One build per
canvas, no device.

Exit status: **0** the list is the union of its per-canvas runs, **1** they
disagree (naming the rows), **2** there is no subject - the rows do not
follow the raster, so every comparison here would pass on any list and the tool
says so instead of passing.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]

# Exit statuses: 0 the union is the parts, 1 they disagree, and 2 no subject
# - the rows do not follow the raster, so there is nothing to check.
NO_SUBJECT = 2

# The generator's header states the rasters it served as height and width.
RASTERS = re.compile(r"rasters \(height, width\) \[(?P<list>[^\]]*)\]")
PAIR = re.compile(r"\((\d+),\s*(\d+)\)")

# The file may state that the rows which follow the canvas are keys a request
# derives rather than rows it holds - `frame_pad`'s tile and the encoder levels'
# `scaled_add` widths. When it says so, no row can depend on the raster and the
# union comparison has no subject; when it does not, a row set that is the same
# for every canvas is a canvas-class row gone missing.
REQUEST_DERIVED = re.compile(
    r"request-derived:.*follow(?:s)? the canvas", re.DOTALL
)


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
        # The header is the generator's own statement of what it served.
        "\n".join(line for line in lines if line.startswith("#")),
        [line for line in lines if "\t" in line],
    )


def rasters(header: str) -> list[tuple[int, int]]:
    """The canvases the header says it served, as height and width pairs."""
    found = RASTERS.search(header)
    if not found:
        raise SystemExit(
            "the generator's header states no rasters: this check reads the "
            "canvases from there rather than restating them, so a header "
            "without them is refused"
        )
    return [
        (int(height), int(width))
        for height, width in PAIR.findall(found["list"])
    ]


def derived_keys(header: str) -> bool:
    """Whether the file says its canvas keys are request-derived.

    True means no row can depend on the raster, so a per-canvas union has no
    subject and a pass would prove nothing; False means the same silence is a
    canvas-class row gone missing.
    """
    return REQUEST_DERIVED.search(header) is not None


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
            "the generator produced no rows: an empty list is a build "
            "failure, not a union"
        )
    canvases = rasters(header)
    if len(canvases) < 2:
        raise SystemExit(
            f"the generator served {len(canvases)} raster(s): a union of one "
            "has nothing to check, and the list is meant to cover every canvas "
            "the deployment serves"
        )

    whole = set(rows)
    singles: set[str] = set()
    per_canvas: list[set[str]] = []
    for height, width in canvases:
        raster, produced = generate("--rasters", f"{height}x{width}")
        if not produced:
            raise SystemExit(
                f"{height}x{width}: the generator produced no rows"
            )
        if rasters(raster) != [(height, width)]:
            raise SystemExit(
                f"--rasters {height}x{width} was asked for and the header "
                f"states {rasters(raster)}: the flag does not reach the rows"
            )
        per_canvas.append(set(produced))
        singles.update(produced)

    # A union check whose parts are all identical proves nothing: it passes on
    # any list, including one whose canvas-class rows are wrong or missing. So
    # the raster must be shown to select something, or the pass is vacuous and
    # says so instead of passing.
    if not selects(per_canvas):
        # Two causes, and they need different answers: the file may state that
        # the keys which follow the canvas are request-derived, in which case no
        # row can depend on the raster and this comparison has no subject; or a
        # canvas-class row may be missing, in which case the parts are missing
        # rather than equal and that is a defect.
        if derived_keys(header):
            print(
                f"{len(canvases)} canvases produced one row set of "
                f"{len(whole)} rows, and the file states this: the keys that "
                "follow the canvas are request-derived, so no row depends on "
                "the raster and one store serves every canvas. This "
                "comparison has no subject, and it is not a pass"
            )
            # Its own status: a script must be able to tell "the union is right"
            # from "there is no union to check", and 0 would conflate them.
            return NO_SUBJECT
        raise SystemExit(
            f"{len(canvases)} canvases produced one row set of {len(whole)} "
            "rows: no row depends on --rasters and the file does not state "
            "that its canvas-class keys are request-derived, so the rows "
            "which should follow the raster are missing rather than equal"
        )

    print(f"{len(canvases)} canvases, {len(whole)} rows in the union run")
    return report(whole, singles)


def selects(per_canvas: list[set[str]]) -> bool:
    """Whether the raster changes any row: false makes the check vacuous."""
    return any(part != per_canvas[0] for part in per_canvas[1:])


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
        "a single-canvas run that disagrees with the multi-canvas one is one "
        "list shadowing another: the flag, the default, or the call the rows "
        "come from"
    )
    return 1


def self_test() -> int:
    """Prove the comparison: a matching union passes, either drift fails."""
    base = {"a\t1", "b\t2", "c\t3"}
    assert report(base, set(base)) == 0
    assert report(base, {"a\t1", "b\t2"}) == 1
    assert report(base, {"a\t1", "b\t2", "c\t3", "d\t4"}) == 1
    # The header's parse: two canvases, in the order the generator prints them.
    stated = "# derived: rasters (height, width) [(672, 1536), (768, 1344)]."
    assert rasters(stated) == [(672, 1536), (768, 1344)], rasters(stated)
    # A raster that selects nothing makes every comparison above pass by
    # construction, so the vacuity is a tested property rather than a comment.
    assert selects([{"a\t1"}, {"a\t1"}]) is False
    assert selects([{"a\t1"}, {"a\t1", "b\t2"}]) is True
    # The file's own statement decides whether a vacuous union is expected or a
    # missing canvas-class row: both readings are exercised here.
    stated_derived = "# request-derived: `frame_pad`'s tile follows the canvas."
    assert derived_keys(stated_derived)
    assert not derived_keys("# request-derived: nothing here.")
    assert NO_SUBJECT != 0 and NO_SUBJECT != 1, NO_SUBJECT
    print(
        "ok   the union comparison, both mismatch directions, the parse, and "
        "the raster's effect"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
