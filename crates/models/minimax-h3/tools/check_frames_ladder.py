#!/usr/bin/env python3
"""Check that the generator's default clip ladder is the ladder it states.

The audio rows are one set per clip length, so the ladder is a deployment statement: a rung the
file does not hold is a request refused at its first audio row. Two things can then go wrong
without any test noticing - **the default drifts from the stated ladder**, and **a stated ladder
silently narrows the default**. Both are one question:

    run with no `--frames`, run with the ladder the header states, compare the rows.

The ladder is read from the generator's own header and never restated here: a tool carrying its
own copy of `17n + 5` could agree with itself while disagreeing with the file.

    python crates/models/minimax-h3/tools/check_frames_ladder.py
    python crates/models/minimax-h3/tools/check_frames_ladder.py --self-test

Exits non-zero, naming the rows that differ, on a mismatch.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]

# The generator's header states the clip lengths it served: `clip lengths: [22, 39, ...]`.
LADDER = re.compile(r"clip lengths: \[(?P<list>[^\]]*)\]")


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
        "\n".join(line for line in lines if line.startswith("#")),
        [line for line in lines if "\t" in line],
    )


def ladder(header: str) -> list[int]:
    """The clip lengths the header says it served."""
    found = LADDER.search(header)
    if not found:
        raise SystemExit(
            "the generator's header states no clip lengths: this check reads the ladder"
            " from there rather than restating the model's frame rule, so the generator"
            " owes the line `# ... clip lengths: [...]` before the check can run"
        )
    return [int(value) for value in found["list"].split(",") if value.strip()]


def report(default: set[str], stated: set[str]) -> int:
    """The comparison itself, so the self-test can drive it without a build."""
    missing = sorted(default - stated)
    extra = sorted(stated - default)
    if not missing and not extra:
        print(
            "the default ladder is the stated ladder:"
            f" {len(default)} rows, no row missing, no row extra"
        )
        return 0
    for label, rows in (
        ("the default has and the stated ladder does not", missing),
        ("the stated ladder has and the default does not", extra),
    ):
        if rows:
            print(f"{len(rows)} row(s) {label}:")
            for row in rows:
                print(f"  {row}")
    print(
        "a default that is not the stated ladder is one deployment statement in two"
        " places: the file then serves clips its own header does not name"
    )
    return 1


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

    header, default = generate()
    if not default:
        raise SystemExit(
            "the generator produced no rows: an empty list is a build failure, not a"
            " ladder"
        )
    clip_frames = ladder(header)
    if not clip_frames:
        raise SystemExit(
            "the header states an empty ladder: a deployment that serves no clip length"
            " cannot answer an audio request, and that is a statement to make on purpose"
        )

    stated_header, stated = generate("--frames", ",".join(map(str, clip_frames)))
    if ladder(stated_header) != clip_frames:
        raise SystemExit(
            f"--frames {clip_frames} was asked for and the header states"
            f" {ladder(stated_header)}: the flag does not reach the rows"
        )
    print(f"{len(clip_frames)} rungs: {clip_frames}")
    return report(set(default), set(stated))


def self_test() -> int:
    """Prove the comparison: a matching ladder passes, either direction of drift fails."""
    base = {"a\t1", "b\t2"}
    assert report(base, set(base)) == 0
    assert report(base, {"a\t1"}) == 1
    assert report(base, {"a\t1", "b\t2", "c\t3"}) == 1
    stated = "# deployment: ... clip lengths: [22, 39, 56]."
    assert ladder(stated) == [22, 39, 56], ladder(stated)
    print("ok   the ladder comparison, both drift directions, and the header's parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
