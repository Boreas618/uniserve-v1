#!/usr/bin/env python3
"""Check that the generator's default clip ladder is the ladder it states.

The audio rows are one set per clip length, so the ladder is a deployment statement: a rung the
file does not hold is a request refused at its first audio row. Two things can then go wrong
without any test noticing - **the default drifts from the stated ladder**, and **a stated ladder
silently narrows the default**. Both are one question:

    run with no `--frames`, run with the ladder the header states, compare the rows.

The ladder is computed from **the model's own frame rule**, read from `config/mod.rs` - `CLIP_FRAMES`,
`CLIP_LATENTS` and `MAX_FRAMES` - and never typed here: a tool carrying its own copy of `17n + 5` could
agree with itself while disagreeing with the file. If the generator's header states its ladder, that
statement is checked too, and the two must agree.

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

# The generator's header may state the clip lengths it served: `clip lengths: [22, 39, ...]`.
LADDER = re.compile(r"clip lengths: \[(?P<list>[^\]]*)\]")

# The model's frame rule lives in three constants. They are read, never restated: the rule is
# `CLIP_FRAMES n + CLIP_LATENTS`, at least one clip, up to `MAX_FRAMES`.
CONSTANTS = pathlib.Path("crates/models/minimax-h3/src/config/mod.rs")
CONSTANT = re.compile(
    r"pub const (?P<name>CLIP_FRAMES|CLIP_LATENTS|MAX_FRAMES): u32 = (?P<value>\d+);"
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
        "\n".join(line for line in lines if line.startswith("#")),
        [line for line in lines if "\t" in line],
    )


def servable() -> list[int]:
    """The clip lengths the model's own rule admits, read from its constants."""
    values = {
        found["name"]: int(found["value"])
        for found in CONSTANT.finditer((ROOT / CONSTANTS).read_text())
    }
    missing = {"CLIP_FRAMES", "CLIP_LATENTS", "MAX_FRAMES"} - set(values)
    if missing:
        raise SystemExit(
            f"{CONSTANTS} states no {', '.join(sorted(missing))}: the frame rule is read"
            " from there rather than restated here, so a constant that is gone is refused"
        )
    # The rule refuses anything shorter than one clip: `num_frames < CLIP_FRAMES + CLIP_LATENTS`
    # is the same comparison `packing::video_latent_frames` makes.
    frames, step = (
        values["CLIP_LATENTS"] + values["CLIP_FRAMES"],
        values["CLIP_FRAMES"],
    )
    rungs = []
    while frames <= values["MAX_FRAMES"]:
        rungs.append(frames)
        frames += step
    return rungs


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
    clip_frames = servable()
    if LADDER.search(header) and ladder(header) != clip_frames:
        raise SystemExit(
            f"the header states the ladder {ladder(header)} and the model's rule gives"
            f" {clip_frames}: a stated ladder that is not the rule is a second statement"
            " of one truth"
        )
    if not clip_frames:
        raise SystemExit(
            "the header states an empty ladder: a deployment that serves no clip length"
            " cannot answer an audio request, and that is a statement to make on purpose"
        )

    stated_header, stated = generate(
        "--frames", ",".join(map(str, clip_frames))
    )
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
    # The rule as the model states it: `17n + 5`, at least one clip, to `MAX_FRAMES`.
    rungs = servable()
    assert rungs[0] == 22 and rungs[1] == 39 and rungs[-1] <= 1024, rungs[:3]
    assert all(rung % 17 == 5 for rung in rungs), rungs
    print(
        "ok   the ladder comparison, both drift directions, and the header's parse"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
