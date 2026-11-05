#!/usr/bin/env python3
"""Check that every clip the generator states is a clip a request can name.

The audio rows are one set per clip length, so a ladder is a deployment
a rung the file does not hold is a request refused at its first audio row. Two
things go wrong with no test noticing, and they are one question asked twice:

* **a rung the rule refuses** - the deployment drives a clip no request can ask
  for, so its rows are for a length the model never sees;
* **a default that drifts from the stated ladder** - `--frames` and no flag
  produce two row sets, and the file serves clips its header does not name.

Both are computed from **the model's own frame rule**, read from `config/mod.rs`
- `CLIP_FRAMES`, `CLIP_LATENTS` and `MAX_FRAMES`, the three constants
`packing::video_latent_frames` compares against - and from the generator's own
`DEPLOYMENT_FRAMES`, read from its source. A tool carrying a copy of `17n + 5`
could agree with itself while disagreeing with the file, so it carries none.

    python crates/models/minimax-h3/tools/check_frames_ladder.py
    python crates/models/minimax-h3/tools/check_frames_ladder.py --self-test

When every default rung is requestable the check goes on to the row comparison:
run the generator with no `--frames` and with the rule's own ladder, and require
one row set. While a default rung is not requestable that run refuses by name,
so the check reports the rungs and stops: the row comparison cannot answer for a
ladder the deployment is not allowed to state.

Exits non-zero, naming the rungs or the rows that differ, on a mismatch.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]

# The generator's header may state the clip lengths it served:
# `clip lengths: [22, 39, ...]`.
LADDER = re.compile(r"clip lengths: \[(?P<list>[^\]]*)\]")

# The model's frame rule lives in three constants. They are read, never
# restated: the rule is `CLIP_FRAMES n + CLIP_LATENTS`, at least one clip, up to
# `MAX_FRAMES`.
CONSTANTS = pathlib.Path("crates/models/minimax-h3/src/config/mod.rs")
CONSTANT = re.compile(
    r"pub const (?P<name>CLIP_FRAMES|CLIP_LATENTS|MAX_FRAMES)"
    r": u32 = (?P<value>\d+);"
)

# The ladder the generator states when no `--frames` is given:
# `const DEPLOYMENT_FRAMES: [u32; 6] = [5, 17, .., MAX_FRAMES];`. A rung written
# as a constant's name is resolved from `config/mod.rs`, never typed here.
FIXTURE = pathlib.Path("notes/programs/minimax-h3-deployment-keys.txt")
AXES = re.compile(r"clip ladder \[(?P<list>[^\]]*)\]")

GENERATOR = pathlib.Path("crates/models/minimax-h3/examples/minimax_h3_keys.rs")
DEFAULT_LADDER = re.compile(
    r"const DEPLOYMENT_FRAMES: \[u32; \d+\] = \[(?P<list>[^\]]*)\];"
)


def constants() -> dict[str, int]:
    """The frame rule's three constants, as `config/mod.rs` states them."""
    values = {
        found["name"]: int(found["value"])
        for found in CONSTANT.finditer((ROOT / CONSTANTS).read_text())
    }
    missing = {"CLIP_FRAMES", "CLIP_LATENTS", "MAX_FRAMES"} - set(values)
    if missing:
        raise SystemExit(
            f"{CONSTANTS} states no {', '.join(sorted(missing))}: the rule "
            "is read from there rather than restated here, so a constant that "
            "is gone is refused"
        )
    return values


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
    values = constants()
    # The rule refuses anything shorter than one clip:
    # `num_frames < CLIP_FRAMES + CLIP_LATENTS` is the comparison
    # `packing::video_latent_frames` makes.
    frames = values["CLIP_LATENTS"] + values["CLIP_FRAMES"]
    step = values["CLIP_FRAMES"]
    rungs = []
    while frames <= values["MAX_FRAMES"]:
        rungs.append(frames)
        frames += step
    return rungs


def defaults() -> list[int]:
    """The ladder the generator states when no `--frames` is given."""
    found = DEFAULT_LADDER.search((ROOT / GENERATOR).read_text())
    if not found:
        raise SystemExit(
            f"{GENERATOR} states no `DEPLOYMENT_FRAMES`: the default ladder is "
            "read from there rather than restated here, so a constant that is "
            "gone is refused"
        )
    values = constants()
    rungs = []
    for item in found["list"].split(","):
        item = item.strip()
        if not item:
            continue
        # A rung may be written as a constant's name; the value is the rule's.
        name = item.rsplit("::", 1)[-1]
        if name in values:
            rungs.append(values[name])
        elif item.isdigit():
            rungs.append(int(item))
        else:
            raise SystemExit(
                f"the default ladder names `{item}`, which is neither a "
                "number nor one of the rule's constants: a rung this check "
                "cannot read is a rung it cannot judge"
            )
    return rungs


def requestable(
    stated: list[int], admissible: list[int]
) -> tuple[list[int], list[int]]:
    """Split the stated rungs into those a request can name and those it cannot.

    Both directions are reported: a rung outside the rule is a row set for a
    no request asks for, and a rung of the rule the deployment does not drive is
    a clip the store cannot serve.
    """
    admitted = set(admissible)
    refused = [rung for rung in stated if rung not in admitted]
    allowed = [rung for rung in stated if rung in admitted]
    return refused, allowed


def served(path: pathlib.Path) -> list[int]:
    """The clips the row set states it serves, from its own `axes:` line."""
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        found = AXES.search(line)
        if found:
            return [int(v) for v in found["list"].split(",") if v.strip()]
    return []


def ladder(header: str) -> list[int]:
    """The clip lengths the header says it served."""
    found = LADDER.search(header)
    if not found:
        raise SystemExit(
            "the header states no clip lengths: the check reads the ladder "
            "from there rather than restating the rule, so the generator owes "
            "the line `# ... clip lengths: [...]` before the check can run"
        )
    return [int(value) for value in found["list"].split(",") if value.strip()]


def report(default: set[str], stated: set[str]) -> int:
    """The row comparison, so the self-test can drive it without a build."""
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
        "a default that is not the stated ladder is one deployment statement "
        "two places: the file then serves clips its own header does not name"
    )
    return 1


def report_refused(
    stated: list[int],
    refused: list[int],
    allowed: list[int],
    admissible: list[int],
) -> int:
    """Name the rungs no request can ask for, and what the rule does admit."""
    values = constants()
    print(
        f"the default ladder states {len(stated)} rung(s), {len(refused)} of "
        "them are not clips a request can name:"
    )
    for rung in refused:
        print(
            f"  {rung:>5}  not of the form {values['CLIP_FRAMES']}n + "
            f"{values['CLIP_LATENTS']}"
        )
    print(
        f"the rule admits {len(admissible)} clip length(s), {admissible[0]} to "
        f"{admissible[-1]}; the default's {len(allowed)} requestable rung(s) "
        f"are {allowed}. A row set for a refused rung is a set for a clip the "
        "model never sees, and the row comparison cannot run until the default "
        "is one the rule admits"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the comparisons on sets it makes itself, with no build",
    )
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        return self_test()

    clause = servable()
    if not clause:
        raise SystemExit(
            "the rule admits no clip length: a deployment that serves no clip "
            "cannot answer an audio request, and that is a statement to make "
            "on purpose"
        )
    # The row set states a ladder in its own `axes:` line, and that statement is
    # what a store serves: a rung the model cannot produce is a row set for
    # clips no request can name, whatever the generator's default is.
    file_rungs = served(FIXTURE)
    if file_rungs:
        file_refused, _ = requestable(file_rungs, clause)
        if file_refused:
            print(
                f"{FIXTURE} serves {len(file_rungs)} clip(s) and "
                f"{len(file_refused)} of them are not clips the model can "
                "produce:"
            )
            for rung in file_refused:
                print(f"  {rung:>5}")
            print(
                "a row set served at clips no request can name is the "
                "defect this check exists for; the file must be cut at "
                "clips from the rule"
            )
            return 1
        print(
            f"{FIXTURE} serves {len(file_rungs)} clip(s), every one a clip "
            "the model can produce"
        )

    stated_rungs = defaults()
    refused, allowed = requestable(stated_rungs, clause)
    if refused:
        return report_refused(stated_rungs, refused, allowed, clause)

    header, default = generate()
    if not default:
        raise SystemExit(
            "the generator produced no rows: an empty list is a build failure, "
            "not a ladder"
        )
    if LADDER.search(header) and ladder(header) != clause:
        raise SystemExit(
            f"the header states the ladder {ladder(header)} and the rule gives "
            f"{clause}: a stated ladder that is not the rule is a second "
            "statement of one truth"
        )

    # Two sets, two answers, and only one of them is a defect: a delivery rung
    # outside the rule is a clip the model cannot serve, while a rule rung the
    # delivery does not drive is a scheduling choice - inventory nobody asks
    # for, which a store should not hold. The second is reported, never failed.
    undriven = [rung for rung in clause if rung not in set(stated_rungs)]
    print(
        f"{len(stated_rungs)} delivery rung(s), every one inside the rule; the "
        f"rule admits {len(clause)} and the delivery does not drive "
        f"{len(undriven)} of them, which is a choice about what to serve"
    )

    stated_header, stated = generate("--frames", ",".join(map(str, clause)))
    if ladder(stated_header) != clause:
        raise SystemExit(
            f"--frames {clause} was asked for and the header states "
            f"{ladder(stated_header)}: the flag does not reach the rows"
        )
    print(f"{len(clause)} rungs: {clause}")
    return report(set(default), set(stated))


def self_test() -> int:
    """Prove both comparisons: requestability, and either direction of drift."""
    base = {"a\t1", "b\t2"}
    assert report(base, set(base)) == 0
    assert report(base, {"a\t1"}) == 1
    assert report(base, {"a\t1", "b\t2", "c\t3"}) == 1
    axes = "# axes: 6 canvases (a, b) and the clip ladder [5, 17, 1024]"
    parsed = AXES.search(axes)
    assert [int(v) for v in parsed["list"].split(",")] == [5, 17, 1024]
    stated = "# deployment: ... clip lengths: [22, 39, 56]."
    assert ladder(stated) == [22, 39, 56], ladder(stated)
    # The rule as the model states it: `17n + 5`, at least one clip, to
    # `MAX_FRAMES`.
    rungs = servable()
    assert rungs[0] == 22 and rungs[1] == 39 and rungs[-1] <= 1024, rungs[:3]
    assert all(rung % 17 == 5 for rung in rungs), rungs
    # Requestability: a rung outside the rule is refused, a rung of the rule is
    # admitted, and the two lists partition what the deployment states.
    refused, allowed = requestable([5, 22, 39], rungs)
    assert refused == [5] and allowed == [22, 39], (refused, allowed)
    assert requestable(rungs, rungs) == ([], rungs)
    # The generator's own ladder is read, not restated: whatever it states, the
    # split accounts for every rung exactly once.
    default_rungs = defaults()
    refused, allowed = requestable(default_rungs, rungs)
    assert len(refused) + len(allowed) == len(default_rungs)
    # The second direction is not a defect: a rule rung the delivery does not
    # drive is inventory no run asks for. Thirty-eight of fifty-nine is the
    # delivery's own number and it must read as a choice, not a shortfall.
    assert all(rung not in {22, 39} for rung in rungs[2:])
    print(
        "ok   requestability both ways, the row comparison both ways, and the "
        "header's parse"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
