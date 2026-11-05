"""Compare one checkpoint's evaluated schedule with the reference's own evaluation of it.

The diagnosis design 10 asks for, on the schedule component: for every sigma point and every network
time, both implementations' value, the absolute difference, and - when anything differs - the index
where it is worst. A number with an index, never a summary word.

Both sides are produced by the code under comparison, not by this file:

* ours is `crates/models/minimax-h3/examples/schedule_dump.rs`, which parses the checkpoint's own
  statement and evaluates `Schedule::sigmas` / `Schedule::timesteps`;
* the reference's is `uniserve/diffusion/schedule.py`'s `uniform_grid`, `ladder` or `block_grid` -
  the functions `uniserve/models/minimax_h3/denoiser.py:112-141` calls - rebuilt from the statement
  the dump records, never from the dump's numbers.

This is a **diagnosis and never an acceptance gate** (design 10): a differing row names the step the
two implementations disagree about and blocks nothing.

    .venv/bin/python crates/models/minimax-h3/tools/compare_schedule.py \
        --dump artifacts/diagnosis/minimax-h3/schedule
    .venv/bin/python crates/models/minimax-h3/tools/compare_schedule.py --self-test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uniserve.diffusion.schedule import (  # noqa: E402
    block_grid,
    ladder,
    uniform_grid,
)


def reference_sigmas(statement: dict, shift: float) -> tuple[list[float], list[float]]:
    """The reference's sigmas and network times for one modality of one statement."""
    kind = statement["kind"]
    if kind == "uniform":
        schedule = uniform_grid(statement["points"], shift=shift, device="cpu")
    elif kind == "dmd":
        schedule = ladder(
            statement["rungs"],
            shift=shift,
            num_train_timesteps=statement["num_train_timesteps"],
            device="cpu",
        )
    elif kind == "blocks":
        schedule = block_grid(
            statement["intervals"],
            statement["nodes"],
            shift=shift,
            max_t=statement["max_t"],
            device="cpu",
        ).schedule
    else:
        raise SystemExit(f"unknown grid kind {kind!r} in the dump's statement")
    return (
        [float(value) for value in schedule.sigmas],
        [float(value) for value in schedule.timesteps],
    )


def compare_one(document: dict) -> list[str]:
    """One dump's rows: shapes, then per-step values, then the worst row."""
    lines = []
    dit = document["dit"]
    statement = document["statement"]
    lines.append(
        f"## {dit}: {statement['kind']} grid, {document['steps']} steps, "
        f"checkpoint {document['checkpoint']}, commit {document['commit']}"
    )
    worst = (0.0, "none", -1, 0.0, 0.0)
    for modality in ("video", "audio"):
        ours = document[modality]
        wanted, wanted_times = reference_sigmas(statement, ours["shift"])
        for field, mine, theirs in (
            ("sigma", ours["sigmas"], wanted),
            ("time", ours["timesteps"], wanted_times),
        ):
            if len(mine) != len(theirs):
                lines.append(
                    f"{modality:5s} {field:5s} SHAPE ours {len(mine)} points, "
                    f"reference {len(theirs)} points"
                )
                continue
            for index, (left, right) in enumerate(zip(mine, theirs, strict=True)):
                # Both sides are **FP32** values: a schedule is evaluated in FP32 on either side
                # (`schedule.py`'s ``dtype=torch.float32``), and this dump states its numbers in the
                # shortest form that round-trips an f32. Comparing the f64 promotions of those two
                # spellings reports an arithmetic difference that no device could see, so the
                # comparison is made in the type both implementations compute in.
                ours32 = float(numpy.float32(left))
                theirs32 = float(numpy.float32(right))
                difference = abs(ours32 - theirs32)
                if difference > worst[0]:
                    worst = (difference, modality, index, ours32, theirs32)
                if difference != 0.0:
                    lines.append(
                        f"{modality:5s} {field:5s} {index:3d} ours {ours32!r} "
                        f"reference {theirs32!r} abs {difference:.3e}"
                    )
    return lines, worst


def report(paths: list[pathlib.Path]) -> int:
    """Print each dump's table; return the count of differing rows."""
    differing = 0
    for path in paths:
        document = json.loads(path.read_text())
        lines, worst = compare_one(document)
        print("\n".join(lines))
        difference, modality, index, left, right = worst
        if difference == 0.0:
            print(
                f"worst  none - every sigma and every time agrees exactly, "
                f"{len(document['video']['sigmas'])} + {len(document['audio']['sigmas'])} points"
            )
        else:
            print(
                f"worst  {modality} index {index}: ours {left!r} reference {right!r} "
                f"abs {difference:.3e}"
            )
            differing += 1
        print()
    return differing


def self_test() -> int:
    """Three checks a comparator of this shape has to pass before its table means anything."""
    failures = []
    statement = {"kind": "uniform", "points": 50}
    base = {
        "checkpoint": "self-test",
        "commit": "self-test",
        "dit": "transformer",
        "grid": "uniform",
        "statement": statement,
        "steps": 49,
    }
    for modality, shift in (("video", 12.0), ("audio", 3.0)):
        sigmas, times = reference_sigmas(statement, shift)
        base[modality] = {"shift": shift, "sigmas": sigmas, "timesteps": times}

    lines, worst = compare_one(base)
    if worst[0] != 0.0:
        failures.append(f"self-test 1: the reference's own values differ from themselves at {worst}")
    if any("abs" in line for line in lines):
        failures.append("self-test 1: a row printed a difference for identical inputs")

    perturbed = json.loads(json.dumps(base))
    perturbed["video"]["sigmas"][7] += 1e-6
    lines, worst = compare_one(perturbed)
    # The perturbation survives FP32 rounding as 1 + 1.3e-6 - the check is its index and its order of
    # magnitude, not its exact value, because the comparison is made in the type both sides compute in.
    if worst[2] != 7 or not 5e-7 < worst[0] < 2e-6:
        failures.append(f"self-test 2: a perturbed point was reported as {worst}, not index 7")
    if not any("abs 1.0" in line or "abs 9.9" in line for line in lines):
        failures.append("self-test 2: the differing row was not printed with its difference")

    short = json.loads(json.dumps(base))
    short["audio"]["sigmas"] = short["audio"]["sigmas"][:-1]
    short["audio"]["timesteps"] = short["audio"]["timesteps"][:-1]
    lines, _ = compare_one(short)
    if not any("SHAPE" in line for line in lines):
        failures.append("self-test 3: a length mismatch was not reported as a shape row")

    for failure in failures:
        print(failure)
    print(f"self-test {len(failures)} failure(s)")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        type=pathlib.Path,
        help="a schedule dump directory, or one schedule-<dit>.json file",
    )
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    if arguments.dump is None:
        parser.error("--dump wants a directory the schedule harness wrote")
    paths = (
        sorted(arguments.dump.glob("schedule-*.json"))
        if arguments.dump.is_dir()
        else [arguments.dump]
    )
    if not paths:
        print(f"no schedule-*.json under {arguments.dump}")
        return 2
    differing = report(paths)
    print(f"verdict {'equal' if differing == 0 else f'{differing} component(s) differ'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
