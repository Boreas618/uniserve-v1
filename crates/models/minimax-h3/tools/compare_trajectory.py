#!/usr/bin/env python3
"""Pair two sampler trajectories and report, per evaluation, how far apart they are.

Design 10's diagnosis, never an acceptance gate: the reference's own trajectory
(``dump_denoiser.py --probe trajectory``) and this model's (``dit_layer --stage denoise``)
both write one H3D1 container per evaluation, and this script pairs them by index and
reports the relative L2 error, the largest absolute difference, the cosine similarity and
whether the two are bit-identical. It writes a report the caller keeps under ``artifacts/``.

What it does and does not decide:

* It reports numbers. It has no threshold, because a tolerance chosen by looking at the
  failure is the one kind of comparison evidence that cannot be trusted. The page is read
  against the reference's own error floor (the video decoder's is 1.007e-3 relative, the
  sampler's is set by the arithmetic - see ``notes/h3/stage3.md``).
* It refuses a pair it cannot interpret: an index on one side and not the other, a shape or
  element type that differs, a trajectory that did not move between two evaluations on
  either side, or two sides whose layout statements disagree. A draw is a valid tensor and a
  plausible one; a trajectory that equals its own draw is not a page. And the trajectory
  files hold latents, whose shapes do not depend on the packed row count, so two sides that
  ran different layouts write files of the same shape - a difference between them would be
  reported as arithmetic while its cause is the arrangement. Both producers state the layout
  they ran in ``denoiser/layout.json``, and the table prints the two statements above its
  rows.

Usage::

    python crates/models/minimax-h3/tools/compare_trajectory.py \
        --ref artifacts/h3-stage3-ref/denoiser --got artifacts/h3-stage3/denoiser \
        --out artifacts/h3-stage3-trajectory.json
    python crates/models/minimax-h3/tools/compare_trajectory.py --self-test

It needs only NumPy and the container reader in ``dump_tables.py`` beside it, so the harness
itself is provable on a host with no checkpoint, no device and no store.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dump_tables import (
    load,  # noqa: E402  (the container reader beside this script)
)

# The modalities the sampler keeps a trajectory for. A task without audio writes the video's
# alone, so a missing audio pair is not a missing pair.
MODALITIES = ("video", "audio")

EVALUATION = re.compile(r"^latents_(?P<modality>[a-z]+)_e(?P<index>\d+)\.h3t$")

# The arrangement both sides must state: the packed rows and the segments they are made of. The
# trajectory files hold the latents, whose shapes do not depend on the row count, so two sides that
# ran different layouts write files of the same shape and different trajectories. Without this
# statement the page reports an arithmetic difference whose cause is an arrangement, which is why a
# pair that does not state it, or states two different ones, is refused rather than reported.
LAYOUT_FIELDS = (
    "frames",
    "canvas",
    "text_rows",
    "condition_rows",
    "video_rows",
    "audio_rows",
    "packed_rows",
)


# The grid each side evaluated and how much of it. Two sides can agree on every layout field and
# still have run different schedules - a FastH3 student's `blocks` grid against the base `uniform`
# one, or eight evaluations against forty-nine - and their trajectories then differ everywhere for a
# reason that is not arithmetic. These fields are required of this side, whose producer states them,
# and read where present on the reference's: a statement that is absent is printed as absent rather
# than taken for agreement.
SCHEDULE_FIELDS = ("schedule", "evaluations", "num_steps")


def layout(directory: pathlib.Path, side: str) -> dict[str, object]:
    """The layout ``side`` states it ran, refused by name when it states none."""
    path = directory / "layout.json"
    if not path.is_file():
        raise SystemExit(
            f"the {side} states no layout: {path} is absent, so the pair cannot be shown to have"
            " run the same packed rows; a tree produced before the producers wrote `layout.json`"
            " is not comparable with one after them, and one produced at the smallest layout is"
            " not comparable with one produced at the case's"
        )
    try:
        stated = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} is not JSON: {error}") from error
    missing = [name for name in LAYOUT_FIELDS if name not in stated]
    if missing:
        raise SystemExit(
            f"{path} omits {', '.join(missing)}; a layout statement carries"
            f" {', '.join(LAYOUT_FIELDS)}"
        )
    return stated


def arrangements(ref: pathlib.Path, got: pathlib.Path) -> tuple[dict, dict]:
    """Both sides' layout statements, refused when they are not the same arrangement."""
    theirs = layout(ref, "reference")
    ours = layout(got, "this side")
    unstated = [name for name in SCHEDULE_FIELDS if name not in ours]
    if unstated:
        raise SystemExit(
            f"this side's layout statement omits {', '.join(unstated)}: a trajectory whose grid"
            " and step count are not stated cannot be shown to be the same schedule as the"
            " other side's, so it is refused rather than tabulated"
        )
    # Half a statement is worse than none: a reference tree that names its grid but not how much of
    # it ran would be read as agreeing about the step count. Both producers state all three
    # (`dump_denoiser.py` since `7e2571c07`), so a partial one is refused as a stale or hand-edited
    # tree rather than completed by assumption.
    unstated_theirs = [name for name in SCHEDULE_FIELDS if name not in theirs]
    if unstated_theirs:
        raise SystemExit(
            f"the reference's layout statement omits {', '.join(unstated_theirs)}: a trajectory"
            " whose grid and step count are not stated cannot be shown to be the same schedule as"
            " this one's, and an absent statement is not a matching one - the reference producer"
            " (`dump_denoiser.py`) states schedule, evaluations and num_steps, so a tree without"
            " them predates that and is regenerated rather than compared"
        )
    differ = [name for name in LAYOUT_FIELDS if theirs[name] != ours[name]]
    # The schedule: refused where both sides state it and they disagree, which is the pair a
    # student's block grid and the base uniform grid make.
    for name in SCHEDULE_FIELDS:
        if name in theirs and theirs[name] != ours[name]:
            differ.append(name)
    if differ:
        raise SystemExit(
            "the two sides ran different layouts, so a difference between their trajectories"
            " would measure the arrangement and not the arithmetic: "
            + "; ".join(
                f"{name}: this side {ours[name]}, reference {theirs[name]}"
                for name in differ
            )
        )
    return theirs, ours


def layout_line(side: str, stated: dict[str, object]) -> str:
    """One layout statement as a line a reader compares with the other side's."""
    height, width = stated["canvas"]
    return (
        f"layout  {side:<9} {stated['packed_rows']} packed rows = {stated['video_rows']} video"
        f" + {stated['audio_rows']} audio + {stated['text_rows']} text"
        f" + {stated['condition_rows']} condition ({stated['frames']} frames,"
        f" canvas {height}x{width}, {stated['schedule']} grid"
        f" {stated['evaluations']} of {stated['num_steps']} evaluations)"
    )


def evaluations(root: pathlib.Path) -> dict[tuple[str, int], pathlib.Path]:
    """Every evaluation file in ``root``, keyed by modality and index.

    The index is the file's own, not its position in a listing: ``e0`` is the draw and ``eN``
    the last step, and a tree that renumbered them would pair two different schedules.
    """
    if not root.is_dir():
        raise SystemExit(
            f"{root} is not a directory; --ref and --got name a case's `denoiser` directory"
        )
    found: dict[tuple[str, int], pathlib.Path] = {}
    for path in sorted(root.iterdir()):
        match = EVALUATION.match(path.name)
        if match is None or match.group("modality") not in MODALITIES:
            continue
        key = (match.group("modality"), int(match.group("index")))
        if key in found:
            raise SystemExit(
                f"{root} holds two files for {key[0]} evaluation {key[1]}"
            )
        found[key] = path
    if not found:
        raise SystemExit(
            f"{root} holds no latents_<modality>_e<i>.h3t; a trajectory stage writes them, and"
            " an empty directory is a stage that did not reach its first evaluation"
        )
    return found


def written(path: pathlib.Path) -> str:
    """When the file was last written, as UTC, so a page can say whether it came from one run.

    A stage writes its trajectory over seconds; a file whose time is far from its neighbours' came
    from another run, and a comparison that pairs it is comparing two tips. The number below is a
    fact about the file, not an inference about the run.
    """
    stamp = path.stat().st_mtime
    return datetime.datetime.fromtimestamp(stamp, datetime.UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def non_finite(tensor: object) -> int:
    """How many of a tensor's elements are not finite.

    `dump_tables.load` answers a torch tensor, so the count uses torch's own predicate where the
    tensor has one: a numpy ufunc on a torch tensor goes through an array-wrap path that numpy
    deprecates, and the warning would land on a page that is meant to be read for its numbers.
    """
    isfinite = getattr(tensor, "isfinite", None)
    if isfinite is not None:
        return int((~isfinite()).sum())
    return int((~np.isfinite(np.asarray(tensor))).sum())


def metrics(got: np.ndarray, ref: np.ndarray) -> dict[str, object]:
    """Relative L2, maximum absolute difference, cosine similarity and bit-equality.

    Both sides are read as float64 for the error columns so a difference of one element of a
    large table does not vanish under the values' own magnitude; ``exact`` is answered on the
    raw bytes, which is the only form of "the same tensor" that survives a rounding question.
    """
    a = np.asarray(got, dtype=np.float64).reshape(-1)
    b = np.asarray(ref, dtype=np.float64).reshape(-1)
    difference = a - b
    scale = float(np.linalg.norm(b))
    norm = float(np.linalg.norm(difference))
    cosine = (
        float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        if np.linalg.norm(a) and np.linalg.norm(b)
        else 0.0
    )
    return {
        "rel_l2": norm / scale if scale else norm,
        "max_abs": float(np.max(np.abs(difference)))
        if difference.size
        else 0.0,
        "cosine": cosine,
        "exact": bool(np.array_equal(np.asarray(got), np.asarray(ref))),
        "elements": int(b.size),
    }


def movement(previous: np.ndarray, current: np.ndarray) -> float:
    """How far one evaluation moved the sample, relative to where it started."""
    a = np.asarray(previous, dtype=np.float64).reshape(-1)
    b = np.asarray(current, dtype=np.float64).reshape(-1)
    scale = float(np.linalg.norm(a))
    return (
        float(np.linalg.norm(b - a)) / scale
        if scale
        else float(np.linalg.norm(b - a))
    )


def compare(ref: pathlib.Path, got: pathlib.Path) -> dict[str, object]:
    """Pair two trees and report every evaluation, refusing what it cannot interpret."""
    their_layout, our_layout = arrangements(ref, got)
    theirs, ours = evaluations(ref), evaluations(got)
    missing_ours = sorted(set(theirs) - set(ours))
    missing_theirs = sorted(set(ours) - set(theirs))
    if missing_ours or missing_theirs:

        def names(keys):
            return ", ".join(f"{modality} e{index}" for modality, index in keys)

        raise SystemExit(
            "the two trees do not hold the same evaluations:"
            f" only the reference has {names(missing_ours) or 'nothing'};"
            f" only this side has {names(missing_theirs) or 'nothing'}"
        )
    report: dict[str, object] = {
        "ref": str(ref),
        "got": str(got),
        "layout": {"ref": their_layout, "got": our_layout},
        "modalities": {},
    }
    for modality in MODALITIES:
        keys = sorted(index for name, index in theirs if name == modality)
        if not keys:
            continue
        rows = []
        previous: dict[str, np.ndarray] = {}
        for index in keys:
            reference = load(theirs[(modality, index)])
            candidate = load(ours[(modality, index)])
            if tuple(reference.shape) != tuple(candidate.shape):
                raise SystemExit(
                    f"{modality} e{index}: the reference is {tuple(reference.shape)} and this"
                    f" side is {tuple(candidate.shape)}"
                )
            if reference.dtype != candidate.dtype:
                raise SystemExit(
                    f"{modality} e{index}: the reference is {reference.dtype} and this side is"
                    f" {candidate.dtype}; a pair of two element types measures the conversion"
                    " and not the arithmetic"
                )
            # A non-finite element is not a comparison: `rel_l2` of a pair holding `nan` is `nan`,
            # and the page would show a number-shaped hole where a broken launch should be named.
            # A magnitude is the wrong signal here - a large finite error and a `nan` mean
            # different things - so the elements are counted and the side is named.
            for side, tensor in (
                ("reference", reference),
                ("this side", candidate),
            ):
                broken = non_finite(tensor)
                if broken:
                    raise SystemExit(
                        f"{modality} e{index}: the {side} holds {broken} non-finite"
                        " elements; that is a broken launch rather than a number to"
                        " compare"
                    )
            row = {
                "evaluation": index,
                **metrics(candidate, reference),
                "ref_written": written(theirs[(modality, index)]),
                "got_written": written(ours[(modality, index)]),
            }
            if index > keys[0]:
                row["ref_moved"] = movement(previous["ref"], reference)
                row["got_moved"] = movement(previous["got"], candidate)
                # A trajectory that did not move is a page about nothing: the draw is a valid
                # tensor, so nothing in a shape or a type refuses it.
                if row["ref_moved"] == 0.0 and row["got_moved"] == 0.0:
                    raise SystemExit(
                        f"{modality} e{index}: neither side moved since e{index - 1}; the pair"
                        " is two copies of one sample and no comparison of them means anything"
                    )
            rows.append(row)
            previous = {"ref": reference, "got": candidate}
        report["modalities"][modality] = rows
    return report


def table(report: dict[str, object]) -> str:
    """The report as the table a caller reads."""
    # The arrangement first: a reader who sees two different row counts on the first two lines has
    # already been told what a difference below them is about.
    lines = [
        layout_line(name, report["layout"][side])
        for side, name in (("ref", "reference"), ("got", "this side"))
    ]
    lines.append("")
    lines.append(
        "modality  evaluation  rel_l2       max_abs      cosine   exact  elements"
    )
    for modality, rows in report["modalities"].items():
        for row in rows:
            lines.append(
                f"{modality:<9} e{row['evaluation']:<10} {row['rel_l2']:<13.6e}"
                f" {row['max_abs']:<12.6e} {row['cosine']:<8.5f}"
                f" {'yes' if row['exact'] else 'no':<6} {row['elements']}"
            )
    return "\n".join(lines)


def _write(
    path: pathlib.Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
    values: np.ndarray,
) -> None:
    """Write one H3D1 container: magic, element code, rank, extents, then the payload."""
    # `dump_tables.CODES`: 0 f64, 1 f32, 2 i64, 3 i32, 4 bf16, 5 f16. The self-test's tensors are
    # FP32 and integer, and a wrong code here is what a first self-test run catches: it labels an
    # FP32 payload `i32` and the reader answers integers with the right magnitude.
    codes = {"<f8": 0, "<f4": 1, "<i8": 2, "<i4": 3, "<f2": 5}
    code = codes[dtype.str]
    header = (
        b"H3D1"
        + bytes([code, len(shape)])
        + b"".join(int(extent).to_bytes(8, "little") for extent in shape)
    )
    path.write_bytes(
        header + np.ascontiguousarray(values).astype(dtype).tobytes()
    )


def self_test() -> int:
    """Prove the harness: pairing, metrics, and the two refusals, on tensors it makes itself."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        ref, got = root / "ref", root / "got"
        ref.mkdir()
        got.mkdir()
        # Both producers state the layout they ran; the trees here are small, so the numbers are
        # the shape of the statement and not a request's.
        stated = {
            "frames": 22,
            "canvas": [256, 256],
            "text_rows": 4,
            "condition_rows": 0,
            "video_rows": 4,
            "audio_rows": 0,
            "packed_rows": 8,
        }
        # `stated` is the layout alone, which is what the mismatch probes vary; `full` is what a
        # producer writes, grid and step count included.
        full = dict(stated, schedule="uniform", evaluations=3, num_steps=3)
        for side in (ref, got):
            (side / "layout.json").write_text(json.dumps(full) + "\n")
        rng = np.random.default_rng(0)
        sample = rng.standard_normal((4, 8)).astype(np.float32)
        for index in range(3):
            moved = (sample * (1.0 - 0.1 * index)).astype(np.float32)
            _write(
                ref / f"latents_video_e{index}.h3t",
                moved.shape,
                moved.dtype,
                moved,
            )
            same = (
                moved
                if index == 0
                else (moved * (1.0 + 1e-3)).astype(np.float32)
            )
            _write(
                got / f"latents_video_e{index}.h3t",
                same.shape,
                same.dtype,
                same,
            )
        report = compare(ref, got)
        rows = report["modalities"]["video"]
        checks = [
            ("evaluation 0 is bit-identical", rows[0]["exact"] is True),
            ("evaluation 1 differs", rows[1]["exact"] is False),
            (
                "its relative error is the scale's",
                abs(rows[1]["rel_l2"] - 1e-3) < 1e-4,
            ),
            ("three evaluations paired", len(rows) == 3),
            ("movement is reported, not zero", rows[1]["ref_moved"] > 0.0),
            (
                "the layout both sides ran is in the report",
                report["layout"]["got"]["packed_rows"] == stated["packed_rows"]
                and report["layout"]["ref"]["text_rows"] == stated["text_rows"],
            ),
            (
                "and its line names the segments",
                layout_line("this side", full)
                == "layout  this side 8 packed rows = 4 video + 0 audio + 4 text + 0 condition"
                " (22 frames, canvas 256x256, uniform grid 3 of 3 evaluations)",
            ),
        ]
        # Two sides that agree on every layout field and disagree on the grid are refused: a
        # FastH3 student's block grid against the base uniform one differs everywhere, and that is
        # not a number about arithmetic.
        student = dict(stated, schedule="blocks", evaluations=3, num_steps=8)
        base = dict(stated, schedule="uniform", evaluations=3, num_steps=3)
        (got / "layout.json").write_text(json.dumps(student) + "\n")
        (ref / "layout.json").write_text(json.dumps(base) + "\n")
        try:
            arrangements(ref, got)
            checks.append(("a grid mismatch is refused", False))
        except SystemExit as error:
            checks.append(
                ("a grid mismatch is refused", "schedule" in str(error))
            )
        # A reference tree that predates its producer stating the grid is refused by name: an
        # absent statement is not a matching one.
        (ref / "layout.json").write_text(json.dumps(stated) + "\n")
        (got / "layout.json").write_text(json.dumps(student) + "\n")
        try:
            arrangements(ref, got)
            checks.append(("an unstated reference grid is refused", False))
        except SystemExit as error:
            checks.append(
                ("an unstated reference grid is refused", "omits schedule" in str(error))
            )
        # Half a statement is refused too: a tree naming its grid but not its step count would be
        # read as agreeing about the steps.
        (ref / "layout.json").write_text(
            json.dumps(dict(stated, schedule="blocks")) + "\n"
        )
        try:
            arrangements(ref, got)
            checks.append(("a partial reference statement is refused", False))
        except SystemExit as error:
            checks.append(
                (
                    "a partial reference statement is refused",
                    "omits evaluations" in str(error),
                )
            )
        # And this side must state its own grid: a trajectory that does not say what it ran is
        # refused rather than tabulated.
        (ref / "layout.json").write_text(json.dumps(full) + "\n")
        (got / "layout.json").write_text(json.dumps(stated) + "\n")
        try:
            arrangements(ref, got)
            checks.append(("this side's own grid is required", False))
        except SystemExit as error:
            checks.append(("this side's own grid is required", "schedule" in str(error)))
        for side in (ref, got):
            (side / "layout.json").write_text(json.dumps(full) + "\n")
        # A non-finite element is refused by name rather than reported as a number.
        broken = sample.copy()
        broken[3] = np.nan
        _write(got / "latents_video_e2.h3t", broken.shape, broken.dtype, broken)
        try:
            compare(ref, got)
            checks.append(("a non-finite element is refused", False))
        except SystemExit as error:
            checks.append(
                (
                    "a non-finite element is refused",
                    "non-finite elements" in str(error),
                )
            )
        restored = (sample * 0.8).astype(np.float32)
        _write(
            got / "latents_video_e2.h3t",
            restored.shape,
            restored.dtype,
            restored,
        )

        # A missing file on one side is refused by name.
        (got / "latents_video_e2.h3t").unlink()
        try:
            compare(ref, got)
            checks.append(("a missing evaluation is refused", False))
        except SystemExit as error:
            checks.append(
                (
                    "a missing evaluation is refused",
                    "only the reference has video e2" in str(error),
                )
            )
        # A trajectory that did not move is refused even though every tensor is valid.
        _write(
            got / "latents_video_e2.h3t",
            sample.shape,
            sample.dtype,
            (sample * 0.8).astype(np.float32),
        )
        for index in range(3):
            _write(
                got / f"latents_video_e{index}.h3t",
                sample.shape,
                sample.dtype,
                sample,
            )
            _write(
                ref / f"latents_video_e{index}.h3t",
                sample.shape,
                sample.dtype,
                sample,
            )
        try:
            compare(ref, got)
            checks.append(("a trajectory that did not move is refused", False))
        except SystemExit as error:
            checks.append(
                (
                    "a trajectory that did not move is refused",
                    "neither side moved" in str(error),
                )
            )

        # Two sides that ran different layouts are refused by the field that differs, because the
        # files they wrote have the same shape and a difference between them would read as
        # arithmetic.
        (got / "layout.json").write_text(
            json.dumps(
                dict(full, condition_rows=192, packed_rows=200),
            )
            + "\n"
        )
        try:
            compare(ref, got)
            checks.append(("a differing layout is refused", False))
        except SystemExit as error:
            checks.append(
                (
                    "a differing layout is refused",
                    "different layouts" in str(error)
                    and "condition_rows: this side 192, reference 0"
                    in str(error)
                    and "packed_rows: this side 200, reference 8" in str(error),
                )
            )
        # A side that states no layout is refused by name: an unstated arrangement is the case the
        # whole check exists for, and assuming it equal is what produced a wrong page.
        (got / "layout.json").unlink()
        try:
            compare(ref, got)
            checks.append(("an unstated layout is refused", False))
        except SystemExit as error:
            checks.append(
                (
                    "an unstated layout is refused",
                    "states no layout" in str(error),
                )
            )
    for name, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
    return 0 if all(passed for _, passed in checks) else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", type=pathlib.Path, help="the reference's `denoiser` directory"
    )
    parser.add_argument(
        "--got", type=pathlib.Path, help="this model's `denoiser` directory"
    )
    parser.add_argument("--out", type=pathlib.Path, help="report.json to write")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the harness on tensors it makes itself, with no checkpoint",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.ref is None or args.got is None:
        parser.error(
            "--ref and --got name the two trees, or --self-test proves the harness"
        )
    report = compare(args.ref, args.got)
    print(table(report))
    # The two trees' write times, so a page states whether each trajectory came from one run.
    for side in ("ref", "got"):
        times = sorted(
            row[f"{side}_written"]
            for rows in report["modalities"].values()
            for row in rows
        )
        if times:
            print(
                f"{side} written  {times[0]} .. {times[-1]}  ({len(times)} files)"
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
