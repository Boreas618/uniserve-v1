#!/usr/bin/env python3
"""Compare the first device run's DiT stack output with the reference's, row block by row block.

Design 10's diagnosis, never an acceptance gate. Both sides exist already:

* **ours** - every run of `dit_layer --stage block` since `188b8b1d5` writes
  `artifacts/h3-stage1/denoiser/stack_out.h3t`, one device pass of the case's layer count at step 0
  over the case's own input rows;
* **the reference's** - one `dump_denoiser.py --probe stack_out` invocation on the same case.

The table names a **row block**, not a tensor: 842 rows by 5376 columns in one number names nothing,
and every block of the packed order (video, audio, text, condition) is a different computation. A
number without a row block is a claim about 842 things that names none of them.

What the table states, in the order it states it:

1. **the shape and the row order** - the file's rows and columns against the case's own statement of
   the four segments, and the reference's shape against ours. Two sides whose orders differ make
   every number below them meaningless while looking healthy, so a disagreement is **refused by
   name** rather than tabulated;
2. **per row block**: `rel_l2`, `max_abs` in the block's own units, the index of the worst element in
   the tensor's coordinates (row, column), the cosine similarity, and `exact`;
3. **the element types of both sides and the widening** - the container's code per side, widened to
   FP32 for the comparison, because a BF16 stack output differs from an FP32 reference by its own
   rounding and that rounding must not be read as arithmetic;
4. **`reference not run`** for every block whose reference file is absent, which is a row that has not
   been measured and is written rather than omitted.

    python crates/models/minimax-h3/tools/compare_stack.py \
        --ours artifacts/h3-stage1/denoiser \
        --ref artifacts/h3-stage1-reference/denoiser \
        --out artifacts/diagnosis/minimax-h3/dit/stack
    python crates/models/minimax-h3/tools/compare_stack.py --self-test

It needs NumPy only: the harness is proved without a device before any checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy

# The `H3D1` container's element codes (`crates/models/minimax-h3/src/diagnosis.rs`, and the
# reference writer's own table in `dump_tables.py`): narrow types carry the same number of bytes and
# are widened here, once, so nothing below the reader sees a half.
ELEMENTS = {
    0: ("f64", False),
    1: ("f32", False),
    2: ("i64", False),
    3: ("i32", False),
    4: ("bf16", True),
    5: ("f16", True),
}

# The packed order's four segments, in the order the layout states them. The row table's groups run
# video, audio, text, condition: a row block that starts anywhere else would compare two different
# computations and call the difference arithmetic.
SEGMENTS = ("video", "audio", "text", "condition")

# The case fields the shape check reads. A case that omits one is refused: the row order is a
# statement about the file, and a file whose order is not stated cannot be shown to be in the order
# the reference ran.
CASE_FIELDS = ("packed_rows", "hidden", "video_rows", "audio_rows", "text_rows", "condition_rows")

EPS = 1e-12


class PairError(Exception):
    """A pair the table cannot interpret, named as such rather than reported as numbers."""


def read_h3t(path: pathlib.Path) -> tuple[numpy.ndarray, str]:
    """Read one `H3D1` tensor, widened to FP32, and the element name it was stored in."""
    data = path.read_bytes()
    if data[:4] != b"H3D1":
        raise PairError(f"{path}: not an H3D1 container")
    code, rank = data[4], data[5]
    if code not in ELEMENTS:
        raise PairError(f"{path}: element code {code} is not one of {sorted(ELEMENTS)}")
    name, narrow = ELEMENTS[code]
    shape = tuple(
        int.from_bytes(data[6 + index * 8 : 14 + index * 8], "little")
        for index in range(rank)
    )
    count = int(numpy.prod(shape)) if shape else 1
    payload = data[6 + rank * 8 :]
    if len(payload) != count * (2 if narrow else numpy.dtype(
        {"f64": numpy.float64, "f32": numpy.float32, "i64": numpy.int64, "i32": numpy.int32}[name]
    ).itemsize):
        raise PairError(f"{path}: a payload that does not match its shape {shape}")
    if narrow:
        # BF16 and F16 widen exactly: every half is a value FP32 holds, so the widening is not a
        # rounding step and the comparison below measures the stack and not this reader.
        raw = numpy.frombuffer(payload, dtype=numpy.uint16).reshape(shape)
        if name == "bf16":
            values = (raw.astype(numpy.uint32) << 16).view(numpy.float32)
        else:
            values = raw.view(numpy.float16).astype(numpy.float32)
        return values.reshape(shape), name
    dtype = {
        "f64": numpy.float64,
        "f32": numpy.float32,
        "i64": numpy.int64,
        "i32": numpy.int32,
    }[name]
    return numpy.frombuffer(payload, dtype=dtype).reshape(shape).astype(
        numpy.float32 if dtype in (numpy.float64, numpy.float32) else dtype
    ), name


def write_h3t(path: pathlib.Path, values: numpy.ndarray, code: int) -> None:
    """Write one `H3D1` tensor, for the self-test's synthetic pairs."""
    array = numpy.ascontiguousarray(values)
    if code in (4, 5):
        if code == 4:
            raw = (array.astype(numpy.float32).view(numpy.uint32) >> 16).astype(numpy.uint16)
        else:
            raw = array.astype(numpy.float16).view(numpy.uint16)
        payload = raw.tobytes()
    else:
        payload = array.tobytes()
    header = b"H3D1" + bytes([code, array.ndim])
    header += b"".join(int(extent).to_bytes(8, "little") for extent in array.shape)
    path.write_bytes(header + payload)


def case_of(directory: pathlib.Path) -> dict:
    """This side's statement of the case, refused by name when it does not state the row order."""
    path = directory / "case.json"
    if not path.is_file():
        raise PairError(
            f"{path} is absent: the shape check reads the case's own statement of the packed rows"
            " and their four segments, and a file whose order is not stated cannot be shown to be"
            " in the order the reference ran"
        )
    try:
        stated = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise PairError(f"{path} is not JSON: {error}") from error
    missing = [name for name in CASE_FIELDS if name not in stated]
    if missing:
        raise PairError(
            f"{path} omits {', '.join(missing)}: the shape check needs {', '.join(CASE_FIELDS)}"
            " - the harness states them beside the tensor it wrote"
        )
    total = sum(int(stated[name + "_rows"]) for name in SEGMENTS)
    if total != int(stated["packed_rows"]):
        raise PairError(
            f"{path}: the four segments state {total} rows and the case's packed count is"
            f" {stated['packed_rows']} - one of the two is wrong, and the row order is what the"
            " table below is an index into"
        )
    return stated


def metrics(got: numpy.ndarray, ref: numpy.ndarray) -> dict[str, object]:
    """The block's numbers, in the tensor's own coordinates for the worst element."""
    a = got.astype(numpy.float64)
    b = ref.astype(numpy.float64)
    difference = numpy.abs(a - b)
    flat = int(numpy.argmax(difference))
    worst = numpy.unravel_index(flat, difference.shape)
    denominator = max(float(numpy.linalg.norm(b)), EPS)
    norm_a, norm_b = float(numpy.linalg.norm(a)), float(numpy.linalg.norm(b))
    cosine = (
        float(numpy.sum(a * b)) / max(norm_a * norm_b, EPS)
        if norm_a > 0.0 and norm_b > 0.0
        else 1.0
    )
    return {
        "rel_l2": float(numpy.linalg.norm(a - b)) / denominator,
        "max_abs": float(difference.flat[flat]),
        "worst_index": [int(index) for index in worst],
        "cosine": cosine,
        "exact": bool(numpy.array_equal(got, ref)),
    }


def blocks(stated: dict) -> list[tuple[str, int, int]]:
    """Each segment's name, first row and last row in the packed order."""
    first = 0
    spans = []
    for name in SEGMENTS:
        count = int(stated[name + "_rows"])
        spans.append((name, first, first + count))
        first += count
    return spans


def compare(ours: pathlib.Path, ref: pathlib.Path) -> dict[str, object]:
    """The table: the shape check, then one row per packed segment."""
    stated = case_of(ours)
    rows, columns = int(stated["packed_rows"]), int(stated["hidden"])

    got_path = ours / "stack_out.h3t"
    if not got_path.is_file():
        raise PairError(
            f"{got_path} is absent: this side of the pair is the harness's own output, written by"
            " `dit_layer --stage block --out <dir>`"
        )
    got, got_name = read_h3t(got_path)
    if got.shape != (rows, columns):
        raise PairError(
            f"{got_path} holds {tuple(got.shape)} and the case states ({rows}, {columns}): the file"
            " is from another layout or another run, and the segments below would index rows it"
            " does not hold"
        )

    table: dict[str, object] = {
        "ours": str(ours),
        # The checkpoint is the case's own statement of the tree it read, so the page names the
        # weights the numbers were computed from without a second flag a caller can get wrong.
        "checkpoint": stated.get("checkpoint", "not stated"),
        "ref": str(ref),
        "shape": [rows, columns],
        "segments": {name: int(stated[name + "_rows"]) for name in SEGMENTS},
        "elements": {"ours": got_name},
        "blocks": [],
        "ref_run": False,
    }

    ref_path = ref / "stack_out.h3t"
    if not ref_path.is_file():
        for name, first, last in blocks(stated):
            table["blocks"].append(
                {
                    "block": name,
                    "rows": [first, last],
                    "state": "reference not run",
                }
            )
        return table

    theirs, ref_name = read_h3t(ref_path)
    table["elements"]["ref"] = ref_name
    if theirs.shape != got.shape:
        raise PairError(
            f"the two sides hold different shapes: ours {tuple(got.shape)}, the reference's"
            f" {tuple(theirs.shape)} - a pair of two different layouts is not a comparison, and"
            " the segments below cannot index both"
        )
    # The input rows both sides started from: the reference ran on this side's `stack_in.h3t`, so a
    # tree whose input differs is a stale run of another case rather than a difference of arithmetic.
    if (ref / "stack_in.h3t").is_file():
        their_in, _ = read_h3t(ref / "stack_in.h3t")
        our_in, _ = read_h3t(ours / "stack_in.h3t")
        if their_in.shape != our_in.shape or not numpy.array_equal(their_in, our_in):
            raise PairError(
                "the two sides started from different input rows: the reference tree's"
                " `stack_in.h3t` is not this side's, so the pair is two runs of two cases and every"
                " number below it would measure the case and not the stack"
            )
        table["input_rows_equal"] = True
    table["ref_run"] = True
    for name, first, last in blocks(stated):
        table["blocks"].append(
            {
                "block": name,
                "rows": [first, last],
                "state": "measured",
                **metrics(got[first:last], theirs[first:last]),
            }
        )
    return table


def markdown(table: dict[str, object], commit: str, checkpoint: str) -> str:
    """The table as the page that goes under `artifacts/`."""
    rows, columns = table["shape"]
    segments = table["segments"]
    lines = [
        "# The DiT stack: ours against the reference, one row block at a time",
        "",
        f"commit `{commit}`, checkpoint `{checkpoint}`. **A diagnosis, never an acceptance gate.**",
        "",
        "| file | what it is |",
        "| --- | --- |",
        f"| `{table['ours']}/stack_out.h3t` | this model's stack output, one device pass at step 0 |",
        f"| `{table['ref']}/stack_out.h3t` | the reference's own stack on the same input rows |",
        "",
        f"The shape check passed: **{rows} rows of {columns}** columns, the case's own segments "
        f"({segments['video']} video + {segments['audio']} audio + {segments['text']} text + "
        f"{segments['condition']} condition).",
        "",
        f"Element types: ours **{table['elements']['ours']}**, the reference's "
        f"**{table['elements'].get('ref', 'not run')}**; both are widened to FP32 once, in the"
        " reader, so no row below it measures the widening.",
        "",
        "| block | rows | rel_l2 | max_abs | worst (row, column) | cosine | exact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for block in table["blocks"]:
        first, last = block["rows"]
        if block["state"] != "measured":
            lines.append(
                f"| {block['block']} | {first}-{last - 1} | reference not run | "
                "reference not run | reference not run | reference not run | no |"
            )
            continue
        lines.append(
            f"| {block['block']} | {first}-{last - 1} | {block['rel_l2']:.6e} | "
            f"{block['max_abs']:.6e} | ({block['worst_index'][0]}, {block['worst_index'][1]}) | "
            f"{block['cosine']:.9f} | {'yes' if block['exact'] else 'no'} |"
        )
    lines += [
        "",
        "`rel_l2` is the block's own Frobenius norm, so a block of 448 rows and a block of 74 are"
        " read against themselves. `worst` is in the tensor's coordinates, not a flat offset: it"
        " names one packed row and one channel.",
        "",
        "## The commands",
        "",
        "```bash",
        "# this side: one device pass of the case's layer count at step 0",
        "cargo run -q -p minimax-h3 --example dit_layer -- --stage block \\",
        "    --root /opt/uv/models/MiniMax-H3 --device 0 --out artifacts/h3-stage1",
        "",
        "# the reference's: its own stack on this side's input rows",
        ".venv/bin/python crates/models/minimax-h3/tools/dump_denoiser.py --probe stack_out \\",
        "    --checkpoint /opt/uv/models/MiniMax-H3 --stack artifacts/h3-stage1/denoiser \\",
        "    --out artifacts/h3-stage1-reference",
        "",
        "# the table",
        "python crates/models/minimax-h3/tools/compare_stack.py \\",
        "    --ours artifacts/h3-stage1/denoiser \\",
        "    --ref artifacts/h3-stage1-reference/denoiser \\",
        "    --out artifacts/diagnosis/minimax-h3/dit/stack",
        "```",
        "",
        "Both producers state the checkpoint as their flag's value; the commit is the tree that ran"
        " the comparison, and it is stamped here so no row is undated.",
    ]
    return "\n".join(lines) + "\n"


def self_test() -> int:
    """Prove the harness: the reader, the block table, and the three refusals."""
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        ours, ref = root / "ours", root / "ref"
        ours.mkdir()
        ref.mkdir()
        stated = {
            "packed_rows": 12,
            "hidden": 6,
            "video_rows": 5,
            "audio_rows": 2,
            "text_rows": 3,
            "condition_rows": 2,
            "dtype": "bf16",
        }
        (ours / "case.json").write_text(json.dumps(stated) + "\n")
        rng = numpy.random.default_rng(7)
        values = rng.standard_normal((12, 6)).astype(numpy.float32)
        write_h3t(ours / "stack_out.h3t", values, 4)
        write_h3t(ours / "stack_in.h3t", values * 0.5, 4)

        # 1. A BF16 pair widens exactly: the reader's value is the half's own value, not a rounding.
        read, name = read_h3t(ours / "stack_out.h3t")
        rounded = (
            (values.view(numpy.uint32) >> 16).astype(numpy.uint16).astype(numpy.uint32) << 16
        ).view(numpy.float32).reshape(values.shape)
        checks.append(("a BF16 tensor reads back as its own halves", name == "bf16" and numpy.array_equal(read, rounded)))

        # 2. The reader against the **other** writer's bytes: the committed fixture is a BF16 H3D1 and
        #    its `.npy` twin, both written by the reference producer (`dump_tables.dump`), so this
        #    proves the widening against an implementation this crate does not own - the one thing a
        #    reader tested only on its own output cannot show.
        fixture = (
            pathlib.Path(__file__).resolve().parents[1] / "tests" / "data" / "h3t-bf16" / "stack_out"
        )
        if fixture.with_suffix(".h3t").is_file():
            theirs, fixture_name = read_h3t(fixture.with_suffix(".h3t"))
            twin = numpy.load(fixture.with_suffix(".npy"))
            checks.append(
                (
                    "a BF16 container written by the reference producer reads as its own .npy twin",
                    fixture_name == "bf16" and numpy.array_equal(theirs, twin),
                )
            )
        else:
            checks.append((f"the reference producer's BF16 fixture is committed at {fixture}.h3t", False))

        # 3. The reference absent: every block says so, and no number is invented.
        table = compare(ours, ref)
        checks.append(
            (
                "an absent reference is four `reference not run` rows",
                not table["ref_run"]
                and len(table["blocks"]) == 4
                and all(block["state"] == "reference not run" for block in table["blocks"]),
            )
        )
        checks.append(
            ("the blocks are in the packed order", [b["block"] for b in table["blocks"]] == ["video", "audio", "text", "condition"])
        )
        checks.append(("and their spans tile the rows", table["blocks"][-1]["rows"][1] == 12))

        # 3. An exact pair is `exact` in every block, and the worst index is the block's own row.
        write_h3t(ref / "stack_out.h3t", values, 4)
        write_h3t(ref / "stack_in.h3t", values * 0.5, 4)
        table = compare(ours, ref)
        checks.append(
            ("an identical pair is exact in every block", all(block["exact"] for block in table["blocks"]))
        )
        checks.append(("and its max_abs is zero", all(block["max_abs"] == 0.0 for block in table["blocks"])))

        # 4. One element moved in the audio block is reported in the audio block, at its own index.
        moved = rounded.copy()
        moved[6, 3] += 0.25
        write_h3t(ref / "stack_out.h3t", moved, 4)
        table = compare(ours, ref)
        audio = table["blocks"][1]
        checks.append(
            (
                "a moved element is attributed to its block and its coordinate",
                audio["block"] == "audio"
                and audio["worst_index"] == [1, 3]
                # 0.25 added to a BF16 value is stored back as a BF16, so the difference is the
                # step and the half's rounding around it - not the exact 0.25 the test added.
                and 0.2 < audio["max_abs"] < 0.3
                and table["blocks"][0]["exact"],
            )
        )
        checks.append(
            (
                "the untouched blocks stay exact",
                table["blocks"][2]["exact"] and table["blocks"][3]["exact"],
            )
        )

        # 5. Two different inputs: the pair is two cases, and is refused rather than tabulated.
        write_h3t(ref / "stack_in.h3t", values * 0.75, 4)
        try:
            compare(ours, ref)
            checks.append(("a stale input pair is refused", False))
        except PairError as error:
            checks.append(("a stale input pair is refused", "input rows" in str(error)))
        write_h3t(ref / "stack_in.h3t", values * 0.5, 4)

        # 6. A shape disagreement is refused by name, not reported as arithmetic.
        write_h3t(ref / "stack_out.h3t", values[:10], 4)
        try:
            compare(ours, ref)
            checks.append(("a shape disagreement is refused", False))
        except PairError as error:
            checks.append(("a shape disagreement is refused", "different shapes" in str(error)))
        write_h3t(ref / "stack_out.h3t", values, 4)

        # 7. A case whose segments do not tile its own row count is refused: the order is a claim.
        broken = dict(stated, video_rows=6)
        (ours / "case.json").write_text(json.dumps(broken) + "\n")
        try:
            compare(ours, ref)
            checks.append(("segments that do not tile the rows are refused", False))
        except PairError as error:
            checks.append(("segments that do not tile the rows are refused", "segments state" in str(error)))
        (ours / "case.json").write_text(json.dumps(stated) + "\n")

        # 8. A case that states no order at all is refused: this is the harness's own new field.
        thin = {name: stated[name] for name in ("packed_rows", "hidden")}
        (ours / "case.json").write_text(json.dumps(thin) + "\n")
        try:
            compare(ours, ref)
            checks.append(("a case that states no segments is refused", False))
        except PairError as error:
            checks.append(("a case that states no segments is refused", "omits" in str(error)))
        (ours / "case.json").write_text(json.dumps(stated) + "\n")

        # 9. The page renders, and it renders the four blocks.
        page = markdown(compare(ours, ref), "self-test", "self-test")
        checks.append(
            (
                "the page renders a row per block",
                all(f"| {name} |" in page for name in ("video", "audio", "text", "condition")),
            )
        )

    for name, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
    print(f"self-test {sum(not passed for _, passed in checks)} failure(s)")
    return 1 if any(not passed for _, passed in checks) else 0


def git_commit() -> str:
    """The tree's commit, which the page stamps beside every number."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=pathlib.Path(__file__).resolve().parents[4],
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a tree without git stamps what it knows
        return "unknown"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=pathlib.Path, default=pathlib.Path("artifacts/h3-stage1/denoiser"))
    parser.add_argument(
        "--ref", type=pathlib.Path, default=pathlib.Path("artifacts/h3-stage1-reference/denoiser")
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/diagnosis/minimax-h3/dit/stack"),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="the checkpoint path the page states; default: the case's own `checkpoint` field",
    )
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        return self_test()
    try:
        table = compare(arguments.ours, arguments.ref)
    except PairError as refusal:
        raise SystemExit(str(refusal)) from refusal
    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "stack.json").write_text(json.dumps(table, indent=2) + "\n")
    page = markdown(table, git_commit(), arguments.checkpoint or table["checkpoint"])
    (arguments.out / "stack.md").write_text(page)
    print(page)
    print(f"wrote {arguments.out / 'stack.md'} and {arguments.out / 'stack.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
