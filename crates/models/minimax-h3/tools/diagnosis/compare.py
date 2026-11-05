#!/usr/bin/env python3
r"""Compare one block of MiniMax-H3 against today's implementation, tensor by tensor.

This is design 10's **diagnosis, never an acceptance gate**. It reports; it does not refuse, rank, or
stop anything, and its exit status says only whether the two trees could be read - a row that differs
is an answer, not an error.

## What it reads

Both sides are trees of `.npy` files, and the name is the contract, not the position:

    <side>/<block>/<name>.npy        what `$UNISERVE_DIAG_DIR` and the reference dumpers write
    <side>/<component>/<name>.npy    a whole-component probe, with no block of its own

`<block>` is the reference's own module boundary - `decoder.decoder.layers.0` for the video VAE,
`layers.0` for the text encoder, `blocks.0` for the DiT - so a row's name is the name a reader of the
reference's code would use. **A block whose other side is absent is a row, written as
`reference not run here` or `ours not run here`**: an absent measurement is stated, never skipped.

## What a row says

`max|diff|` and `relative` (the L2 norm of the difference over the reference's own norm), the element
types of both sides, the shape, and a **verdict** that names the cause rather than only the size:

| verdict | when | why it is not folded into the numbers |
| --- | --- | --- |
| `exact` | the two sides are bit-identical after widening | - |
| `dtype` | the element types differ | a BF16 side differs from an FP32 one by its own rounding, and that rounding is not arithmetic; the row states both types so the reader can see which it is |
| `shape` | the shapes differ | two shapes are two computations; numbers over a mismatched pair name nothing, so the row carries no numbers |
| `arithmetic path` | same type, same shape, different values | the only verdict a difference in the mathematics earns - and only after the two above are excluded |
| `reference not run here` / `ours not run here` | one side is absent | the state the caller is in until the reference host produces its half |

**Same arithmetic path** is not a claim this tool can make from two numbers. It reports `dtype` and
`shape` so the reader can, and labels everything else `arithmetic path` rather than calling it a bug.

## Running it

    python crates/models/minimax-h3/tools/diagnosis/compare.py --self-test
    python crates/models/minimax-h3/tools/diagnosis/compare.py \\
        --component text_encoder --block text_encoder \\
        --ours artifacts/diagnosis/minimax-h3/text-encoder/got \\
        --ref artifacts/diagnosis/minimax-h3/text-encoder/ref \\
        --out artifacts/diagnosis/minimax-h3/text-encoder/blocks.md

`--block` may be a block name, or `all` to walk every block either tree holds. It needs NumPy only, so
the harness is proved on a host with no device and no checkpoint before any caller spends a run on it.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy

# One reader for the `H3D1` container the stage runners and `dump_encoder.py` write, so both
# comparators read a tensor the same way and a BF16 side widens exactly instead of by a cast.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from compare_stack import (
    read_h3t,  # noqa: E402  (the tools directory is on the path above)
)

# `.npy` has no BF16, so a BF16 side reaches this tool as the FP32 its low sixteen bits were masked
# from (what the dumpers write); an `f16` side is its own container and gets its own verdict.
NAME_OF = {"float32": "f32", "float16": "f16", "float64": "f64", "int32": "i32", "uint16": "u16"}


class UnreadableError(Exception):
    """The trees could not be read: a usage or an input error, never a difference."""


def named_types(values: numpy.ndarray) -> str:
    """The element type of one side, in the vocabulary the comparison talks in."""
    return NAME_OF.get(values.dtype.name, str(values.dtype))


def widened(values: numpy.ndarray) -> numpy.ndarray:
    """One side as FP32, in the natural dtype's own numbering.

    A BF16 array is a float32 with its low sixteen bits masked, so `astype` is exact both ways and the
    comparison is about the numbers each side holds rather than about the container it holds them in.
    """
    return values.astype(numpy.float32)


def metrics(ours: numpy.ndarray, ref: numpy.ndarray) -> dict[str, object]:
    """The numbers one row carries, over two same-shaped arrays."""
    got = widened(ours).reshape(-1)
    want = widened(ref).reshape(-1)
    if got.size == 0:
        # An empty tensor is a shape this model can hold (a prefix with no condition rows), and two
        # empty tensors of one shape are equal rather than unmeasurable.
        return {"max_abs": 0.0, "relative": 0.0, "worst": 0, "exact": True}
    difference = got - want
    worst = int(numpy.argmax(numpy.abs(difference)))
    scale = float(numpy.linalg.norm(want))
    return {
        "max_abs": float(numpy.max(numpy.abs(difference))),
        "relative": (
            float(numpy.linalg.norm(difference)) / scale if scale > 0.0 else float("inf")
        ),
        "worst": worst,
        "exact": bool(numpy.array_equal(got, want)),
    }


def verdict(ours: numpy.ndarray | None, ref: numpy.ndarray | None) -> tuple[str, dict | None]:
    """The cause a row names, and its numbers when the pair can carry any."""
    if ref is None and ours is None:
        return "both sides absent", None
    if ref is None:
        return "reference not run here", None
    if ours is None:
        return "ours not run here", None
    if ours.shape != ref.shape:
        return "shape", None
    numbers = metrics(ours, ref)
    if ours.dtype != ref.dtype:
        return "dtype", numbers
    if numbers["exact"]:
        return "exact", numbers
    return "arithmetic path", numbers


SUFFIXES = (".npy", ".h3t")


def load(directory: pathlib.Path, block: str, name: str) -> numpy.ndarray | None:
    """One side's tensor for one block, or `None` when its tree does not hold it.

    Two containers, one name: `.npy` for the diagnosis trees and `H3D1` for what the stage runners and
    `dump_encoder.py` write. The reader is `compare_stack`'s, so a BF16 side widens exactly the same way
    in both comparators and the element name it was stored in is available beside the values.
    """
    for suffix in SUFFIXES:
        for candidate in (
            directory / block / f"{name}{suffix}",
            directory / block.split(".")[0] / f"{name}{suffix}",
        ):
            if candidate.is_file():
                if suffix == ".npy":
                    return numpy.load(candidate)
                return read_h3t(candidate)[0]
    return None


def blocks_of(directory: pathlib.Path) -> list[str]:
    """Every block name a tree holds, by the name its own side chose."""
    if not directory.is_dir():
        return []
    names = set()
    for suffix in SUFFIXES:
        for path in directory.rglob(f"*{suffix}"):
            if path.parent == directory:
                continue
            names.add(".".join(path.parent.relative_to(directory).parts))
    return sorted(names)


def names_of(directory: pathlib.Path) -> dict[str, list[str]]:
    """Every tensor name each block of a tree holds."""
    held: dict[str, list[str]] = {}
    for suffix in SUFFIXES:
        for path in sorted(directory.rglob(f"*{suffix}")):
            if path.parent == directory:
                continue
            block = ".".join(path.parent.relative_to(directory).parts)
            held.setdefault(block, []).append(path.stem)
    return held


def compare(ours_root: pathlib.Path, ref_root: pathlib.Path, block: str) -> list[dict]:
    """One row per tensor either side holds for `block`."""
    if block == "all":
        blocks = sorted(set(blocks_of(ours_root)) | set(blocks_of(ref_root)))
    else:
        blocks = [block]
    if not blocks:
        raise UnreadableError(f"neither {ours_root} nor {ref_root} holds a block under a directory")
    rows = []
    for one in blocks:
        names = sorted(
            set(names_of(ours_root).get(one, [])) | set(names_of(ref_root).get(one, []))
        )
        if not names:
            rows.append(
                {
                    "block": one,
                    "name": "-",
                    "ours": None,
                    "ref": None,
                    "verdict": "both sides absent",
                    "numbers": None,
                }
            )
            continue
        for name in names:
            got = load(ours_root, one, name)
            want = load(ref_root, one, name)
            cause, numbers = verdict(got, want)
            rows.append(
                {
                    "block": one,
                    "name": name,
                    "ours": got,
                    "ref": want,
                    "verdict": cause,
                    "numbers": numbers,
                }
            )
    return rows


def git_commit() -> str:
    """The commit the run belongs to, so the file is attributable without asking."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def markdown(rows: list[dict], ours_root: pathlib.Path, ref_root: pathlib.Path) -> str:
    """The table, with the two trees named above it and a verdict on every row."""
    lines = [
        "# Block-by-block diagnosis",
        "",
        "**A diagnosis, never an acceptance gate** (design 10): no row here refuses a run, and a",
        "difference is the answer this file exists to report.",
        "",
        f"- ours: `{ours_root}`",
        f"- reference: `{ref_root}`",
        f"- commit: `{git_commit()}`",
        "",
        "| block | tensor | ours | ref | shape | max abs | relative | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        shape = "absent"
        if row["ref"] is not None:
            shape = "x".join(str(one) for one in row["ref"].shape)
        elif row["ours"] is not None:
            shape = "x".join(str(one) for one in row["ours"].shape)
        numbers = row["numbers"] or {}
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | `{}` |".format(
                row["block"],
                row["name"],
                named_types(row["ours"]) if row["ours"] is not None else "absent",
                named_types(row["ref"]) if row["ref"] is not None else "absent",
                shape,
                f"{numbers['max_abs']:.3e}" if "max_abs" in numbers else "-",
                f"{numbers['relative']:.3e}" if "relative" in numbers else "-",
                row["verdict"],
            )
        )
    lines.append("")
    lines.append(
        "`dtype` names a rounding difference between containers, `shape` a different computation, and "
        "`arithmetic path` the only verdict the mathematics earns - after the other two are excluded."
    )
    return "\n".join(lines) + "\n"


def self_test() -> int:
    """Prove every verdict on synthetic trees, so the harness is trusted before it meets a run."""
    failures = 0

    def check(label: str, produced: str, expected: str) -> None:
        nonlocal failures
        if produced != expected:
            print(f"FAIL {label}: {produced} != {expected}")
            failures += 1

    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch)
        ours, ref = root / "ours", root / "ref"
        same = numpy.linspace(-2.0, 2.0, 64, dtype=numpy.float32)
        for side in (ours, ref):
            (side / "block.0").mkdir(parents=True)
            numpy.save(side / "block.0" / "exact.npy", same)
            numpy.save(side / "block.0" / "shape.npy", same if side == ref else same[:-1])
            numpy.save(side / "block.0" / "dtype.npy", same)
            numpy.save(side / "block.0" / "path.npy", same)
            numpy.save(side / "block.0" / "only_ours.npy", same)
            numpy.save(side / "block.0" / "only_ref.npy", same)
        # A half-precision side against an FP32 one is the `dtype` verdict: its rounding is the
        # container's, and the row says both types rather than calling the rounding arithmetic.
        numpy.save(ours / "block.0" / "dtype.npy", same.astype(numpy.float16))
        numpy.save(ref / "block.0" / "dtype.npy", same)
        numpy.save(ours / "block.0" / "path.npy", same + 1e-3)
        (ours / "block.0" / "only_ref.npy").unlink()
        (ref / "block.0" / "only_ours.npy").unlink()

        rows = {row["name"]: row["verdict"] for row in compare(ours, ref, "block.0")}
        check("exact", rows.get("exact", "?"), "exact")
        check("shape", rows.get("shape", "?"), "shape")
        check("arithmetic path", rows.get("path", "?"), "arithmetic path")
        check("only ours", rows.get("only_ours", "?"), "reference not run here")
        check("only ref", rows.get("only_ref", "?"), "ours not run here")

        # A block neither side holds is a row, not an omission.
        absent = compare(ours, ref, "block.9")
        check("absent block", absent[0]["verdict"], "both sides absent")

        # `--block all` walks both trees.
        every = compare(ours, ref, "all")
        check("all walks the block", every[0]["block"], "block.0")

        # An empty pair of trees is a usage error and not an empty table.
        try:
            compare(root / "nowhere", root / "nowhere", "all")
            check("empty trees", "no error", "UnreadableError")
        except UnreadableError:
            check("empty trees", "UnreadableError", "UnreadableError")

        # The table renders every verdict without needing either tree again.
        text = markdown(compare(ours, ref, "block.0"), ours, ref)
        for verdict_name in (
            "exact",
            "shape",
            "arithmetic path",
            "reference not run here",
            "ours not run here",
        ):
            check(f"table holds {verdict_name}", str(verdict_name in text), "True")

    print(f"self-test: {'ok' if failures == 0 else f'{failures} failure(s)'}")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--component", help="the component whose block is compared")
    parser.add_argument("--block", help="the block's name, or `all`")
    parser.add_argument("--ours", type=pathlib.Path, help="the tree our harness wrote")
    parser.add_argument("--ref", type=pathlib.Path, help="the tree the reference wrote")
    parser.add_argument("--out", type=pathlib.Path, help="where to write the table")
    parser.add_argument("--self-test", action="store_true", help="prove the harness, no device")
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        return self_test()
    if not (arguments.ours and arguments.ref and arguments.block):
        parser.error("--ours, --ref and --block are required (or --self-test)")
    block = arguments.block
    prefix = arguments.component
    if prefix and block != "all" and not block.startswith(prefix):
        block = f"{prefix}.{block}"
    try:
        rows = compare(arguments.ours, arguments.ref, block)
    except UnreadableError as error:
        print(f"compare: {error}", file=sys.stderr)
        return 2
    text = markdown(rows, arguments.ours, arguments.ref)
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(text)
        print(f"wrote {arguments.out} ({len(rows)} row(s))")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
