#!/usr/bin/env python3
"""Read two trees of `.npy` tensors and state, per tensor, a number with an index.

This is the comparison half of design 10's block-by-block diagnosis: the candidate's harness writes
`<block>/<name>.npy`, a `tools/dump_*.py` writes the reference's under the same names in its own tree,
and this reads the pair. It is a **diagnosis, never an acceptance gate**: it always exits 0, it never
compares a missing file into a failure, and its job is to name the block that differs rather than to
decide whether a build is good.

    python3 crates/models/minimax-h3/tools/compare_diagnosis.py \\
        --candidate artifacts/diagnosis/minimax-h3/fasth3/conditioned \\
        --reference artifacts/diagnosis/minimax-h3/fasth3/conditioned-ref

Every row carries: the block and file, whether the two shapes agree, the maximum absolute difference
and **the index where it is worst** - the index is the part that localises a defect, because a
maximum alone says "somewhere" and a mean says nothing about where. A pair whose dtypes differ is
compared after the candidate's is cast to the reference's, and the row says which was cast, so a
type disagreement is visible rather than silently converted away.

The provenance page the candidate's harness writes beside its tensors (`<block>/case.json`) is read
and printed before the rows: a number with no commit beside it cannot be dated, and a comparison
between two runs of unknown commits is two numbers without a subject.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The tensors a conditioned encode produces on the candidate's side, in the order a reader should
# meet them: the input the encoder read, then the request's draw, then what the conditioner wrote.
PREFERENCE = ("pixels.npy", "condition_noise.npy", "condition_rows.npy")


def pairs(candidate: Path, reference: Path) -> list[tuple[str, Path | None, Path | None]]:
    """Every tensor either tree holds, as (relative name, candidate path, reference path)."""
    found: dict[str, tuple[Path | None, Path | None]] = {}

    def add(root: Path, side: int) -> None:
        if not root.is_dir():
            return
        for path in sorted(root.rglob("*.npy")):
            name = path.relative_to(root).as_posix()
            pair = found.setdefault(name, (None, None))
            found[name] = (path, pair[1]) if side == 0 else (pair[0], path)

    add(candidate, 0)
    add(reference, 1)
    def order(name: str) -> tuple[int, str]:
        # A tree keeps its block directories, so the preference is on the file's own name: the reader
        # meets pixels, then the draw, then what the conditioner wrote, in every block.
        leaf = Path(name).name
        return (PREFERENCE.index(leaf) if leaf in PREFERENCE else len(PREFERENCE), name)

    return [(name, found[name][0], found[name][1]) for name in sorted(found, key=order)]


def pages(candidate: Path, reference: Path) -> None:
    """Print each tree's provenance page, when it wrote one."""
    for label, root in (("candidate", candidate), ("reference", reference)):
        for page in sorted(root.rglob("case.json")) if root.is_dir() else []:
            try:
                fields = json.loads(page.read_text())
            except (OSError, ValueError) as error:
                print(f"{label} page {page.name}: unreadable ({error})")
                continue
            commit = fields.get("commit", "unstated")
            checkpoint = fields.get("checkpoint", "unstated")
            print(f"{label} page {page.parent.name}/{page.name}: commit {commit}, checkpoint {checkpoint}")


def worst(values: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """The largest absolute difference and the index where it is, NaN treated as a difference."""
    if values.size == 0:
        return 0.0, ()
    absolute = np.abs(values)
    flat = int(np.nanargmax(absolute)) if np.isfinite(absolute).any() else 0
    index = np.unravel_index(flat, values.shape)
    worst_value = float(absolute.reshape(-1)[flat])
    if not np.isfinite(worst_value):
        return float("inf"), index
    return worst_value, index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    options = parser.parse_args(argv)

    pages(options.candidate, options.reference)
    rows = pairs(options.candidate, options.reference)
    if not rows:
        print(f"no .npy under {options.candidate} or {options.reference}: nothing to compare")
        return 0

    print()
    print("| tensor | shapes | max abs diff | index of worst | compared |")
    print("| --- | --- | --- | --- | --- |")
    for name, candidate, reference in rows:
        if candidate is None:
            print(f"| `{name}` | - | - | - | candidate missing |")
            continue
        if reference is None:
            print(f"| `{name}` | - | - | - | reference missing |")
            continue
        try:
            mine = np.load(candidate)
            theirs = np.load(reference)
        except (OSError, ValueError) as error:
            print(f"| `{name}` | - | - | - | unreadable: {error} |")
            continue
        shapes = f"{mine.shape} vs {theirs.shape}"
        if mine.shape != theirs.shape:
            print(f"| `{name}` | {shapes} | - | - | shape disagreement |")
            continue
        cast = ""
        if mine.dtype != theirs.dtype:
            cast = f", candidate {mine.dtype} cast to {theirs.dtype}"
            mine = mine.astype(theirs.dtype)
        difference, index = worst(mine.astype(np.float64) - theirs.astype(np.float64))
        print(
            f"| `{name}` | {shapes} | {difference:.6e} | {tuple(int(part) for part in index)} "
            f"| {mine.size} values{cast} |"
        )

    print()
    print(
        "This is a diagnosis, not a gate: a row that differs names a block, and no row here fails a "
        "build or blocks a merge (specs/worker.md 10)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
