"""Compare the sparse layout's tile and segment tables with the reference's own.

The sparse attention's arithmetic runs on a device, but which key tiles a query tile reads is
decided by host tables: ``variable_block_sizes`` (valid rows of each tile), ``tile_segments``
(which segment a tile belongs to, ``-1`` for dense), ``segment_starts`` (where each segment's tiles
begin when the keys are ordered by segment) and ``segment_keep`` (how many tiles of each segment a
video query keeps). A wrong entry changes the model silently: a kept tile is a column a query reads,
so a table that keeps one tile too many produces a plausible page rather than a failure.

``crates/models/minimax-h3/examples/vsa_tables.rs`` writes our tables for a case; this compares them
with the reference's ``packing.segment_tables`` for the same case, entry by entry.

    cargo run -q -p minimax-h3 --example vsa_tables -- --out artifacts/h3-vsa
    .venv/bin/python crates/models/minimax-h3/tools/compare_vsa_tables.py \
        --case artifacts/h3-vsa

Exit codes: 0 equal, 1 differ, 2 a case the two sides cannot be compared at.

The second claim this makes needs no case at all and is checked by ``--self-test``: at
``sparsity = 0`` every segment keeps all of its tiles, so the reference's own selector keeps every
live key tile for every live query - the sparse mask equals the dense one. That is the strongest
statement available without a device, because the dense path is the one with instruments behind it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import types

FIELDS = (
    "variable_block_sizes",
    "tile_segments",
    "segment_starts",
    "segment_keep",
)
SCALARS = ("padded_tokens", "video_start", "video_tiles")


def install_kernel_stub() -> bool:
    """Let the reference's CPU table builders run where its kernel package is not installed.

    ``uniserve.models.minimax_h3.packing`` imports through the library's package tree, and the
    compiled ``uniserve_kernels`` package is not present on every host that has the reference's
    Python. The tables are CPU arithmetic that never calls a kernel, so a stub that refuses when
    called keeps the reference's own code while removing an import a host comparison does not need.
    Returns whether the stub was installed, so the verdict can say which branch produced it.
    """
    try:
        importlib.import_module("uniserve_kernels.triton")

        return False
    except ModuleNotFoundError:
        pass
    kernel = types.ModuleType("uniserve_kernels.triton")

    def require_kernel(name, unsupported, **kwargs):
        raise ComparisonError(
            f"the reference asked for the {name} kernel, which this host does not have;"
            f" the comparison needs its CPU path"
        )

    kernel.require_kernel = require_kernel
    package = types.ModuleType("uniserve_kernels")
    package.triton = kernel
    sys.modules["uniserve_kernels"] = package
    sys.modules["uniserve_kernels.triton"] = kernel
    return True


def reference_modules():
    """The reference's image config and its segment packing, loaded where they can be.

    `uniserve.models.minimax_h3.packing` cannot be imported as a package member on a host without
    the compiled worker library - the package's `__init__` reads the library's task table - so the
    module is loaded by path. Its own imports are the light ones (`uniserve.media`,
    `uniserve.models`), which resolve normally.
    """
    import importlib.util

    root = pathlib.Path().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from uniserve.media import image

    spec = importlib.util.spec_from_file_location(
        "reference_packing", root / "uniserve/models/minimax_h3/packing.py"
    )
    packing = importlib.util.module_from_spec(spec)
    sys.modules["reference_packing"] = packing
    spec.loader.exec_module(packing)
    return image, packing


class ComparisonError(Exception):
    """A comparison the two sides cannot be run at, rather than a difference between them."""


def read(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def reference(case: dict) -> dict:
    """The reference's own tables for the case our example ran."""
    image, packing = reference_modules()

    canvas = image.Config(height=case["canvas"][0], width=case["canvas"][1])
    kept = packing.segment_packing(
        num_frames=case["frames"],
        canvas=canvas,
        text_rows=case["text_rows"],
        condition_rows=case["condition_rows"],
        tile=case["tile"],
        token_multiple=case["token_multiple"],
    )
    tables = packing.segment_tables(
        kept,
        num_frames=case["frames"],
        canvas=canvas,
        num_text_tokens=case["text_tokens"],
        segments=(),
        vision_spans=(),
        sparsity=case["sparsity"],
        reference_keep=case["reference_keep"],
    )
    return {
        "variable_block_sizes": [
            int(value) for value in tables.variable_block_sizes
        ],
        "tile_segments": [int(value) for value in tables.tile_segments],
        "segment_starts": [int(value) for value in tables.segment_starts],
        "segment_keep": [int(value) for value in tables.segment_keep],
        "padded_tokens": int(kept.padded_tokens),
        "video_start": int(kept.video_start),
        "video_tiles": int(kept.video_tiles),
    }


def first_difference(
    name: str, ours: list, theirs: list, limit: int
) -> str | None:
    """Where two tables first disagree, or None when they agree over the entries that are read."""
    if len(ours) != len(theirs):
        return f"{name}: ours holds {len(ours)} entries, the reference {len(theirs)}"
    for index in range(min(len(ours), limit)):
        if ours[index] != theirs[index]:
            return f"{name}: element {index} is {ours[index]} where the reference has {theirs[index]}"
    return None


def compared(
    case: dict, tables: dict, expected: dict
) -> tuple[list[str], list[str]]:
    """The equal lines and the differences, each named by table and element index."""
    equal: list[str] = []
    differ: list[str] = []
    for name in SCALARS:
        if int(tables[name]) != int(expected[name]):
            differ.append(
                f"{name}: ours {tables[name]}, the reference {expected[name]}"
            )
        else:
            equal.append(f"{name} {tables[name]}")
    # `segment_keep` is over-allocated to the tile count on both sides, and its entries past the
    # last segment are unread on both: only the entries each side fills are compared.
    segments = int(tables["segments"])
    if segments <= 0:
        differ.append("segments: the case holds no video segment to compare")
    limits = {
        "variable_block_sizes": len(expected["variable_block_sizes"]),
        "tile_segments": len(expected["tile_segments"]),
        "segment_starts": segments,
        "segment_keep": segments,
    }
    for name in FIELDS:
        problem = first_difference(
            name, tables[name], expected[name], limits[name]
        )
        if problem is None:
            equal.append(
                f"{name} {limits[name]} of {len(tables[name])} entries"
            )
        else:
            differ.append(problem)
    return equal, differ


def dense_equivalence(case: dict, tables: dict) -> str | None:
    """Whether our tables, through the reference's selector, keep every live key tile.

    Only meaningful at ``sparsity = 0``: then no tile is evicted, the compression contributes
    nothing, and the sparse mask is the dense one. Running the *reference's* selector over *our*
    tables is what makes this a statement about our tables rather than about the reference's.
    """
    if case["sparsity"] != 0:
        return None
    import torch

    root = str(pathlib.Path().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from uniserve.nn.attention.vsa.segments import Segments, select

    tiles = int(tables["padded_tokens"]) // int(case["tile"])
    sizes = torch.tensor(tables["variable_block_sizes"], dtype=torch.int32)
    ours = Segments(
        int(case["tile"]),
        int(tables["padded_tokens"]),
        sizes,
        torch.tensor(tables["tile_segments"], dtype=torch.int32),
        torch.tensor(tables["segment_starts"], dtype=torch.int32),
        torch.tensor(tables["segment_keep"], dtype=torch.int32),
    )
    _, counts = select(
        torch.zeros((1, tiles, tiles), dtype=torch.float32), ours
    )
    live = int((sizes > 0).sum())
    observed = sorted({int(count) for count in counts[0][sizes > 0]})
    if observed != [live]:
        return (
            f"dense equivalence: a live query keeps {observed} of {live} live key tiles at "
            "sparsity 0, not all of them"
        )
    return f"dense equivalence: every live query keeps all {live} live key tiles at sparsity 0"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=pathlib.Path, help="the directory our example wrote"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the comparator's own falsifications and exit",
    )
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        return self_test()
    if arguments.case is None:
        raise SystemExit("--case or --self-test is required")

    case = read(arguments.case / "case.json")
    tables = read(arguments.case / "tables.json")
    stub = install_kernel_stub()
    try:
        expected = reference(case)
    except ComparisonError as problem:
        raise SystemExit(f"refused: {problem}") from problem
    equal, differ = compared(case, tables, expected)

    print(
        f"case    {case['frames']} frames, canvas {case['canvas']}, "
        f"tile {case['tile']}, sparsity {case['sparsity']}, "
        f"reference keep {case['reference_keep']}"
    )
    for line in equal:
        print(f"equal   {line}")
    for line in differ:
        print(f"differ  {line}")
    if stub:
        print(
            "note    the kernel package is absent on this host; a refusing stub was installed"
        )
    if not differ:
        dense = dense_equivalence(case, tables)
        if dense is not None:
            if dense.startswith("dense equivalence: a live"):
                differ.append(dense)
                print(f"differ  {dense}")
            else:
                print(f"equal   {dense}")
    print("verdict " + ("equal" if not differ else "differ"))
    return 0 if not differ else 1


def self_test() -> int:
    """The comparator's falsifications, and the sparsity-zero equivalence.

    A comparator that has only ever been green is not evidence: each case below feeds it a table it
    must reject, and the last one checks the claim that makes the sparse path comparable at all.
    """
    import torch

    install_kernel_stub()

    case = {
        "frames": 22,
        "canvas": [256, 256],
        "text_rows": 128,
        "condition_rows": 192,
        "text_tokens": 128,
        "tile": 64,
        "sparsity": 0.8,
        "reference_keep": None,
        "token_multiple": 64,
        "padded_tokens": 0,
        "video_tiles": 0,
    }
    expected = reference(case)
    # The reference load put the repository root on the path; the selector imports through it.
    from uniserve.nn.attention.vsa.segments import Segments, select

    ours = {
        name: list(values) if isinstance(values, list) else values
        for name, values in expected.items()
    }
    ours["segments"] = max(ours["tile_segments"]) + 1
    equal, differ = compared(case, ours, expected)
    if differ:
        print(f"self-test 1: our own table was rejected: {differ[0]}")
        return 1
    print(f"self-test 1: the reference's own table passes ({len(equal)} lines)")

    broken = {
        name: list(values) if isinstance(values, list) else values
        for name, values in ours.items()
    }
    broken["segment_keep"][0] += 1
    _, differ = compared(case, broken, expected)
    if not differ:
        print("self-test 2: a raised keep entry was accepted")
        return 1
    print(f"self-test 2: a raised keep entry is rejected - {differ[0]}")

    swapped = {
        name: list(values) if isinstance(values, list) else values
        for name, values in ours.items()
    }
    sizes = swapped["variable_block_sizes"]
    sizes[0], sizes[-1] = sizes[-1], sizes[0]
    if sizes[0] == sizes[-1]:
        print(
            "self-test 3: the two tiles hold equal sizes, nothing was falsified"
        )
        return 1
    _, differ = compared(case, swapped, expected)
    if not differ:
        print("self-test 3: a swapped tile size was accepted")
        return 1
    print(f"self-test 3: a swapped tile size is rejected - {differ[0]}")

    dense = dict(case, sparsity=0.0)
    dense_expected = reference(dense)
    segments = max(dense_expected["tile_segments"]) + 1
    keep = dense_expected["segment_keep"][:segments]
    counts = []
    for segment in range(segments):
        counts.append(
            sum(
                1
                for entry in dense_expected["tile_segments"]
                if entry == segment
            )
        )
    if keep != counts:
        print(f"self-test 4: at sparsity 0 the keeps are {keep}, not {counts}")
        return 1
    print(
        f"self-test 4: at sparsity 0 every segment keeps all of its tiles {counts}"
    )

    # The mask itself, through the reference's own selector: at sparsity 0 a live query keeps every
    # live key tile, which is the dense mask.
    tiles = dense_expected["padded_tokens"] // dense["tile"]
    sizes = torch.tensor(
        dense_expected["variable_block_sizes"], dtype=torch.int64
    )
    tables = Segments(
        dense["tile"],
        dense_expected["padded_tokens"],
        torch.tensor(dense_expected["variable_block_sizes"], dtype=torch.int32),
        torch.tensor(dense_expected["tile_segments"], dtype=torch.int32),
        torch.tensor(dense_expected["segment_starts"], dtype=torch.int32),
        torch.tensor(dense_expected["segment_keep"], dtype=torch.int32),
    )
    scores = torch.zeros((1, tiles, tiles), dtype=torch.float32)
    _, kept_counts = select(scores, tables)
    live = int((sizes > 0).sum())
    # A live query is a tile with valid rows; an empty tile keeps nothing, which is correct and
    # would otherwise read as a query that kept too few.
    observed = sorted({int(count) for count in kept_counts[0][sizes > 0]})
    if observed != [live]:
        print(
            f"self-test 5: at sparsity 0 a live query keeps {observed} of {live} live "
            "key tiles, not all of them"
        )
        return 1
    print(
        f"self-test 5: at sparsity 0 every live query keeps all {live} live key tiles "
        "through the reference's own selector - the dense mask"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
