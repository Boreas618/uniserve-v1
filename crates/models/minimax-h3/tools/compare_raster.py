#!/usr/bin/env python3
"""Compare two video decodes of one clip: the raster, every frame, and every seam band.

Design 10's diagnosis, never an acceptance gate. The reference's decoder is run on the candidate's
own latents (``dump_vae_blocks.py --probe decode_out``) and both sides write
``<component>/raster.h3t`` - ``[frames, rows, columns, channels]``, channels last, in the decode's
element type - and a ``case.json``. This script pairs them and reports, per row, whether the two
shapes agree, the largest absolute difference, and **the index where it is worst**. A number with
an index names a place in the picture; a summary word does not.

Why the rows are what they are:

* **the whole raster** is the row the delivery statement quotes, and it is the least informative one:
  a single element on a frame boundary can own its maximum;
* **one row per frame** is the temporal path - the upsampler emits ``latent_frames x
  temporal_compression`` frames and the clip keeps fewer, so a defect in the trim shows as one frame
  out of step rather than as a uniform residual;
* **one row per seam** is the stitch. The grid states every overlap it blends (``case.json``'s
  ``seams``, in output pixels), and a blend faded with the wrong weight moves a band of a few dozen
  pixels that a whole-raster maximum almost never names;
* **one row per channel** is the colour path, which runs once after the network and would otherwise
  hide under a per-pixel maximum.

What it refuses, because a comparison between two clips is not a page:

* two cases that disagree about the canvas, the latent extents, the tile count or the seam bands -
  the two sides decoded different requests, and a residual between them measures the difference
  between two inputs;
* a container that is not rank 4, or an element type it cannot read exactly;
* a side whose file is missing: a page exists only when both did.

It has no threshold. A row that differs is reported and nothing is gated on it.

Usage::

    .venv/bin/python crates/models/minimax-h3/tools/compare_raster.py \\
        --ref artifacts/h3-stage2-ref --got artifacts/h3-stage2 \\
        --out artifacts/diagnosis/minimax-h3/video-vae/raster.json

    .venv/bin/python crates/models/minimax-h3/tools/compare_raster.py --self-test

Both directories are the *root* holding ``video_decoder/`` (the shape both producers write), or the
component directory itself.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# The container's element codes, as `tools/dump_tables.py` writes them. Only the exact widths are
# read: a raster is the decoder's output at the decode's type, and a code outside this table would
# have to be guessed at rather than read.
ELEMENTS = {
    0: ("float64", 8),
    1: ("float32", 4),
    5: ("float16", 2),
}

COMPONENT = "video_decoder"

# What a `case.json` must state for two sides to be the same request, and the word each field is
# called in a refusal.
CASE_FIELDS = ("canvas", "latent_frames", "latent_height", "latent_width", "tiles", "seams")


def component_directory(root: Path) -> Path:
    """The directory holding `raster.h3t`: the root both producers write, or the component itself."""
    if (root / COMPONENT / "raster.h3t").exists():
        return root / COMPONENT
    return root


def read_raster(directory: Path) -> tuple[str, tuple[int, ...], np.ndarray]:
    """Read one H3D1 raster: its element name, its shape and its values in FP64."""
    path = directory / "raster.h3t"
    data = path.read_bytes()
    if len(data) < 6 or data[:4] != b"H3D1":
        raise SystemExit(f"{path}: not an H3D1 container")
    code, rank = data[4], data[5]
    if code not in ELEMENTS:
        raise SystemExit(
            f"{path}: element code {code} is not one this comparison reads exactly "
            f"({sorted(ELEMENTS)})"
        )
    name, width = ELEMENTS[code]
    if rank != 4:
        raise SystemExit(f"{path}: rank {rank}; a decode writes [frames, rows, columns, channels]")
    shape = struct.unpack_from(f"<4Q", data, 6)
    payload = data[6 + 8 * 4 :]
    count = int(np.prod(shape))
    if len(payload) != count * width:
        raise SystemExit(
            f"{path}: holds {len(payload)} bytes; {shape} at {name} needs {count * width}"
        )
    values = np.frombuffer(payload, dtype=np.dtype(name)).reshape(shape)
    return name, shape, values.astype(np.float64)


def read_case(directory: Path) -> dict:
    """One side's `case.json`, with the fields a comparison needs present by name."""
    path = directory / "case.json"
    if not path.exists():
        raise SystemExit(f"{path}: the side wrote no case; a decode writes one beside its raster")
    case = json.loads(path.read_text())
    for field in CASE_FIELDS:
        if field not in case:
            raise SystemExit(
                f"{path}: states no `{field}`; a comparison cannot tell whether the two sides "
                f"decoded the same request"
            )
    return case


def difference(
    mine: np.ndarray, theirs: np.ndarray
) -> tuple[float, tuple[int, ...] | None]:
    """The largest absolute difference and where it is, over any two arrays of one shape."""
    delta = np.abs(mine - theirs)
    if delta.size == 0:
        return 0.0, None
    flat = int(np.argmax(delta))
    index = tuple(int(part) for part in np.unravel_index(flat, delta.shape))
    value = float(delta.reshape(-1)[flat])
    # A NaN on either side is a difference this comparison cannot order; it is reported as an
    # infinity so the row cannot read as zero, and the index names the element that holds it.
    if not np.isfinite(value):
        value = float("inf")
    return value, index


def rows(ref: np.ndarray, got: np.ndarray, seams: list[dict]) -> list[dict]:
    """One row per part of the picture: the whole, each frame, each channel, each seam band."""
    table: list[dict] = []

    def add(
        name: str,
        mine: np.ndarray,
        theirs: np.ndarray,
        origin: tuple[int, int, int, int] = (0, 0, 0, 0),
        dropped_last: bool = False,
    ) -> None:
        """One row, with the worst element named **in the raster's own coordinates**.

        A band's or a frame's slice has its own indices, and a reader needs the place in the picture:
        the slice's index is placed back into the raster by its origin, at the axis the slice kept.
        """
        value, index = difference(mine, theirs)
        if index is not None:
            if len(index) == 4:
                # A band keeps all four axes and its coordinates are relative to where it starts.
                index = tuple(part + shift for part, shift in zip(index, origin))
            elif dropped_last:
                index = (*index, origin[-1])
            else:
                index = (origin[0], *index)
        table.append({"row": name, "max": value, "index": index})

    add("raster", got, ref)
    for frame in range(got.shape[0]):
        add(f"frame {frame}", got[frame], ref[frame], origin=(frame, 0, 0, 0))
    for channel in range(got.shape[3]):
        add(
            f"channel {channel}",
            got[..., channel],
            ref[..., channel],
            origin=(0, 0, 0, channel),
            dropped_last=True,
        )
    for seam in seams:
        axis, start, end = seam["axis"], int(seam["start"]), int(seam["end"])
        if axis == "rows":
            add(
                f"seam rows [{start}, {end})",
                got[:, start:end],
                ref[:, start:end],
                origin=(0, start, 0, 0),
            )
        elif axis == "columns":
            add(
                f"seam columns [{start}, {end})",
                got[:, :, start:end],
                ref[:, :, start:end],
                origin=(0, 0, start, 0),
            )
        else:
            raise SystemExit(f"a seam states axis `rows` or `columns`, not `{axis}`")
    return table


def render(label: str, table: list[dict], ref_case: dict, got_case: dict) -> str:
    """The table, as the page a reader keeps under `artifacts/`."""
    lines = [
        f"# {label}",
        "",
        "Design 10's diagnosis. No threshold: a row that differs names a place and gates nothing.",
        "",
        "| | |",
        "| --- | --- |",
        f"| reference | `{ref_case.get('latents', '?')}` latents, {ref_case['latent_dtype']} |",
        f"| candidate | `{got_case.get('latents', '?')}` latents, {got_case['latent_dtype']} |",
        f"| checkpoint | `{got_case.get('checkpoint', 'unstated')}` |",
        f"| comparison run from | `{commit()}` |",
        f"| canvas | {got_case['canvas']['height']}x{got_case['canvas']['width']} px, "
        f"{got_case['tiles']} tiles, {len(got_case['seams'])} seam bands |",
        "",
        "| row | max abs difference | worst index (frame, row, column, channel) |",
        "| --- | --- | --- |",
    ]
    for row in table:
        index = "none" if row["index"] is None else str(tuple(row["index"]))
        lines.append(f"| {row['row']} | {row['max']:.6e} | `{index}` |")
    return "\n".join(lines) + "\n"


def commit() -> str:
    """The tree this comparison was run from, so a number can be dated.

    Best effort by design: a checkout without git still produces the numbers, and the report says
    `unknown` rather than leaving the field out - a file that omits it reads as one that forgot.
    """
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git is not a reason to lose the numbers
        return "unknown"


def compare(ref_root: Path, got_root: Path) -> tuple[str, dict]:
    """Pair two decodes: refuse different requests, then report every row."""
    ref_directory, got_directory = component_directory(ref_root), component_directory(got_root)
    ref_case, got_case = read_case(ref_directory), read_case(got_directory)
    for field in CASE_FIELDS:
        if ref_case[field] != got_case[field]:
            raise SystemExit(
                f"the two sides disagree about `{field}`: {ref_case[field]!r} against "
                f"{got_case[field]!r}. They decoded different requests, and a residual between "
                f"them would measure the difference between two inputs"
            )
    ref_dtype, ref_shape, ref = read_raster(ref_directory)
    got_dtype, got_shape, got = read_raster(got_directory)
    if ref_shape != got_shape:
        raise SystemExit(f"the shapes differ: {ref_shape} against {got_shape}")
    table = rows(ref, got, got_case["seams"])
    report = {
        "commit": commit(),
        "checkpoint": got_case.get("checkpoint", "unstated"),
        "reference": {"element": ref_dtype, "shape": list(ref_shape)},
        "candidate": {"element": got_dtype, "shape": list(got_shape)},
        "case": got_case,
        "rows": table,
    }
    label = (
        f"video decoder: {got_shape[0]} frames of {got_shape[1]}x{got_shape[2]}x{got_shape[3]}, "
        f"{got_dtype}"
    )
    return render(label, table, ref_case, got_case), report


def write(directory: Path, contents: tuple[str, dict]) -> None:
    """Write the table and its JSON block under one directory, named for the component."""
    directory.mkdir(parents=True, exist_ok=True)
    text, report = contents
    (directory / f"{COMPONENT}.md").write_text(text)
    (directory / f"{COMPONENT}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(text, end="")


def self_test() -> int:
    """Prove the comparison on arrays whose answer is known: nothing here needs a device."""
    frames, height, width, channels = 2, 8, 8, 3
    base = np.zeros((frames, height, width, channels), dtype=np.float64)
    base[0, 2, 3, 1] = 1.0
    # Two differences: one inside a seam band, and one larger outside every band.
    inside = base.copy()
    inside[1, 4, 2, 0] = 0.25
    outside = inside.copy()
    outside[0, 6, 6, 2] = 0.5
    outside[1, 1, 1, 1] = 0.125
    seams = [{"axis": "rows", "start": 4, "end": 6}, {"axis": "columns", "start": 2, "end": 3}]
    table = {row["row"]: row for row in rows(base, outside, seams)}
    failures = []
    if table["raster"]["index"] != (0, 6, 6, 2) or abs(table["raster"]["max"] - 0.5) > 0:
        failures.append(f"the whole raster should name (0, 6, 6, 2): {table['raster']}")
    if table["seam rows [4, 6)"]["index"] != (1, 4, 2, 0):
        failures.append(f"a row band should name (1, 4, 2, 0): {table['seam rows [4, 6)']}")
    if table["seam columns [2, 3)"]["index"] != (1, 4, 2, 0):
        failures.append(f"a column band should name (1, 4, 2, 0): {table['seam columns [2, 3)']}")
    if table["frame 0"]["index"] != (0, 6, 6, 2):
        failures.append(f"a frame row is named in the raster's coordinates: {table['frame 0']}")
    if table["channel 1"]["index"] != (1, 1, 1, 1):
        failures.append(f"a channel row keeps the frame: {table['channel 1']}")
    if table["raster"]["max"] == 0.0:
        failures.append("a difference of 0.5 was reported as none")
    same = {row["row"]: row for row in rows(base, base, seams)}
    if any(row["max"] != 0.0 for row in same.values()):
        failures.append("two identical rasters are not all zeros")
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"self-test: {len(failures)} failure(s)")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", type=Path, help="the reference's tree")
    parser.add_argument("--got", type=Path, help="the candidate's tree")
    parser.add_argument("--out", type=Path, help="the directory the table and JSON go to")
    parser.add_argument("--self-test", action="store_true", help="prove the comparison only")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    if not (arguments.ref and arguments.got):
        parser.error("--ref and --got name the two trees (or pass --self-test)")
    contents = compare(arguments.ref, arguments.got)
    if arguments.out:
        write(arguments.out, contents)
    else:
        print(contents[0], end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
