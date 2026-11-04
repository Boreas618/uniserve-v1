#!/usr/bin/env python3
"""Dump the layout tables of one H3 request from today's implementation.

The Rust side of the comparison is `crates/models/minimax-h3/examples/compare_blocks.rs`; the case
format, the tensor container and how to read the report are in `notes/h3/numeric-diagnosis.md`.

Run it from the repository root, with the repository's environment:

    .venv/bin/python crates/models/minimax-h3/tools/dump_tables.py --case <case.json> --out <dir>

It needs no checkpoint and no device: every table it writes is host arithmetic over the request's
frame count, canvas and capacities.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from uniserve.media import image, video
from uniserve.models import Condition, ConditionRole
from uniserve.models.minimax_h3 import packing

# The container's dtype codes: 0 f64, 1 f32, 2 i64, 3 i32 (notes/h3/numeric-diagnosis.md).
CODES = {
    np.dtype(np.float64): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.int64): 2,
    np.dtype(np.int32): 3,
}

ROLES = {
    "first_frame": ConditionRole.FIRST_FRAME,
    "last_frame": ConditionRole.LAST_FRAME,
    "reference": ConditionRole.REFERENCE,
}


def dump(out: Path, name: str, values) -> None:
    """Write one tensor in the H3D1 container."""
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    array = np.ascontiguousarray(array)
    code = CODES[array.dtype]
    header = b"H3D1" + bytes([code, array.ndim])
    header += struct.pack(f"<{array.ndim}Q", *array.shape)
    (out / f"{name}.h3t").write_bytes(header + array.tobytes())


def canvas_of(size: dict) -> image.Config:
    return image.Config(height=size["height"], width=size["width"])


def condition_of(case: dict) -> Condition:
    pixels = case.get("canvas")
    frames = case.get("frames", 1)
    return Condition(
        role=ROLES[case["role"]],
        video=None
        if pixels is None
        else video.Config(frame=canvas_of(pixels), num_frames=frames),
        audio_samples=case.get("audio_samples", 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path, help="the case description")
    parser.add_argument("--out", required=True, type=Path, help="the directory to write")
    arguments = parser.parse_args()

    case = json.loads(arguments.case.read_text())
    arguments.out.mkdir(parents=True, exist_ok=True)
    canvas = canvas_of(case["canvas"])
    conditions = tuple(condition_of(condition) for condition in case.get("conditions", ()))
    segments = packing.condition_segments(conditions, canvas)
    spans = tuple(tuple(span) for span in case.get("vision_spans", ()))
    frames = case["num_frames"]

    attention = case["attention"]
    if attention["kind"] == "dense":
        plan = packing.dense_packing(
            num_frames=frames,
            canvas=canvas,
            text_rows=case["text_capacity"],
            condition_rows=case["condition_capacity"],
            token_multiple=case["token_multiple"],
        )
        tables = packing.dense_tables(
            plan,
            num_frames=frames,
            canvas=canvas,
            num_text_tokens=case["num_text_tokens"],
            segments=segments,
            vision_spans=spans,
        )
        dump(arguments.out, "dense.position_ids", tables.position_ids)
        dump(arguments.out, "dense.token_tags", tables.token_tags)
        dump(arguments.out, "dense.groups", tables.groups)
        dump(arguments.out, "dense.prefix_index", tables.prefix_index)
        for name, value in (
            ("dense.video_rows", plan.video_rows),
            ("dense.audio_rows", plan.audio_rows),
            ("dense.prefix_start", plan.prefix_start),
            ("dense.zero_row", plan.zero_row),
            ("dense.padded_tokens", plan.padded_tokens),
            ("dense.used", tables.used),
        ):
            dump(arguments.out, name, np.asarray([value], dtype=np.int64))
    elif attention["kind"] == "sparse":
        plan = packing.segment_packing(
            num_frames=frames,
            canvas=canvas,
            text_rows=case["text_capacity"],
            condition_rows=case["condition_capacity"],
            tile=attention["tile"],
            token_multiple=case["token_multiple"],
        )
        tables = packing.segment_tables(
            plan,
            num_frames=frames,
            canvas=canvas,
            num_text_tokens=case["num_text_tokens"],
            segments=segments,
            vision_spans=spans,
            sparsity=attention["sparsity"],
            reference_keep=attention.get("reference_keep"),
        )
        dump(arguments.out, "segment.position_ids", tables.position_ids)
        dump(arguments.out, "segment.token_tags", tables.token_tags)
        dump(arguments.out, "segment.groups", tables.groups)
        dump(arguments.out, "segment.prefix_index", tables.prefix_index)
        dump(arguments.out, "segment.variable_block_sizes", tables.variable_block_sizes)
        dump(arguments.out, "segment.tile_segments", tables.tile_segments)
        dump(arguments.out, "segment.segment_starts", tables.segment_starts)
        dump(arguments.out, "segment.segment_keep", tables.segment_keep)
        dump(arguments.out, "segment.video_indices", plan.video_indices)
        dump(arguments.out, "segment.video_raster_indices", plan.video_raster_indices)
        for name, value in (
            ("segment.audio_rows", plan.audio_rows),
            ("segment.audio_start", plan.audio_start),
            ("segment.video_tiles", plan.video_tiles),
            ("segment.video_start", plan.video_start),
            ("segment.padded_tokens", plan.padded_tokens),
        ):
            dump(arguments.out, name, np.asarray([value], dtype=np.int64))
    else:
        raise SystemExit(f"unknown attention kind {attention['kind']!r}")

    # `case.json` is the dumper's own copy, so a comparison directory is self-contained.
    (arguments.out / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    written = len(list(arguments.out.glob("*.h3t")))
    print(f"{written} tensors and case.json in {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
