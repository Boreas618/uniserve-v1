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

# The container's dtype codes: 0 f64, 1 f32, 2 i64, 3 i32, 4 bf16, 5 f16, 6 i16
# (notes/h3/numeric-diagnosis.md). The comparators read a half-precision payload as the dtype its
# producer computed in: a BF16 activation widened before writing is twice its width on disk and
# changes none of the numbers, so the container keeps it narrow.
CODES = {
    np.dtype(np.float64): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.int64): 2,
    np.dtype(np.int32): 3,
    np.dtype(np.float16): 5,
    np.dtype(np.int16): 6,
}

#: The codes whose payload is half precision, and the torch dtype each stands for.
HALVES = {4: torch.bfloat16, 5: torch.float16}

# The condition roles by the name a case writes, resolved where the reference is imported: this
# module's container (`dump`, `load`) must be usable without the reference's compiled extensions, so
# nothing of `uniserve` is imported at module scope.
ROLES = {
    "first_frame": "FIRST_FRAME",
    "last_frame": "LAST_FRAME",
    "reference": "REFERENCE",
}


def dump(out: Path, name: str, values) -> None:
    """Write one tensor twice: as H3D1 for the Rust comparator, and as `.npy` for the Python one.

    `examples/compare_blocks.rs` reads the self-describing H3D1 file; a `.npy`-per-tensor tree is
    what `tools/diagnosis/compare.py` reads. Both are written from the same numbers, so the two
    comparators can never disagree about what a dump holds. NumPy has no BF16 dtype, so a BF16
    tensor is written narrow in the container and widened to FP32 in the `.npy`; doubling a BF16
    value is exact, so the widening changes no number either comparator reads.
    """
    saved = None
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu().contiguous()
        if tensor.dtype is torch.bfloat16:
            code = 4
            array = tensor.view(torch.uint16).numpy()
            saved = tensor.float().numpy()
        else:
            array = tensor.numpy()
            code = CODES[array.dtype]
            saved = array
    else:
        array = np.ascontiguousarray(np.asarray(values))
        code = CODES[array.dtype]
        saved = array
    array = np.ascontiguousarray(array)
    header = b"H3D1" + bytes([code, array.ndim])
    header += struct.pack(f"<{array.ndim}Q", *array.shape)
    (out / f"{name}.h3t").write_bytes(header + array.tobytes())
    np.save(out / f"{name}.npy", saved)


def load(path: Path) -> torch.Tensor:
    """Read one H3D1 tensor, in the dtype its producer computed it in.

    The inverse of `dump`, and the reason a dump can be an input: the reference producer of the
    denoiser stack runs on the bytes the candidate wrote rather than on a second computation of
    them, so a comparison is about arithmetic and not about who drew the noise.
    """
    data = path.read_bytes()
    if len(data) < 6 or data[:4] != b"H3D1":
        raise ValueError(f"{path}: not an H3D1 tensor")
    code, rank = data[4], data[5]
    header = 6 + 8 * rank
    if len(data) < header:
        raise ValueError(f"{path}: a truncated header")
    shape = struct.unpack_from(f"<{rank}Q", data, 6)
    dtype = {
        0: torch.float64,
        1: torch.float32,
        2: torch.int64,
        3: torch.int32,
        6: torch.int16,
    }.get(code)
    if dtype is None:
        dtype = HALVES.get(code)
        if dtype is None:
            raise ValueError(f"{path}: unknown dtype code {code}")
    payload = data[header:]
    count = 1
    for dimension in shape:
        count *= dimension
    if len(payload) != count * torch.empty((), dtype=dtype).element_size():
        raise ValueError(f"{path}: a payload that does not match its shape")
    values = torch.frombuffer(bytearray(payload), dtype=dtype)
    return values.reshape(shape)


def canvas_of(size: dict):
    from uniserve.media import image

    return image.Config(height=size["height"], width=size["width"])


def condition_of(case: dict):
    from uniserve.media import video
    from uniserve.models import Condition, ConditionRole

    pixels = case.get("canvas")
    frames = case.get("frames", 1)
    return Condition(
        role=ConditionRole[ROLES[case["role"]]],
        video=None
        if pixels is None
        else video.Config(frame=canvas_of(pixels), num_frames=frames),
        audio_samples=case.get("audio_samples", 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", required=True, type=Path, help="the case description"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="the directory to write"
    )
    arguments = parser.parse_args()

    case = json.loads(arguments.case.read_text())
    # The encoder's clip rule is the condition's latent count, and it is the checkpoint's, not the
    # case's: the comparator reads these three numbers, so a case that has drifted from the
    # reference's `video_vae` section is refused here rather than compared against.
    if "clip" not in case:
        raise SystemExit(
            f"{arguments.case}: no clip; the case must state the encoder's rule"
        )
    arguments.out.mkdir(parents=True, exist_ok=True)
    canvas = canvas_of(case["canvas"])
    conditions = tuple(
        condition_of(condition) for condition in case.get("conditions", ())
    )
    from uniserve.models.minimax_h3 import packing

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
        dump(
            arguments.out,
            "segment.variable_block_sizes",
            tables.variable_block_sizes,
        )
        dump(arguments.out, "segment.tile_segments", tables.tile_segments)
        dump(arguments.out, "segment.segment_starts", tables.segment_starts)
        dump(arguments.out, "segment.segment_keep", tables.segment_keep)
        dump(arguments.out, "segment.video_indices", plan.video_indices)
        dump(
            arguments.out,
            "segment.video_raster_indices",
            plan.video_raster_indices,
        )
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
