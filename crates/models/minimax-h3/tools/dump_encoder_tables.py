#!/usr/bin/env python3
"""Dump the text encoder's host tables from today's implementation (specs/worker.md 10).

The reference half of ``examples/compare_encoder_tables.rs``: this script reads the same
``case.json``, builds the same seven tables from the reference's own code
(``uniserve.models.qwen3_vl.positions.rope_index``, ``uniserve.nn.rope.RotaryEmbedding``,
``uniserve.nn.vision.position``), and writes one ``.npy`` per table under
``<out>/text_encoder/`` so that ``tools/diagnosis/compare.py`` can report the difference.

The names are the ones the model declares in ``Encoder::state_tables``:
``prompt_positions``, ``vision_destinations``, ``vision_position_index``,
``vision_position_weight``, ``vision_cosine``, ``vision_sine``, ``vision_lengths``. The tables are
concatenated over the prompt's images in prompt order, exactly as the executor uploads them, and a
prompt with no images has the first two only.

It needs no checkpoint and no GPU: every value is host arithmetic, which is the half a structural
mistake (a wrong row, a wrong order, a wrong rotation axis) shows in first. This is a
**diagnosis, never an acceptance gate** (``specs/worker.md`` 10).

Usage::

    .venv/bin/python crates/models/minimax-h3/tools/dump_encoder_tables.py \\
        --case case.json --out artifacts/diagnosis/<run>/ref \\
        [--text-encoder /opt/uv/models/MiniMax-H3/text_encoder]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from uniserve.models.qwen3_vl.positions import rope_index
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.vision.position import PositionEmbedding, merged_grid_coordinates

DEFAULT_CHECKPOINT = pathlib.Path("/opt/uv/models/MiniMax-H3/text_encoder")

#: The four learned-table rows each output point blends, corner-aligned bilinear.
TAPS = 4


def config_of(root: pathlib.Path) -> dict:
    """The checkpoint's own numbers, as the Rust example reads them."""
    metadata = json.loads((root / "config.json").read_text())
    text = metadata.get("text_config", metadata)
    vision = metadata["vision_config"]
    scaling = text.get("rope_scaling") or {}
    # The placeholder ids sit beside `text_config` in the checkpoint's top level.
    return {
        "image_token_id": int(metadata.get("image_token_id", text.get("image_token_id"))),
        "video_token_id": int(metadata.get("video_token_id", text.get("video_token_id"))),
        "head_dim": int(text["head_dim"]),
        "rope_theta": float(text.get("rope_theta", 1e4)),
        "merge": int(vision["spatial_merge_size"]),
        "position_grid": _square_root(int(vision["num_position_embeddings"])),
        "vision_head_dim": int(vision["hidden_size"]) // int(vision["num_heads"]),
        "vision_theta": 10_000.0,
        "sections": tuple(int(width) for width in scaling.get("mrope_section", ())),
        "interleaved": bool(scaling.get("mrope_interleaved", False)),
    }


def _square_root(entries: int) -> int:
    """The side of the square learned position table."""
    side = int(round(entries**0.5))
    if side * side != entries:
        raise ValueError(f"the position table's {entries} entries are not square")
    return side


def blocks_of(case: dict, config: dict) -> list[tuple[int, int, int]]:
    """The prompt's blocks, in prompt order: one per tower call.

    An image block is one placeholder run under its whole grid; a video block is one run per
    temporal patch and one tower call covering all of them
    (``uniserve/models/qwen3_vl/positions.py:31-52``, ``vision.py::VisionTower.forward``).
    """
    merge = config["merge"]
    images = list(case.get("image_grids", []))
    videos = [tuple(grid) for grid in case.get("video_grids", [])]
    blocks: list[tuple[int, int, int]] = []
    image_next = 0
    video_next = 0
    remaining = videos[0][0] if videos else 0
    row = 0
    ids = case["ids"]
    while row < len(ids):
        token = ids[row]
        if token not in (config["image_token_id"], config["video_token_id"]):
            row += 1
            continue
        start = row
        while row < len(ids) and ids[row] == token:
            row += 1
        run = row - start
        if token == config["image_token_id"]:
            if image_next >= len(images):
                raise ValueError(f"an image run of {run} placeholders with no grid left")
            grid = tuple(images[image_next])
            image_next += 1
            if run != _merged_rows(grid, merge):
                raise ValueError(f"an image run of {run} placeholders for a {grid} grid")
            blocks.append(grid)
            continue
        if video_next >= len(videos):
            raise ValueError(f"a video run of {run} placeholders with no grid left")
        grid = videos[video_next]
        if run != _merged_rows((1, grid[1], grid[2]), merge):
            raise ValueError(f"a video run of {run} placeholders for one step of a {grid} grid")
        remaining -= 1
        if remaining == 0:
            blocks.append(grid)
            video_next += 1
            remaining = videos[video_next][0] if video_next < len(videos) else 0
    if image_next != len(images):
        raise ValueError(f"the image grid {images[image_next]} has no placeholder run")
    if video_next != len(videos):
        raise ValueError(f"the video grid {videos[video_next]} has fewer runs than its steps")
    return blocks


def _merged_rows(grid: tuple[int, int, int], merge: int) -> int:
    """A grid's merged rows: one per ``merge x merge`` block, ``time`` steps deep."""
    time, height, width = grid
    if height % merge or width % merge:
        raise ValueError(f"the grid {grid} is not whole {merge}x{merge} blocks")
    return time * (height // merge) * (width // merge)


def tables(case: dict, config: dict) -> dict[str, np.ndarray]:
    """Every table the case describes, keyed by the name both sides write."""
    ids = list(case["ids"])
    merge = config["merge"]
    blocks = blocks_of(case, config)

    positions = rope_index(
        ids,
        image_grids=[tuple(grid) for grid in case.get("image_grids", [])],
        video_grids=[tuple(grid) for grid in case.get("video_grids", [])],
        image_token_id=config["image_token_id"],
        video_token_id=config["video_token_id"],
        spatial_merge_size=merge,
    )
    out: dict[str, np.ndarray] = {
        "prompt_positions": positions.numpy().astype(np.int32),
        "vision_destinations": destinations(
            ids, config["image_token_id"], config["video_token_id"]
        ),
    }
    if not blocks:
        return out

    index: list[np.ndarray] = []
    weight: list[np.ndarray] = []
    cosine: list[np.ndarray] = []
    sine: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for grid in blocks:
        one = block_tables(grid, config)
        index.append(one["position_index"])
        weight.append(one["position_weight"])
        cosine.append(one["cosine"])
        sine.append(one["sine"])
        lengths.append(one["lengths"])
    out["vision_position_index"] = np.concatenate(index).astype(np.int32)
    out["vision_position_weight"] = np.concatenate(weight).astype(np.float32)
    out["vision_cosine"] = np.concatenate(cosine).astype(np.float32)
    out["vision_sine"] = np.concatenate(sine).astype(np.float32)
    out["vision_lengths"] = np.concatenate(lengths).astype(np.int32)
    return out


def destinations(ids: list[int], image_token_id: int, video_token_id: int) -> np.ndarray:
    """The prompt row each merged vision row takes, in prompt order.

    Source-major, because that is what the row operator reads
    (``ops::rows``: ``destination[indices[j]] = source[j]``). The reference states the
    inverse, a destination-major mask (``uniserve/models/qwen3_vl/encoder.py``:
    ``mask.cumsum(0) - 1``); the placeholder positions are its nonzero entries.
    """
    return np.array(
        [
            row
            for row, token in enumerate(ids)
            if token in (image_token_id, video_token_id)
        ],
        dtype=np.int32,
    )


def block_tables(grid: tuple[int, int, int], config: dict) -> dict[str, np.ndarray]:
    """One block's resample taps, 2-D rotary factors and attention offsets.

    A temporal grid repeats one time step's tables on every step, because the tower treats each
    temporal patch as its own sequence over the same ``height x width`` grid
    (``uniserve/models/qwen3_vl/vision.py::VisionTower.forward``).
    """
    time, height, width = grid
    merge = config["merge"]

    rows, columns = merged_grid_coordinates(height, width, merge, device="cpu")
    coordinates = torch.stack((rows, columns), dim=-1).repeat(time, 1)
    rotary = RotaryEmbedding(config["vision_head_dim"] // 2, theta=config["vision_theta"])
    step_cosine, step_sine = rotary(coordinates, dtype=torch.float32, sequence_length=0)
    patches = int(step_cosine.shape[0])

    # The four learned-table rows each patch blends, and their weights.
    index, weight = _taps(config["position_grid"], height, width, merge)

    return {
        "position_index": np.tile(index, (time, 1)),
        "position_weight": np.tile(weight, (time, 1)),
        "cosine": step_cosine.flatten(1).numpy().reshape(patches, -1),
        "sine": step_sine.flatten(1).numpy().reshape(patches, -1),
        "lengths": (np.arange(time + 1, dtype=np.int32) * np.int32(height * width)),
    }


def _taps(
    side: int, height: int, width: int, merge: int
) -> tuple[np.ndarray, np.ndarray]:
    """The four learned-table rows of every patch's sample, and their FP32 weights.

    Corner-aligned bilinear, in the reference's own order: a patch at merged position ``p`` is the
    patch ``_merge_order`` names, its sample is `PositionEmbedding.interpolate`'s
    (``uniserve/nn/vision/position.py:167-200``), and the four taps are top-left, top-right,
    bottom-left, bottom-right (``:196-210``).
    """
    vertical = _samples(side, height)
    horizontal = _samples(side, width)
    index: list[tuple[int, int, int, int]] = []
    weight: list[tuple[float, float, float, float]] = []
    for row, column in _merge_order(height, width, merge):
        top, bottom, down = vertical[row]
        left, right, across = horizontal[column]
        index.append(
            (
                top * side + left,
                top * side + right,
                bottom * side + left,
                bottom * side + right,
            )
        )
        weight.append(
            (
                (1.0 - down) * (1.0 - across),
                (1.0 - down) * across,
                down * (1.0 - across),
                down * across,
            )
        )
    return np.array(index, dtype=np.int32), np.array(weight, dtype=np.float32)


def _samples(extent: int, size: int) -> list[tuple[int, int, float]]:
    """Each of `size` samples' lower and upper table index and its fractional weight."""
    coordinate = torch.linspace(0, extent - 1, size)
    lower = coordinate.int()
    upper = (lower + 1).clamp(max=extent - 1)
    return list(
        zip(
            lower.tolist(),
            upper.tolist(),
            (coordinate - lower).tolist(),
        )
    )


def _grid(height: int, width: int) -> list[tuple[int, int]]:
    """Every patch's `(row, column)` in raster order."""
    return [(row, column) for row in range(height) for column in range(width)]


def _merge_order(height: int, width: int, merge: int) -> list[tuple[int, int]]:
    """The merged-row order: each `merge x merge` block's patches, blocks in raster order."""
    order = []
    for block_row in range(height // merge):
        for block_column in range(width // merge):
            for inner_row in range(merge):
                for inner_column in range(merge):
                    order.append(
                        (
                            block_row * merge + inner_row,
                            block_column * merge + inner_column,
                        )
                    )
    return order


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--text-encoder", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    arguments = parser.parse_args(argv)

    case = json.loads(arguments.case.read_text())
    config = config_of(arguments.text_encoder)
    directory = arguments.out / "text_encoder"
    directory.mkdir(parents=True, exist_ok=True)
    written = tables(case, config)
    for name, values in written.items():
        np.save(directory / f"{name}.npy", values)
    print(f"dump_encoder_tables: wrote {len(written)} tables to {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
