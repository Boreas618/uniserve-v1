#!/usr/bin/env python3
"""Dump the text encoder's prepared tables from today's implementation (design 10).

The other half of ``examples/compare_encoder_tables.rs``: this script reads the same
``case.json``, builds the same six tables from the reference's own code
(``uniserve.models.qwen3_vl.positions.rope_index``, ``uniserve.nn.rope.RotaryEmbedding``,
``uniserve.nn.vision.position``), and writes one ``.npy`` per table under
``<out>/text_encoder/`` so that ``tools/diagnosis/compare.py`` can report the difference.

It needs no checkpoint and no GPU: every value is host arithmetic, which is the half where a
structural mistake shows first. This is a **diagnosis, never an acceptance gate**
(``specs/worker.md`` 10).

Usage::

    .venv/bin/python crates/models/minimax-h3/tools/dump_encoder_tables.py \\
        --case case.json --out artifacts/diagnosis/<run>/ref
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


def tables(case: dict) -> dict[str, np.ndarray]:
    """Every table the case describes, keyed by the name both sides write."""
    ids = list(case["ids"])
    image_token_id = case["image_token_id"]
    video_token_id = case["video_token_id"]
    merge = case["merge"]
    image_grids = [tuple(grid) for grid in case.get("image_grids", [])]
    video_grids = [tuple(grid) for grid in case.get("video_grids", [])]

    positions = rope_index(
        ids,
        image_grids=image_grids,
        video_grids=video_grids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        spatial_merge_size=merge,
    )
    out: dict[str, np.ndarray] = {
        "prompt_positions": positions.numpy().astype(np.int32),
        "vision_rows": placeholder_rows(ids, image_token_id, video_token_id),
    }

    for name, grid in (
        *((f"image{index}", grid) for index, grid in enumerate(image_grids)),
        *((f"video{index}", grid) for index, grid in enumerate(video_grids)),
    ):
        out.update(media(case, name, grid))
    return out


def placeholder_rows(
    ids: list[int], image_token_id: int, video_token_id: int
) -> np.ndarray:
    """The vision feature row each placeholder takes, ``-1`` for a text token.

    Exactly the reference encoder's expression: the k-th placeholder of the prompt
    takes row k of the concatenated feature table
    (``uniserve/models/qwen3_vl/encoder.py``: ``mask.cumsum(0) - 1``).
    """
    mask = np.array(
        [token in (image_token_id, video_token_id) for token in ids], dtype=bool
    )
    return np.where(mask, np.cumsum(mask, dtype=np.int32) - 1, -1).astype(np.int32)


def media(case: dict, name: str, grid: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """One block's rotary factors, blended position row and attention lengths.

    A temporal grid repeats one time step's factors and samples on every step, because
    the tower treats each temporal patch as its own sequence
    (``uniserve/models/qwen3_vl/vision.py::VisionTower.forward``).
    """
    time, height, width = grid
    merge = case["merge"]
    head_dim = case["head_dim"]
    position_grid = case["position_grid"]

    rows, columns = merged_grid_coordinates(height, width, merge, device="cpu")
    coordinates = torch.stack((rows, columns), dim=-1).repeat(time, 1)

    rotary = RotaryEmbedding(head_dim // 2, theta=case["rope_theta"])
    cosine, sine = rotary(coordinates, dtype=torch.float32, sequence_length=0)
    cosine = cosine.flatten(1).numpy()
    sine = sine.flatten(1).numpy()

    # The blended table row index: the reference's own `PositionEmbedding.interpolate` over a
    # table whose single column is the row index, so its output is the blend of the four rows.
    embedding = PositionEmbedding((position_grid, position_grid), 1)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(position_grid * position_grid, dtype=torch.float32).reshape(-1, 1)
        )
    blended = (
        embedding.interpolate(height, width, merge=merge).repeat(time, 1).flatten().numpy()
    )

    patches = int(cosine.shape[0])
    lengths = (np.arange(time + 1, dtype=np.int32) * np.int32(height * width)).astype(
        np.int32
    )
    return {
        f"{name}_cosine": cosine.reshape(patches, -1),
        f"{name}_sine": sine.reshape(patches, -1),
        f"{name}_position_row": blended.astype(np.float32),
        f"{name}_lengths": lengths,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    arguments = parser.parse_args(argv)

    case = json.loads(arguments.case.read_text())
    directory = arguments.out / "text_encoder"
    directory.mkdir(parents=True, exist_ok=True)
    written = tables(case)
    for name, values in written.items():
        np.save(directory / f"{name}.npy", values)
    print(f"dump_encoder_tables: wrote {len(written)} tables to {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
