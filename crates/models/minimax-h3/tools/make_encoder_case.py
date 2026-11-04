#!/usr/bin/env python3
r"""Write the case files the encoder diagnosis compares, one per canvas.

Both halves of the diagnosis read a case as JSON: `ids` is the prompt's token
ids with the towers' placeholder runs in place, and `image_grids` /
`video_grids` give each block's grid in *patches* as `[temporal, height,
width]`, in the order the blocks appear. The cases are generated rather than
written out because a served canvas is thousands of placeholders long and the
interesting part is the grid, not the list:

    .venv/bin/python \
        crates/models/minimax-h3/tools/make_encoder_case.py \
        --name dmd832x480 \
        --out artifacts/diagnosis/encoder-host/case_dmd832x480.json

The ids are the checkpoint's own: `image_token_id` 151655 and `video_token_id`
151656 are what the chat template puts in a prompt, and the surrounding
integers are text tokens whose values only have to differ from each other for
the position rule to be checkable.
"""

from __future__ import annotations

import argparse
import json
import pathlib

IMAGE_TOKEN = 151655
VIDEO_TOKEN = 151656

# The canvases as `(width, height)` pixels, named the way the service names
# them; a block's grid is the raster divided by the tower's 16-pixel patch.
CANVASES: dict[str, tuple[int, int]] = {
    "developed1344x768": (1344, 768),
    "dmd1344x768": (1344, 768),
    "dmd832x480": (832, 480),
    "dmd480x832": (480, 832),
    "edge": (96, 64),
}

PATCH = 16
MERGE = 2


def grid(height: int, width: int, temporal: int) -> list[int]:
    """A block's grid in patches, `[temporal, height, width]`."""
    if height % PATCH or width % PATCH:
        raise SystemExit(
            f"{height}x{width} is not a whole number of {PATCH}-pixel patches"
        )
    return [temporal, height // PATCH, width // PATCH]


def case(
    name: str, images: int, videos: int, temporal: int = 2
) -> dict[str, object]:
    """A prompt of text tokens around `images` image and `videos` video runs."""
    width, height = CANVASES[name]
    block = grid(height, width, temporal)
    image_patches = block[1] * block[2]
    video_patches = block[0] * image_patches

    ids: list[int] = [9707]
    image_grids: list[list[int]] = []
    video_grids: list[list[int]] = []
    token = 100
    for _ in range(images):
        ids.append(token)
        token += 1
        ids.extend([IMAGE_TOKEN] * image_patches)
        ids.append(11)
        image_grids.append(grid(height, width, 1))
    for _ in range(videos):
        ids.append(token)
        token += 1
        ids.extend([VIDEO_TOKEN] * video_patches)
        ids.append(12)
        video_grids.append(block)
    ids.append(token)

    return {"ids": ids, "image_grids": image_grids, "video_grids": video_grids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", choices=sorted(CANVASES), default="dmd832x480"
    )
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--videos", type=int, default=1)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    written = case(arguments.name, arguments.images, arguments.videos)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(written))

    width, height = CANVASES[arguments.name]
    block = grid(height, width, 2)
    merged = block[1] * block[2] // (MERGE * MERGE)
    print(
        f"{arguments.out}: {len(written['ids'])} ids, "
        f"image grids {written['image_grids']}, "
        f"video grids {written['video_grids']} "
        f"({merged} merged rows a temporal step)"
    )


if __name__ == "__main__":
    main()
