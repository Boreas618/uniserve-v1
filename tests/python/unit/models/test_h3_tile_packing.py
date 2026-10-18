"""A sparse H3 layout's extent sizes the tables of its packing.

``tile_extent`` sizes the ``[text | audio | tiled video | padding]`` rows of
a layout from its prompt length, frame count, canvas and alignment;
``tile_packing`` fills the tables of the same rows. ``segment_extent`` and
``segment_packing`` do the same for a multi-segment layout. A denoiser
evaluates a layout from the extent alone, including which audio and video
rows fall below a rank's shard boundary, so both must describe the same
rows.
"""

import dataclasses

import pytest
import torch

from uniserve.media import image
from uniserve.models.minimax_h3.packing import (
    SegmentExtent,
    TileExtent,
    segment_extent,
    segment_packing,
    tile_extent,
    tile_packing,
)

pytestmark = pytest.mark.unit

WIDE = image.Config(768, 1344)


@pytest.mark.parametrize(
    ("num_text_tokens", "text_rows", "num_frames", "token_multiple"),
    [(100, None, 124, 256), (100, 1024, 124, 256), (1000, 1024, 362, 512)],
)
def test_an_extent_sizes_its_packing(
    num_text_tokens, text_rows, num_frames, token_multiple
):
    layout = {
        "num_text_tokens": num_text_tokens,
        "num_frames": num_frames,
        "canvas": WIDE,
        "token_multiple": token_multiple,
        "text_rows": text_rows,
    }
    extent = tile_extent(**layout)
    packing = tile_packing(**layout)

    for field in dataclasses.fields(TileExtent):
        assert getattr(packing, field.name) == getattr(extent, field.name)
    # The padded rows are whole tiles in pairs at the alignment, and the
    # tables cover exactly them.
    assert extent.padded_tokens % max(token_multiple, 128) == 0
    assert extent.padded_tokens // 64 % 2 == 0
    assert packing.position_ids.shape == (extent.padded_tokens, 3)
    assert packing.token_tags.shape == (extent.padded_tokens,)
    assert packing.tile_valid_sizes.shape == (extent.padded_tokens // 64,)
    # Every valid row is a prompt, audio or video token.
    assert int(packing.tile_valid_sizes.sum()) == extent.num_tokens
    counts = (
        packing.text_indices.numel()
        + packing.audio_indices.numel()
        + packing.video_indices.numel()
    )
    assert counts == extent.num_tokens
    # The video tiles follow the dense text and audio prefix.
    assert int(packing.video_indices.min()) == extent.prefix_tiles * 64
    video_end = (extent.prefix_tiles + extent.video_tiles) * 64
    assert int(packing.video_indices.max()) < video_end <= extent.padded_tokens


def test_a_larger_text_segment_keeps_the_valid_tokens():
    own = tile_extent(num_text_tokens=100, num_frames=124, canvas=WIDE)
    wide = tile_extent(
        num_text_tokens=100, num_frames=124, canvas=WIDE, text_rows=1024
    )
    assert wide.num_tokens == own.num_tokens
    assert wide.video_tiles == own.video_tiles
    # The segment's 1024 rows replace the prompt's own 128.
    assert wide.prefix_tiles - own.prefix_tiles == (1024 - 128) // 64


@pytest.mark.parametrize(
    "layout",
    [
        {"num_text_tokens": 0},
        {"token_multiple": 100},
        {"text_rows": 100},
        {"num_text_tokens": 100, "text_rows": 64},
        {"audio_frames": 0},
    ],
)
def test_an_extent_refuses_what_its_packing_refuses(layout):
    arguments = {"num_text_tokens": 10, "num_frames": 124, "canvas": WIDE}
    arguments.update(layout)
    for size in (tile_extent, tile_packing):
        with pytest.raises(ValueError):
            size(**arguments)


SEGMENT_LAYOUTS = [
    # Text and condition capacities, tile and alignment of OmniRef layouts,
    # including a canvas whose token grid ends inside the 4x4x8 tiles.
    {
        "num_frames": 124,
        "canvas": WIDE,
        "text_rows": 8192,
        "condition_rows": 49152,
        "tile": 128,
        "token_multiple": 512,
    },
    {
        "num_frames": 22,
        "canvas": image.Config(256, 448),
        "text_rows": 256,
        "condition_rows": 0,
        "tile": 128,
        "token_multiple": 128,
    },
    {
        "num_frames": 39,
        "canvas": image.Config(704, 1248),
        "text_rows": 128,
        "condition_rows": 384,
        "tile": 64,
        "token_multiple": 256,
    },
]


@pytest.mark.parametrize("layout", SEGMENT_LAYOUTS)
def test_a_segment_extent_sizes_its_packing(layout):
    extent = segment_extent(**layout)
    packing = segment_packing(**layout)

    for field in dataclasses.fields(SegmentExtent):
        assert getattr(packing, field.name) == getattr(extent, field.name)
    assert extent.padded_tokens % layout["token_multiple"] == 0
    assert packing.video_valid_sizes.shape == (extent.video_tiles,)
    frames, height, width = extent.video_grid
    assert int(packing.video_valid_sizes.sum()) == frames * height * width
    video_end = extent.video_start + extent.video_tiles * extent.tile
    assert int(packing.video_indices.min()) == extent.video_start
    assert int(packing.video_indices.max()) < video_end <= extent.padded_tokens


def _counted_below(extent, packing, tile):
    """Each tile boundary's audio and video rows below it: by the extent
    and by the packing's tables.
    """  # noqa: D205
    boundaries = torch.arange(0, extent.padded_tokens + 1, tile)
    counted = [
        (extent.audio_rows_below(row), extent.video_rows_below(row))
        for row in boundaries.tolist()
    ]
    tabled = list(
        zip(
            torch.searchsorted(packing.audio_indices, boundaries).tolist(),
            torch.searchsorted(packing.video_indices, boundaries).tolist(),
            strict=True,
        )
    )
    return counted, tabled


@pytest.mark.parametrize("layout", SEGMENT_LAYOUTS)
def test_a_segment_extent_counts_the_rows_below_each_tile_boundary(layout):
    extent = segment_extent(**layout)
    counted, tabled = _counted_below(
        extent, segment_packing(**layout), extent.tile
    )
    assert counted == tabled
    with pytest.raises(ValueError):
        extent.video_rows_below(extent.video_start + 1)


@pytest.mark.parametrize("num_frames", [22, 124, 362])
def test_a_tile_extent_counts_the_rows_below_each_tile_boundary(num_frames):
    layout = {
        "num_text_tokens": 300,
        "num_frames": num_frames,
        "canvas": image.Config(704, 1248),
        "token_multiple": 512,
        "text_rows": 512,
    }
    counted, tabled = _counted_below(
        tile_extent(**layout), tile_packing(**layout), 64
    )
    assert counted == tabled
