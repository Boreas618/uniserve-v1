"""A sparse H3 layout's extent sizes the tables of its packing.

``segment_extent`` sizes the ``[text | conditions | audio | tiled video |
padding]`` rows of a layout from its frame count, canvas, capacities and
alignment; ``segment_packing`` fills the tables of the same rows. A denoiser
evaluates a layout from the extent alone, including which audio and video
rows fall below a rank's shard boundary and how many key tiles each query
tile may attend, so both must describe the same rows and the bounds must
hold every request's selection.
"""

import dataclasses

import pytest
import torch

from uniserve.media import image
from uniserve.models.minimax_h3.packing import (
    Segment,
    SegmentExtent,
    SegmentKind,
    segment_extent,
    segment_packing,
    segment_tables,
    segment_tiles,
)
from uniserve.nn.attention import vsa
from uniserve.nn.attention.vsa.segments import select

pytestmark = pytest.mark.unit

WIDE = image.Config(768, 1344)

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
    assert packing.video_variable_block_sizes.shape == (extent.video_tiles,)
    frames, height, width = extent.video_grid
    assert (
        int(packing.video_variable_block_sizes.sum()) == frames * height * width
    )
    video_end = extent.video_start + extent.video_tiles * extent.tile
    assert int(packing.video_indices.min()) == extent.video_start
    assert int(packing.video_indices.max()) < video_end <= extent.padded_tokens


@pytest.mark.parametrize("layout", SEGMENT_LAYOUTS)
def test_a_segment_extent_counts_the_rows_below_each_tile_boundary(layout):
    extent = segment_extent(**layout)
    packing = segment_packing(**layout)
    boundaries = torch.arange(0, extent.padded_tokens + 1, extent.tile)
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
    assert counted == tabled
    with pytest.raises(ValueError):
        extent.video_rows_below(extent.video_start + 1)


# A reference video of 22 frames on a 256x448 raster: 6 latent frames of a
# 16x28 latent raster, 8x14 tokens each.
REFERENCE = Segment(SegmentKind.VIDEO, 0, None, 6, 16, 28, 0)


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("references", [(), (REFERENCE,)])
def test_every_query_tile_keeps_at_most_its_key_bound(tile, references):
    canvas = image.Config(256, 448)
    sparsity = 0.75
    packing = segment_packing(
        num_frames=39,
        canvas=canvas,
        text_rows=2 * tile,
        condition_rows=tile
        * sum(segment_tiles(segment, tile) for segment in references),
        tile=tile,
        token_multiple=2 * tile,
    )
    tables = segment_tables(
        packing,
        num_frames=39,
        canvas=canvas,
        num_text_tokens=tile + 3,
        segments=references,
        vision_spans=(),
        sparsity=sparsity,
        reference_keep=0.5 if references else None,
    )
    bounds = packing.key_bounds(sparsity)
    segments = vsa.Segments(
        tile,
        packing.padded_tokens,
        tables.variable_block_sizes,
        tables.tile_segments,
        tables.segment_starts,
        tables.segment_keep,
        dense_prefix_tiles=packing.dense_prefix_tiles,
        key_bounds=bounds,
    )
    tiles = packing.padded_tokens // tile
    torch.manual_seed(tile + len(references))
    _, counts = select(torch.randn(3, tiles, tiles), segments)

    assert (counts <= torch.tensor(bounds, dtype=torch.int32)).all()
    # The leading dense tiles keep every live key tile.
    live = int((tables.variable_block_sizes > 0).sum())
    for index in range(packing.dense_prefix_tiles):
        if tables.variable_block_sizes[index] > 0:
            assert (counts[:, index] == live).all()
