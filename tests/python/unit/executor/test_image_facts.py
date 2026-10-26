"""Vision blocks and image facts (``uniserve.executor.image.facts``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.python.fixtures.stub import load_stub
from uniserve.executor.image.facts import ImageFacts, Injection
from uniserve.executor.image.layout import ImageLayout
from uniserve.executor.text.context import block_row
from uniserve.processing import BranchSource, FeatureLayout, PositionLayout

pytestmark = pytest.mark.unit

START, END = 151_652, 151_653

# A framing image denoiser whose generated image spans two positions.
LAYOUT = ImageLayout(
    patch_size=16,
    latent_channels=3,
    downsample=16,
    framing_tokens=2,
    image_unconditional=BranchSource.START,
    max_sequence_tokens=1024,
    rope_advance=2,
    guidance_branches=3,
    prediction_dtype=torch.bfloat16,
)


def _injection(layout: FeatureLayout, positions: PositionLayout) -> Injection:
    return Injection(layout, positions, start_token=START, end_token=END)


def _unasked(height: int, width: int, input_images: int) -> tuple[int, int]:
    raise AssertionError("the facts ask no patch grid")


def test_framed_temporal_blocks_hold_their_features_between_markers():
    block = _injection(FeatureLayout.FRAMED, PositionLayout.TEMPORAL).block(
        3, 9, False, None
    )
    assert block.token_ids.tolist() == [START, 1, 1, 1, END]
    assert block.features == range(1, 4)
    np.testing.assert_array_equal(block.positions, [9] * 5)
    assert block.positions.dtype == np.int64


def test_sequential_blocks_take_consecutive_positions():
    block = _injection(FeatureLayout.DIRECT, PositionLayout.SEQUENTIAL).block(
        4, 10, False, None
    )
    assert block.token_ids.tolist() == [1] * 4
    np.testing.assert_array_equal(block.positions, [10, 11, 12, 13])


def test_spatial_blocks_take_raster_coordinates_of_the_pooled_grid():
    spatial = _injection(FeatureLayout.DIRECT, PositionLayout.TEMPORAL_SPATIAL)

    # A 4x6 patch grid pooled 2x2 into 2x3 features.
    block = spatial.block(6, 5, False, (4, 6))
    assert block.features == range(0, 6)
    np.testing.assert_array_equal(
        block.positions,
        [[5] * 6, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
    )

    # A closing block sits one temporal position later and ends with the
    # end marker two past the conditioning position.
    block = spatial.block(6, 5, True, (4, 6))
    assert block.token_ids.tolist() == [1, 1, 1, 1, 1, 1, END]
    np.testing.assert_array_equal(
        block.positions,
        [
            [6, 6, 6, 6, 6, 6, 7],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 1, 2, 0, 1, 2, 0],
        ],
    )


@pytest.mark.parametrize("positions", list(PositionLayout))
def test_every_layouts_block_places_as_a_text_computation_row(positions):
    """A text computation places each block the image facts lay out.

    The text computation reads a block's positions as one axis or three;
    every injection layout's block becomes a row with three position axes
    whose first is the block's temporal or sequential position.
    """
    block = _injection(FeatureLayout.FRAMED, positions).block(
        6, 5, False, (4, 6)
    )
    row = block_row(block, slot=2, prefix=5, spatial=True, selection=0)
    assert row.tokens.tolist() == block.token_ids.tolist()
    assert row.positions.shape == (3, len(block))
    expected = np.asarray(block.positions).reshape(-1, len(block))[0]
    np.testing.assert_array_equal(row.positions[0], expected)


def test_grids_that_do_not_pool_into_the_features_are_refused():
    spatial = _injection(FeatureLayout.DIRECT, PositionLayout.TEMPORAL_SPATIAL)
    # 24 patches over 4 features is a pooling of 6, not a square.
    with pytest.raises(ValueError):
        spatial.block(4, 0, False, (4, 6))
    with pytest.raises(ValueError):
        spatial.block(6, 0, False, None)
    with pytest.raises(ValueError):
        spatial.block(0, 0, False, (4, 6))
    unresolved = Injection(
        FeatureLayout.FRAMED,
        PositionLayout.TEMPORAL,
        start_token=START,
        end_token=None,
    )
    with pytest.raises(ValueError):
        unresolved.block(2, 0, False, None)


@pytest.mark.parametrize(
    "injection,layout,spatial,advance",
    [
        (None, None, False, 1),
        (PositionLayout.TEMPORAL, None, False, 1),
        (PositionLayout.SEQUENTIAL, None, False, 1),
        (PositionLayout.TEMPORAL_SPATIAL, None, True, 1),
        (None, LAYOUT, True, 2),
        (PositionLayout.TEMPORAL, LAYOUT, True, 2),
    ],
)
def test_text_rows_are_spatial_with_an_image_denoiser_or_spatial_features(
    injection, layout, spatial, advance
):
    facts = ImageFacts(
        None
        if injection is None
        else _injection(FeatureLayout.DIRECT, injection),
        layout,
        _unasked,
    )
    assert facts.spatial_positions() is spatial
    assert facts.rope_advance == advance


def test_each_grid_is_asked_once_and_a_closing_image_ignores_input_images():
    asked = []

    def patch_grid(height: int, width: int, input_images: int):
        asked.append((height, width, input_images))
        return [height // 16, width // 16 + input_images]

    facts = ImageFacts(None, None, patch_grid)

    # A generated image keeps the single-image bound whatever its request's
    # input images.
    assert facts.grid(48, 64, True, 3) == (3, 4)
    assert facts.grid(48, 64, True, 1) == (3, 4)
    # A request's input image takes its request's input image count.
    assert facts.grid(48, 64, False, 2) == (3, 6)
    assert facts.grid(48, 64, False, 2) == (3, 6)
    assert asked == [(48, 64, 0), (48, 64, 2)]


@pytest.mark.parametrize("answer", [[0, 4], [3, -1], [3], [3, 4, 5], [3.0, 4]])
def test_grids_that_are_not_two_positive_extents_are_refused(answer):
    facts = ImageFacts(None, None, lambda height, width, images: answer)
    with pytest.raises(ValueError, match="positive extents"):
        facts.grid(48, 64, False, 1)


def test_a_loaded_model_states_its_image_facts():
    facts = ImageFacts.from_loaded(load_stub())

    # The stub injects features directly at temporal-spatial positions and
    # places an end token only.
    assert facts.injection == Injection(
        FeatureLayout.DIRECT,
        PositionLayout.TEMPORAL_SPATIAL,
        start_token=None,
        end_token=1007,
    )
    assert facts.spatial_positions()
    assert facts.rope_advance == 2
    assert facts.grid(48, 64, False, 1) == (3, 4)
