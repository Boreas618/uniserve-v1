"""An image denoiser's latent layout places its framed sequences."""

import numpy as np
import pytest
import torch

from tests.python.fixtures.stub import load_stub
from uniserve.executor.image.layout import ImageLayout
from uniserve.processing import BranchSource

pytestmark = pytest.mark.unit


def _layout(framing: int) -> ImageLayout:
    return ImageLayout(
        patch_size=2,
        latent_channels=16,
        downsample=16,
        framing_tokens=framing,
        image_unconditional=BranchSource.CONDITIONING,
        max_sequence_tokens=4096,
        rope_advance=2,
        guidance_branches=3,
        prediction_dtype=torch.bfloat16,
    )


def test_image_positions_raster_latent_patches_inside_their_framing():
    # A 32x48 image of 16-pixel latent patches: a 2x3 grid.
    framed = _layout(2)
    assert framed.latent_rows(32, 48) == 6
    assert framed.sequence_length(32, 48) == 8
    assert framed.width == 64
    np.testing.assert_array_equal(
        framed.positions(32, 48, 7),
        [
            [7] * 8,
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 2, 0, 1, 2, 0],
        ],
    )
    unframed = _layout(0)
    np.testing.assert_array_equal(
        unframed.positions(32, 48, 0),
        [[0] * 6, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
    )
    with pytest.raises(ValueError, match="latent patch"):
        unframed.latent_rows(8, 48)


def test_latent_conditioning_frames_the_image_around_its_rope_span():
    positions = _layout(2).conditioning_positions(32, 48, 10)
    np.testing.assert_array_equal(
        positions[0], [10, 11, 11, 11, 11, 11, 11, 12]
    )
    with pytest.raises(ValueError, match="framed"):
        _layout(0).conditioning_positions(32, 48, 10)


def test_a_loaded_denoiser_states_its_layout():
    # The stub frames images with two tokens and its image-unconditional
    # branch reads the start of a conversation.
    layout = ImageLayout.of(load_stub().module.denoiser)
    assert (
        layout.patch_size,
        layout.latent_channels,
        layout.downsample,
        layout.framing_tokens,
        layout.max_sequence_tokens,
        layout.rope_advance,
        layout.guidance_branches,
    ) == (16, 3, 16, 2, 1024, 2, 3)
    assert layout.image_unconditional is BranchSource.START
    assert layout.max_row_tokens == 1026


def test_layouts_frame_images_with_zero_or_two_tokens():
    with pytest.raises(ValueError, match="zero or two"):
        ImageLayout(
            patch_size=2,
            latent_channels=16,
            downsample=16,
            framing_tokens=1,
            image_unconditional=BranchSource.START,
            max_sequence_tokens=4096,
            rope_advance=2,
            guidance_branches=3,
            prediction_dtype=torch.bfloat16,
        )
