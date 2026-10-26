"""Packed images and token blocks encode as each image does alone."""

import pytest
import torch
from torch import nn

from uniserve.models import PatchEncoder
from uniserve.models.sensenova_u1 import vision
from uniserve.nn.functional import patchify

pytestmark = pytest.mark.unit


def test_neo_segments_encode_each_image_as_it_encodes_alone():
    # 4-pixel patches merged 2x2 into one token, at most three images of 32
    # patch rows each.
    config = vision.Config(16, 24, 0.5, 4, 3, 10000.0)
    torch.manual_seed(11)
    network = vision.Encoder(config)
    encoder = PatchEncoder(
        network,
        nn.Identity(),
        patch_size=4,
        downsample=2,
        output_size=24,
        output_dtype=torch.float32,
        max_patches=32,
        max_slots=3,
    )
    shapes = [(4, 6), (2, 4)]
    images = [torch.randn(height * width, 48) for height, width in shapes]
    alone = [
        network(pixels, torch.tensor([shape]), (shape,))
        for pixels, shape in zip(images, shapes, strict=True)
    ]

    # Both images back to back, an empty segment, then 16 padding rows of
    # arbitrary finite values under a two-row grid.
    packed = torch.cat((*images, torch.randn(16, 48)))
    grids = torch.tensor([(4, 6), (2, 4), (0, 0), (2, 8)])
    features = encoder.encode_segments(packed, grids)

    assert features.shape == (48 // 4, 24)
    torch.testing.assert_close(features[:6], alone[0])
    torch.testing.assert_close(features[6:8], alone[1])
    # Patch rows that are not whole features are refused.
    with pytest.raises(ValueError, match="whole features"):
        encoder.encode_segments(packed[:-1], grids)


def test_neo_token_blocks_encode_as_their_whole_images():
    # 4-pixel patches merged 2x2: each token holds an 8x8 pixel block.
    config = vision.Config(16, 24, 0.5, 4, 3, 10000.0)
    torch.manual_seed(5)
    network = vision.Encoder(config)
    images = [torch.randn(3, 16, 24), torch.randn(3, 8, 16)]
    whole = [
        network(
            pixels[None],
            torch.tensor([[pixels.shape[1] // 4, pixels.shape[2] // 4]]),
            ((pixels.shape[1] // 4, pixels.shape[2] // 4),),
        )
        for pixels in images
    ]

    # Both images' token blocks packed back to back, then two padding
    # tokens of arbitrary finite pixels at the grid origin.
    blocks, coordinates = [], []
    for pixels in images:
        rows, columns = pixels.shape[1] // 8, pixels.shape[2] // 8
        blocks.append(patchify(pixels, patch_size=8))
        token = torch.arange(rows * columns)
        coordinates.append(torch.stack((token // columns, token % columns), 1))
    blocks.append(torch.randn(2, 3 * 8 * 8))
    coordinates.append(torch.zeros(2, 2, dtype=torch.int64))
    tokens = vision.TokenEncoder(config)
    tokens.load_state_dict(network.state_dict())
    features = tokens(torch.cat(blocks), torch.cat(coordinates))

    # The grid and token paths run the same per-patch and per-square
    # reductions on differently shaped convolution inputs, so only the
    # summation order may differ: at most a few FP32 rounding steps over the
    # 64-term dense reduction, bounded by 64 * eps32 ~ 8e-6 of each value's
    # scale.
    assert features.shape == (6 + 2 + 2, 24)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[:6], whole[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(features[6:8], whole[1], rtol=1e-5, atol=1e-5)
