"""Images packed into fixed slots encode as each image does alone."""

import pytest
import torch
from torch import nn

from uniserve.models import PatchEncoder
from uniserve.models.sensenova_u1 import vision

pytestmark = pytest.mark.unit


def test_neo_slots_encode_each_image_as_it_encodes_alone():
    # 4-pixel patches merged 2x2 into one token, three slots of 32 rows.
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

    # Padding rows hold arbitrary finite values.
    packed = torch.randn(3 * 32, 48)
    for slot, pixels in enumerate(images):
        packed[slot * 32 : slot * 32 + pixels.shape[0]] = pixels
    grids = torch.tensor(encoder.packed_grids(shapes, 3))
    features = encoder.encode_packed(packed, grids)

    assert features.shape == (3 * 32 // 4, 24)
    for slot, expected in enumerate(alone):
        torch.testing.assert_close(
            features[slot * 8 : slot * 8 + expected.shape[0]], expected
        )
