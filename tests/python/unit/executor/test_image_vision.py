"""Vision encodes pack images back to back in patch-row buckets."""

import pytest
import torch
from torch import nn

from tests.python.fixtures.stub import load_stub
from uniserve.bindings.workspace import Scratch
from uniserve.executor.image.plan import chunk_rows, segment_grids, shape_groups
from uniserve.executor.image.vision import (
    VisionEncoder,
    VisionImage,
    features,
)
from uniserve.models import PatchEncoder, VisionInput
from uniserve.models.sensenova_u1 import vision

pytestmark = pytest.mark.unit


def test_segment_grids_describe_images_empty_slots_and_padding():
    # Two images in three slots of a 192-row bucket, downsampled by 2: each
    # image's grid, an empty slot, then a 2-row grid over the remaining 96
    # patch rows.
    assert segment_grids([(4, 8), (8, 8)], 3, 192, 2) == [
        (4, 8),
        (8, 8),
        (0, 0),
        (2, 48),
    ]
    assert segment_grids([], 1, 64, 2) == [(0, 0), (2, 32)]
    # Images filling the bucket leave an empty padding segment.
    assert segment_grids([(8, 8)], 1, 64, 2) == [(8, 8), (0, 0)]


def test_segment_grids_reject_unaligned_oversized_and_excess_images():
    with pytest.raises(ValueError, match="downsampling"):
        segment_grids([(3, 4)], 1, 64, 2)
    with pytest.raises(ValueError, match="exceed"):
        segment_grids([(10, 8)], 1, 64, 2)
    with pytest.raises(ValueError, match="slots"):
        segment_grids([(2, 2), (2, 2)], 1, 64, 2)
    with pytest.raises(ValueError, match="downsampling"):
        segment_grids([(0, 4)], 1, 64, 2)
    with pytest.raises(ValueError, match="partial"):
        segment_grids([(2, 2)], 1, 66, 2)


def test_packed_segments_encode_each_image_as_it_encodes_alone():
    # 4-pixel patches merged 2x2 into one token.
    config = vision.Config(16, 24, 0.5, 4, 3, 10000.0)
    torch.manual_seed(13)
    network = vision.Encoder(config)
    shapes = [(4, 6), (2, 4)]
    images = [torch.randn(rows * columns, 48) for rows, columns in shapes]
    alone = [
        network(pixels, torch.tensor([shape]), (shape,))
        for pixels, shape in zip(images, shapes, strict=True)
    ]

    # Both images back to back in a bucket of 48 patch rows and four slots;
    # the padding rows hold arbitrary finite values.
    grids = segment_grids(shapes, 4, 48, 2)
    pixels = torch.cat((*images, torch.randn(48 - 32, 48)))
    features = network(pixels, torch.tensor(grids), None)

    assert features.shape == (48 // 4, 24)
    torch.testing.assert_close(features[:6], alone[0])
    torch.testing.assert_close(features[6:8], alone[1])


def test_groups_split_into_chunks_within_their_row_budget():
    rows = [3, 4, 2, 9, 1]
    chunks = chunk_rows([0, 1, 2, 3, 4], 6, lambda index: rows[index])
    assert chunks == [[0], [1, 2], [3], [4]]
    # A slot bound also splits a chunk.
    assert chunk_rows(range(5), 100, lambda _: 1, members=2) == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_unpacked_images_group_by_pixel_shape_in_appearance_order():
    a, b, c = (3, 32, 32), (3, 16, 32), (3, 32, 32)
    assert shape_groups([(0, a), (1, b), (2, c)]) == [[0, 2], [1]]
    # Members keep their own indices.
    assert shape_groups([(4, b), (7, a)]) == [[4], [7]]


def _neo_tower(seed: int) -> PatchEncoder:
    """A NEO tower of 4-pixel patches merged 2x2.

    It packs at most three images of 32 patch rows.
    """
    torch.manual_seed(seed)
    return PatchEncoder(
        vision.Encoder(vision.Config(16, 24, 0.5, 4, 3, 10000.0)),
        nn.Identity(),
        patch_size=4,
        downsample=2,
        output_size=24,
        output_dtype=torch.float32,
        max_patches=32,
        max_slots=3,
    )


def _encoder(module: PatchEncoder) -> VisionEncoder:
    return VisionEncoder(
        module=module,
        owner="model.vision_encoder",
        stream=None,
        device=-1,
        scratch=Scratch(),
        attention="torch",
        graphs=False,
        pool=None,
        peers=(),
        pixel_dtype=torch.float32,
        row_budget=64,
        max_images=4,
    )


def test_packed_images_encode_into_their_products_as_alone():
    module = _neo_tower(17)
    shapes = [(4, 6), (2, 4), (2, 2), (4, 4)]
    images = [torch.randn(rows * columns, 48) for rows, columns in shapes]
    alone = [
        module.network(pixels, torch.tensor([shape]), (shape,))
        for pixels, shape in zip(images, shapes, strict=True)
    ]
    outputs = [
        torch.full((rows * columns // 4, 24), float("nan"))
        for rows, columns in shapes
    ]
    # Four images exceed the tower's three images a call: two packed calls.
    _encoder(module).encode(
        [
            VisionImage(pixels, shape, out)
            for pixels, shape, out in zip(images, shapes, outputs, strict=True)
        ]
    )
    for actual, expected in zip(outputs, alone, strict=True):
        torch.testing.assert_close(actual, expected)


def test_an_unpacked_tower_encodes_each_pixel_shape_together():
    stub = load_stub()
    module = stub.module.vision_encoder
    assert module.max_patches is None
    width, _ = features(module)
    pixels = [
        torch.rand(rows * columns, 3 * 16 * 16).to(torch.bfloat16)
        for rows, columns in ((2, 3), (2, 3), (1, 2))
    ]
    grids = [(2, 3), (2, 3), (1, 2)]
    outputs = [
        torch.zeros((rows * columns, width), dtype=torch.bfloat16)
        for rows, columns in grids
    ]
    _encoder(module).encode(
        [
            VisionImage(image, grid, out)
            for image, grid, out in zip(pixels, grids, outputs, strict=True)
        ]
    )
    for image, grid, out in zip(pixels, grids, outputs, strict=True):
        (expected,) = module.encode(
            VisionInput((image,), (torch.tensor([grid]),), (grid,))
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
