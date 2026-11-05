"""Encoded image inputs preserve the model's pixel values and patch layout."""

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from uniserve.media.pixels import (
    image_size,
    patch_grid_shape,
    prepare_host_image,
    prepare_tensor_image,
    prepared_shape,
)
from uniserve.processing import (
    ImageProcessor,
    PatchTransform,
    PixelBounds,
    StrideResize,
    TowerTransform,
)

pytestmark = pytest.mark.unit


def _prepare(processor, tower, payload, *, input_images=1):
    """Prepare a request input image on the host, as a host-pool task does."""
    return prepare_host_image(
        processor, tower, payload, input_images=input_images, pin=False
    )


@pytest.mark.parametrize("normalization", ("signed_unit", "imagenet"))
@pytest.mark.parametrize("patches", (False, True))
@pytest.mark.parametrize("dtype", ("float32", "bfloat16"))
def test_encoded_pixels_match_channel_normalization(
    normalization, patches, dtype
):
    # Include every byte value in each RGB channel. Geometry already satisfies
    # both tower policies, so the expected values do not depend on a resizer.
    raw = (np.arange(24 * 32 * 3).reshape(24, 32, 3) % 256).astype(np.uint8)
    image = Image.fromarray(raw)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    transform = (
        PatchTransform(4, 1, PixelBounds(24 * 32, 24 * 32), normalization)
        if patches
        else TowerTransform(StrideResize(32, 24, 1, 24 * 32), normalization)
    )
    processor = ImageProcessor(
        vit=transform, output_dtype=getattr(torch, dtype)
    )

    result = _prepare(
        processor, "vit", base64.b64encode(encoded.getvalue()).decode()
    )

    expected = (
        torch.from_numpy(raw).permute(2, 0, 1).contiguous().float() / 255.0
    )
    if normalization == "signed_unit":
        expected = (expected - 0.5) / 0.5
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        expected = (expected - mean) / std
    if patches:
        # Encoder rows are grid-major; each row contains channel-major pixels.
        expected = torch.stack(
            [
                expected[:, y : y + 4, x : x + 4].reshape(-1)
                for y in range(0, 24, 4)
                for x in range(0, 32, 4)
            ]
        )
        assert result.grid_shape == (6, 8)
    assert (result.height, result.width) == (24, 32)
    torch.testing.assert_close(
        result.pixels, expected.to(getattr(torch, dtype)), rtol=0, atol=0
    )


@pytest.mark.parametrize("patches", (False, True))
@pytest.mark.parametrize("size", ((16, 16), (40, 24), (24, 72)))
def test_prepared_shapes_describe_the_prepared_pixels(patches, size):
    height, width = size
    raw = np.full((height, width, 3), 77, dtype=np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    payload = base64.b64encode(encoded.getvalue()).decode()
    processor = ImageProcessor(
        vae=TowerTransform(StrideResize(64, 32, 16, 4096)),
        vit=(
            PatchTransform(4, 1, PixelBounds(16 * 16, 48 * 48))
            if patches
            else TowerTransform(StrideResize(64, 48, 16, 4096))
        ),
    )
    assert image_size(payload) == (height, width)
    for tower in ("vit", "vae"):
        # A request input image, of a request with two of them.
        host = _prepare(processor, tower, payload, input_images=2)
        assert prepared_shape(processor, tower, height, width, 2) == (
            (host.height, host.width),
            host.grid_shape,
            tuple(host.pixels.shape),
        )
        # A generated image keeps the single-image bound.
        resident = prepare_tensor_image(
            processor,
            tower,
            torch.from_numpy(raw).permute(2, 0, 1).float() / 255,
            device=torch.device("cpu"),
            signed_unit=False,
        )
        assert prepared_shape(processor, tower, height, width, None) == (
            (resident.height, resident.width),
            resident.grid_shape,
            tuple(resident.pixels.shape),
        )


@pytest.mark.parametrize("layout", ("channels_first", "channels_last"))
def test_a_tower_transform_with_a_patch_layout_prepares_patch_rows(layout):
    # The rows hold the pixels the transform prepares without the layout,
    # one row per stride square in raster order, for a request input image
    # and for a generated one alike.
    raw = (np.arange(40 * 56 * 3).reshape(40, 56, 3) % 251).astype(np.uint8)
    resize = StrideResize(64, 32, 8, 4096)
    pixels = ImageProcessor(vit=TowerTransform(resize))
    rows = ImageProcessor(vit=TowerTransform(resize, patch_layout=layout))
    payload = _png(Image.fromarray(raw))
    generated = torch.from_numpy(raw).permute(2, 0, 1).float() / 255

    for prepare in (
        lambda processor: _prepare(processor, "vit", payload),
        lambda processor: prepare_tensor_image(
            processor,
            "vit",
            generated,
            device=torch.device("cpu"),
            signed_unit=False,
        ),
    ):
        image, packed = prepare(pixels), prepare(rows)
        channels, height, width = image.pixels.shape
        grid = (height // 8, width // 8)
        squares = image.pixels.reshape(channels, grid[0], 8, grid[1], 8)
        expected = (
            squares.permute(1, 3, 2, 4, 0)
            if layout == "channels_last"
            else squares.permute(1, 3, 0, 2, 4)
        ).reshape(grid[0] * grid[1], channels * 64)
        assert packed.grid_shape == grid
        assert (packed.height, packed.width) == (image.height, image.width)
        torch.testing.assert_close(packed.pixels, expected, rtol=0, atol=0)
        assert prepared_shape(rows, "vit", 40, 56, 1)[1:] == (
            grid,
            (grid[0] * grid[1], channels * 64),
        )


@pytest.mark.parametrize("tower", ("vae", "vit"))
def test_image_sources_preserve_the_same_model_canvas(tower):
    raw = np.full((16, 16, 3), 128, dtype=np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    processor = ImageProcessor(
        vae=TowerTransform(StrideResize(64, 32, 16, 4096)),
        vit=TowerTransform(StrideResize(64, 48, 16, 4096)),
        output_dtype=torch.float32,
    )
    uploaded = _prepare(
        processor, tower, base64.b64encode(encoded.getvalue()).decode()
    )
    resident = prepare_tensor_image(
        processor,
        tower,
        torch.from_numpy(raw).permute(2, 0, 1).float() / 255,
        device=torch.device("cpu"),
        signed_unit=False,
    )
    assert (uploaded.height, uploaded.width) == (32, 32)
    assert (resident.height, resident.width) == (32, 32)
    expected_size = 32 if tower == "vae" else 48
    assert (
        uploaded.pixels.shape
        == resident.pixels.shape
        == (3, expected_size, expected_size)
    )


def _png(image: Image.Image) -> str:
    """Return an image as a base64 PNG payload, which decodes losslessly."""
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


@pytest.mark.parametrize("alpha", ("white", "drop"))
def test_alpha_policy_sets_the_color_of_transparent_pixels(alpha):
    # Opaque pixels keep their color under both policies; fully transparent
    # ones become white or keep their stored color.
    raw = np.zeros((24, 32, 4), dtype=np.uint8)
    raw[..., :3] = (10, 20, 30)
    raw[:, :16, 3] = 255
    processor = ImageProcessor(
        vit=TowerTransform(StrideResize(32, 24, 1, 24 * 32), "signed_unit"),
        output_dtype=torch.float32,
        alpha=alpha,
    )

    result = _prepare(processor, "vit", _png(Image.fromarray(raw, mode="RGBA")))

    stored = torch.tensor([10, 20, 30]).float() / 255 * 2 - 1
    transparent = torch.ones(3) if alpha == "white" else stored
    torch.testing.assert_close(result.pixels[:, 0, 0], stored)
    torch.testing.assert_close(result.pixels[:, 0, 31], transparent)
