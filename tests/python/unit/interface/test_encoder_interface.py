"""Encoder bindings write the stub model's encodings into product regions."""

import base64
import io

import numpy as np
import pytest
import torch
import tvm_ffi
from PIL import Image

from tests.python.fixtures.stub import load_stub
from uniserve_models import interface
from uniserve_models.media.pixels import (
    prepare_host_image,
    prepare_tensor_image,
)
from uniserve_models.model import VisionInput

pytestmark = pytest.mark.unit

# A product arena larger than every product of these tests.
ARENA_BYTES = 1 << 16


@pytest.fixture(scope="module")
def loaded():
    table = interface.table()
    return table, load_stub(table)


def bound(table, model, component, root, path):
    """Bind ``path`` within the ``root`` binding of ``component`` on CPU."""
    prepared, requirements = table["component.prepare"](
        model,
        {
            "component": component,
            "path": root,
            "device": -1,
            "stream": 0,
            "size": None,
            "attention": "auto",
            "vsa": "auto",
            "matmul": "auto",
            "moe": "auto",
            "derive_host_lengths": True,
            "experts": None,
            "weights": None,
            "cache": None,
        },
    )
    assert not requirements["workspaces"]
    table["component.bind"](prepared, {"workspaces": {}})
    arena = torch.zeros(ARENA_BYTES, dtype=torch.uint8)
    module = ".".join(part for part in (root, path) if part)
    info = table["encoder.describe"](model, module, "bfloat16")
    steps = table["encoder.bind"](
        prepared, model, path, tvm_ffi.convert({"products": arena})
    )
    return info, steps, arena


def payload(height, width, seed=0):
    raw = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


def region(offset, dtype, elements):
    return [offset, dtype, [elements]]


def test_vision_features_fill_their_product_regions(loaded):
    table, model = loaded
    info, steps, arena = bound(
        table, model, "vision_encoder", "vision_encoder", ""
    )
    assert info["kind"] == "vision"
    assert (info["patch_size"], info["downsample"]) == (16, 1)
    assert (info["max_patches"], info["pixel_dtype"]) == (0, "bfloat16")

    images = [
        table["image.prepare_input"](model, "vit", payload(32, 48, seed), 2, 0)
        for seed in (1, 2)
    ]
    assert [list(image["grid"]) for image in images] == [[2, 3], [2, 3]]
    out = [region(0, "bfloat16", 4096), region(8192, "bfloat16", 4096)]
    shapes = steps["encode"]([image["image"] for image in images], out)

    processor = model.image_processor
    for index, (image, shape) in enumerate(zip(images, shapes, strict=True)):
        host = prepare_host_image(
            processor, "vit", payload(32, 48, index + 1), input_images=2, pin=0
        )
        (expected,) = model.module.vision_encoder.encode(
            VisionInput(
                (host.pixels,),
                (torch.tensor([host.grid_shape]),),
                (host.grid_shape,),
            )
        )
        assert list(shape) == list(expected.shape)
        written = interface.regions.view(
            arena, [index * 8192, "bfloat16", list(shape)]
        )
        torch.testing.assert_close(written, expected, rtol=0, atol=0)


def test_resident_images_encode_like_their_decoded_pixels(loaded):
    table, model = loaded
    _, steps, arena = bound(table, model, "model", "", "latent_encoder")

    # A generated [3, 32, 32] BF16 image product in [-1, 1].
    pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(3, 32, 32)
    image = interface.regions.view(arena, [0, "bfloat16", [3, 32, 32]])
    image.copy_(pixels)
    prepared = steps["prepare_resident"]([0, "bfloat16", [3, 32, 32]], "vae", 1)
    assert (prepared["height"], prepared["width"]) == (32, 32)
    assert prepared["grid"] is None

    (shape,) = steps["encode"](
        [prepared["image"]], [region(8192, "bfloat16", 4096)]
    )
    decoded = prepare_tensor_image(
        model.image_processor,
        "vae",
        image,
        device=torch.device("cpu"),
        signed_unit=True,
    )
    expected = model.module.latent_encoder.encode(decoded.pixels[None])[0]
    assert list(shape) == list(expected.shape) == [4, 3 * 16 * 16]
    written = interface.regions.view(arena, [8192, "bfloat16", list(shape)])
    torch.testing.assert_close(written, expected, rtol=0, atol=0)


def test_products_beyond_their_region_are_refused(loaded):
    table, model = loaded
    _, steps, _ = bound(table, model, "vision_encoder", "vision_encoder", "")
    image = table["image.prepare_input"](model, "vit", payload(32, 48), 1, 0)
    with pytest.raises(Exception, match="exceeds its product bound"):
        steps["encode"]([image["image"]], [region(0, "bfloat16", 8)])


def test_inline_images_are_validated_on_the_host(loaded):
    table, model = loaded
    with pytest.raises(Exception, match="not a valid encoded image"):
        table["image.prepare_input"](model, "vit", "bm90IGFuIGltYWdl", 1, 0)
    with pytest.raises(Exception, match="no input images"):
        table["image.prepare_input"](model, "vit", payload(16, 16), 0, 0)
