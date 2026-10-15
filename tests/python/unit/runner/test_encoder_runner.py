"""An image model's encoders write their encodings into product tensors."""

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from tests.python.fixtures.stub import load_stub
from uniserve_models import interface
from uniserve_models.bindings.context import Scratch
from uniserve_models.media.pixels import (
    prepare_host_image,
    prepare_tensor_image,
)
from uniserve_models.model import VisionInput
from uniserve_models.runner.encoder import EncoderRunner

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    table = interface.table()
    return table, load_stub()


def constructed(model, module):
    """A runner of the encoder at ``module`` of the stub, on the host."""
    runner = EncoderRunner.image(model, module, 0, -1, Scratch(), [])
    return (model, module), runner


def payload(height, width, seed=0):
    raw = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


def product(elements):
    """A BF16 product tensor of ``elements`` values."""
    return torch.zeros(elements, dtype=torch.bfloat16)


def test_vision_features_fill_their_products(loaded):
    table, model = loaded
    info, runner = constructed(model, "vision_encoder")
    assert EncoderRunner.kind(*info) == "vision"
    assert EncoderRunner.patch_size(*info) == 16
    assert EncoderRunner.downsample(*info) == 1
    assert EncoderRunner.max_patches(*info) == 0
    assert EncoderRunner.pixel_dtype(info[0]) == "bfloat16"

    images = [
        EncoderRunner.prepare_image(model, "vit", payload(32, 48, seed), 2, 0)
        for seed in (1, 2)
    ]
    # Canvas height and width, then the patch grid.
    assert [EncoderRunner.image_facts(image)[:4] for image in images] == [
        [32, 48, 2, 3],
        [32, 48, 2, 3],
    ]
    out = [product(4096), product(4096)]
    shapes = runner.encode(images, *out)

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
        written = out[index][: expected.numel()].reshape(list(shape))
        torch.testing.assert_close(written, expected, rtol=0, atol=0)


def test_resident_images_encode_like_their_decoded_pixels(loaded):
    table, model = loaded
    _, runner = constructed(model, "latent_encoder")

    # A generated [3, 32, 32] BF16 image product in [-1, 1].
    pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(3, 32, 32)
    image = pixels.to(torch.bfloat16)
    prepared = runner.prepare_resident(image, "vae", 1)
    # A latent tower has no patch grid.
    assert EncoderRunner.image_facts(prepared)[:4] == [32, 32, 0, 0]

    out = product(4096)
    (shape,) = runner.encode([prepared], out)
    decoded = prepare_tensor_image(
        model.image_processor,
        "vae",
        image,
        device=torch.device("cpu"),
        signed_unit=True,
    )
    expected = model.module.latent_encoder.encode(decoded.pixels[None])[0]
    assert list(shape) == list(expected.shape) == [4, 3 * 16 * 16]
    written = out[: expected.numel()].reshape(list(shape))
    torch.testing.assert_close(written, expected, rtol=0, atol=0)


def test_encodings_larger_than_their_product_are_refused(loaded):
    table, model = loaded
    _, runner = constructed(model, "vision_encoder")
    image = EncoderRunner.prepare_image(model, "vit", payload(32, 48), 1, 0)
    with pytest.raises(Exception, match="exceeds its product bound"):
        runner.encode([image], product(8))


def test_inline_images_are_validated_on_the_host(loaded):
    _, model = loaded
    with pytest.raises(Exception, match="not a valid encoded image"):
        EncoderRunner.prepare_image(model, "vit", "bm90IGFuIGltYWdl", 1, 0)
    with pytest.raises(Exception, match="no input images"):
        EncoderRunner.prepare_image(model, "vit", payload(16, 16), 0, 0)
