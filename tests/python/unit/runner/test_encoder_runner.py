"""Image encoders prepare images into worker storage and fill products."""

import base64
import io
from math import prod

import numpy as np
import pytest
import torch
from PIL import Image

from tests.python.fixtures.stub import load_stub
from uniserve.bindings.context import Scratch
from uniserve.media.pixels import (
    prepare_host_image,
    prepare_tensor_image,
)
from uniserve.models import VisionInput
from uniserve.runner.encoder import EncoderRunner

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    return load_stub()


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


def storage(model, facts):
    """Pixel storage of the shape ``facts`` report, as the worker leases."""
    dtype = getattr(torch, EncoderRunner.pixel_dtype(model))
    return torch.full(facts[4:], float("nan"), dtype=dtype)


def prepared(model, tower, encoded, input_images):
    """An inline image prepared into storage as a host lane prepares it:
    its size, its facts, then its pixels.
    """  # noqa: D205
    height, width = EncoderRunner.image_size(encoded)
    facts = EncoderRunner.image_facts(model, tower, height, width, input_images)
    pixels = storage(model, facts)
    EncoderRunner.prepare_image(model, tower, encoded, input_images, pixels)
    return facts, pixels


def product(elements):
    """A BF16 product tensor of ``elements`` values."""
    return torch.zeros(elements, dtype=torch.bfloat16)


def test_vision_features_fill_their_products(loaded):
    model = loaded
    info, runner = constructed(model, "vision_encoder")
    assert EncoderRunner.kind(*info) == "vision"
    assert EncoderRunner.patch_size(*info) == 16
    assert EncoderRunner.downsample(*info) == 1
    assert EncoderRunner.max_patches(*info) == 0
    assert EncoderRunner.pixel_dtype(info[0]) == "bfloat16"

    images = [
        prepared(model, "vit", payload(32, 48, seed), 2) for seed in (1, 2)
    ]
    # Canvas height and width, then the patch grid.
    assert [facts[:4] for facts, _ in images] == [
        [32, 48, 2, 3],
        [32, 48, 2, 3],
    ]
    grids = [extent for facts, _ in images for extent in facts[2:4]]
    out = [product(4096), product(4096)]
    shapes = runner.encode(grids, *(pixels for _, pixels in images), *out)

    processor = model.image_processor
    for index, ((_, pixels), shape) in enumerate(
        zip(images, shapes, strict=True)
    ):
        host = prepare_host_image(
            processor, "vit", payload(32, 48, index + 1), input_images=2, pin=0
        )
        torch.testing.assert_close(pixels, host.pixels, rtol=0, atol=0)
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
    model = loaded
    _, runner = constructed(model, "latent_encoder")

    # A generated [3, 32, 32] BF16 image product in [-1, 1].
    pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(3, 32, 32)
    image = pixels.to(torch.bfloat16)
    facts = EncoderRunner.image_facts(model, "vae", 32, 32, 0)
    # A latent tower has no patch grid.
    assert facts[:4] == [32, 32, 0, 0]
    resident = storage(model, facts)
    runner.prepare_resident(image, "vae", 1, resident)

    out = product(4096)
    (shape,) = runner.encode(resident, out)
    decoded = prepare_tensor_image(
        model.image_processor,
        "vae",
        image,
        device=torch.device("cpu"),
        signed_unit=True,
    )
    torch.testing.assert_close(resident, decoded.pixels, rtol=0, atol=0)
    expected = model.module.latent_encoder.encode(decoded.pixels[None])[0]
    assert list(shape) == list(expected.shape) == [4, 3 * 16 * 16]
    written = out[: expected.numel()].reshape(list(shape))
    torch.testing.assert_close(written, expected, rtol=0, atol=0)


def test_posterior_noise_is_drawn_from_the_request_seed(loaded):
    model = loaded
    _, runner = constructed(model, "latent_encoder")
    elements = prod(model.module.latent_encoder.posterior_shape(32, 32))

    def draw(seed, index):
        noise = torch.zeros(elements + 64)
        runner.draw_noise(seed, index, 32, 32, noise)
        return noise

    first = draw(7, 0)
    assert torch.equal(first, draw(7, 0))
    assert not torch.equal(first, draw(7, 1))
    assert not torch.equal(first, draw(8, 0))
    # Only the posterior's leading elements are drawn.
    assert torch.count_nonzero(first[elements:]) == 0
    assert torch.count_nonzero(first[:elements]) == elements


def test_a_posterior_mean_takes_no_noise(loaded):
    model = loaded
    _, runner = constructed(model, "latent_encoder")
    facts, pixels = prepared(model, "vae", payload(32, 32), 1)
    noise = torch.zeros(
        prod(model.module.latent_encoder.posterior_shape(32, 32))
    )
    with pytest.raises(Exception, match="posterior mean takes none"):
        runner.encode(pixels, product(4096), noise)


def test_encodings_larger_than_their_product_are_refused(loaded):
    model = loaded
    _, runner = constructed(model, "vision_encoder")
    facts, pixels = prepared(model, "vit", payload(32, 48), 1)
    with pytest.raises(Exception, match="exceeds its product bound"):
        runner.encode(facts[2:4], pixels, product(8))


def test_inline_images_are_validated_on_the_host(loaded):
    model = loaded
    with pytest.raises(Exception, match="not a valid encoded image"):
        EncoderRunner.image_size("bm90IGFuIGltYWdl")
    facts = EncoderRunner.image_facts(model, "vit", 16, 16, 1)
    with pytest.raises(Exception, match="no input images"):
        EncoderRunner.prepare_image(
            model, "vit", payload(16, 16), 0, storage(model, facts)
        )
    # Storage that is not the image's prepared shape is refused.
    with pytest.raises(Exception, match="disagree"):
        EncoderRunner.prepare_image(
            model,
            "vit",
            payload(16, 16),
            1,
            torch.zeros(1, dtype=torch.bfloat16),
        )
