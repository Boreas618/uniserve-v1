"""Inline images are prepared on the host into worker storage."""

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from tests.python.fixtures.stub import load_stub
from uniserve.media.pixels import prepare_host_image
from uniserve.runner.encoder import EncoderRunner

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    return load_stub()


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
    """An inline image prepared into storage as a host lane prepares it.

    Its size, its facts, then its pixels.
    """
    height, width = EncoderRunner.image_size(encoded)
    facts = EncoderRunner.image_facts(model, tower, height, width, input_images)
    pixels = storage(model, facts)
    EncoderRunner.prepare_image(model, tower, encoded, input_images, pixels)
    return facts, pixels


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


def test_prepared_pixels_are_the_policys_host_preparation(loaded):
    model = loaded
    encoded = payload(32, 48, 1)
    facts, pixels = prepared(model, "vit", encoded, 2)
    # Canvas height and width, then the patch grid and the pixel shape.
    assert facts[:4] == [32, 48, 2, 3]
    host = prepare_host_image(
        model.image_processor, "vit", encoded, input_images=2, pin=False
    )
    torch.testing.assert_close(pixels, host.pixels, rtol=0, atol=0)
    assert EncoderRunner.pixel_dtype(model) == "bfloat16"
