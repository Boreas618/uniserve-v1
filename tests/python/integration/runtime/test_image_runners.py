"""The image runners drive the stub model through the numerical interface.

The worker's image harness (``tests.python.fixtures.image_harness``) binds
one component on the host, runs one runner operation and returns its
Rust-owned results, which these tests compare with direct library
computation.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
import tvm_ffi
from PIL import Image

from tests.python.fixtures.image_harness import image_harness
from tests.python.fixtures.stub import load_stub
from uniserve_models import interface
from uniserve_models.media import image
from uniserve_models.media.pixels import prepare_host_image, quantize_image_hwc
from uniserve_models.model import VisionInput

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def stub():
    table = interface.table()
    return table, load_stub(table)


def _payload(height: int, width: int, seed: int) -> str:
    raw = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


def _encode(stub, component, root, path, tower, payloads):
    table, model = stub
    *results, counts = image_harness().encode(
        tvm_ffi.convert(
            {
                "table": table,
                "loaded": model,
                "component": component,
                "root": root,
                "path": path,
                "tower": tower,
                "payloads": payloads,
            }
        )
    )
    return [torch.from_dlpack(value) for value in results], list(counts)


def test_vision_features_match_the_encoder_one_shape_per_call(stub):
    _, model = stub
    payloads = [_payload(32, 48, 1), _payload(48, 48, 2), _payload(32, 48, 3)]
    features, (replays, eager) = _encode(
        stub, "vision_encoder", "vision_encoder", "", "vit", payloads
    )

    # Two pixel shapes: two numerical calls, in first-appearance order.
    assert (replays, eager) == (0, 2)
    for payload, actual in zip(payloads, features, strict=True):
        host = prepare_host_image(
            model.image_processor, "vit", payload, input_images=3, pin=False
        )
        (expected,) = model.module.vision_encoder.encode(
            VisionInput(
                (host.pixels,),
                (torch.tensor([host.grid_shape]),),
                (host.grid_shape,),
            )
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_latent_encodings_match_the_autoencoder(stub):
    _, model = stub
    payloads = [_payload(32, 32, 4), _payload(32, 32, 5)]
    latents, (_, eager) = _encode(
        stub, "model", "", "latent_encoder", "vae", payloads
    )
    assert eager == 1
    for payload, actual in zip(payloads, latents, strict=True):
        host = prepare_host_image(
            model.image_processor, "vae", payload, input_images=2, pin=False
        )
        expected = model.module.latent_encoder.encode(host.pixels[None])[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decoded_images_stage_quantized_pixels_across_sub_batches(stub):
    table, model = stub
    latents = torch.linspace(-1.5, 1.5, 16 * 768).reshape(16, 768)
    latents = latents.to(torch.bfloat16)
    sizes = [[32, 32], [32, 64], [32, 32]]
    # Staging holds one 32x64 image, so the batch decodes in three calls.
    results = image_harness().decode(
        tvm_ffi.convert(
            {
                "table": table,
                "loaded": model,
                "component": "model",
                "root": "",
                "path": "image_decoder",
                "latents": latents,
                "sizes": sizes,
                "staging_bytes": 32 * 64 * 3,
            }
        )
    )
    decoder = model.module.image_decoder
    first = 0
    for index, (height, width) in enumerate(sizes):
        rows = height // 16 * (width // 16)
        (decoded,) = decoder.decode(
            (latents[first : first + rows],),
            sizes=(image.Config(height, width),),
        )
        first += rows
        pixels = torch.from_dlpack(results[2 * index])
        product = torch.from_dlpack(results[2 * index + 1])
        expected = quantize_image_hwc(decoded, value_range=(-1.0, 1.0))
        assert torch.equal(pixels, expected.reshape(-1))
        assert torch.equal(product, decoded)
