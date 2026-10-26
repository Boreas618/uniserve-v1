"""The latent codec encodes and decodes images as the model does.

Without graphs the codecs run the model's own encode and decode on each
image's tensors; these tests compare them with the stub model's library
computation on the host.
"""

from __future__ import annotations

import copy

import pytest
import torch

from tests.python.fixtures.stub import load_stub
from uniserve.bindings.workspace import Scratch
from uniserve.executor.image.codec import LatentEncoder, PixelDecoder
from uniserve.media import image
from uniserve.media.pixels import prepare_tensor_image, quantize_image_hwc
from uniserve.nn.vae.layers import DiagonalGaussian

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def stub():
    return load_stub()


def _encoder(module) -> LatentEncoder:
    return LatentEncoder(
        module=module,
        owner="model.latent_encoder",
        stream=None,
        device=-1,
        scratch=Scratch(),
        attention="torch",
        sizes=[],
        pixel_dtype=torch.bfloat16,
        pool=None,
        peers=(),
    )


def _decoder(stub) -> PixelDecoder:
    return PixelDecoder(
        module=stub.module.image_decoder,
        owner="model.image_decoder",
        stream=None,
        device=-1,
        scratch=Scratch(),
        attention="torch",
        sizes=[],
        latent_width=3 * 16 * 16,
        latent_dtype=torch.bfloat16,
        pool=None,
        peers=(),
    )


def test_a_resident_image_encodes_like_its_prepared_pixels(stub):
    module = stub.module.latent_encoder
    # A generated [3, 32, 32] BF16 image product in [-1, 1].
    resident = torch.linspace(-1, 1, 3 * 32 * 32).reshape(3, 32, 32)
    resident = resident.to(torch.bfloat16)
    prepared = prepare_tensor_image(
        stub.image_processor,
        "vae",
        resident,
        device=torch.device("cpu"),
        signed_unit=True,
    )
    out = torch.full((4, 768), float("nan"), dtype=torch.bfloat16)
    _encoder(module).encode(prepared.pixels, out, seed=7, draw=0)
    expected = module.encode(prepared.pixels[None])[0]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_a_sampled_posterior_draws_from_the_request_seed(stub):
    module = copy.copy(stub.module.latent_encoder)
    module.posterior = DiagonalGaussian(sample=True)
    encoder = _encoder(module)
    pixels = torch.rand(3, 32, 32).to(torch.bfloat16)

    def encode(seed, draw):
        out = torch.zeros((4, 768), dtype=torch.bfloat16)
        encoder.encode(pixels, out, seed=seed, draw=draw)
        return out

    first = encode(7, 0)
    assert torch.equal(first, encode(7, 0))
    # Each latent encoding of a request draws anew, and each request apart.
    assert not torch.equal(first, encode(7, 1))
    assert not torch.equal(first, encode(8, 0))


def test_an_encoding_larger_than_its_product_is_refused(stub):
    pixels = torch.rand(3, 32, 32).to(torch.bfloat16)
    with pytest.raises(ValueError, match="exceeds its product"):
        _encoder(stub.module.latent_encoder).encode(
            pixels, torch.zeros(8, dtype=torch.bfloat16), seed=0, draw=0
        )


def test_decoded_images_fill_their_product_and_staged_bytes(stub):
    latents = torch.linspace(-1.2, 1.2, 4 * 768).reshape(4, 768)
    latents = latents.to(torch.bfloat16)
    product = torch.zeros(3, 32, 32, dtype=torch.bfloat16)
    staged = torch.zeros(3 * 32 * 32, dtype=torch.uint8)
    decoder = _decoder(stub)
    decoder.decode(latents, 32, 32, product, staged)

    (decoded,) = stub.module.image_decoder.decode(
        (latents,), sizes=(image.Config(32, 32),)
    )
    expected = quantize_image_hwc(decoded, value_range=(-1.0, 1.0))
    assert torch.equal(staged, expected.reshape(-1))
    torch.testing.assert_close(product, decoded.to(torch.bfloat16))

    # A resident product quantizes like a decoded image.
    again = torch.zeros_like(staged)
    decoder.quantize(product, (-1.0, 1.0), again)
    assert torch.equal(
        again,
        quantize_image_hwc(product, value_range=(-1.0, 1.0)).reshape(-1),
    )
