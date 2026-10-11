"""Image facts, guidance prefixes and image decoding through the runners."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from tests.python.fixtures.stub import load_stub
from uniserve_models import interface
from uniserve_models.bindings.context import Scratch
from uniserve_models.media import image
from uniserve_models.media.pixels import quantize_image_hwc
from uniserve_models.processing import FlowPrompt, resolve_prefix
from uniserve_models.runner.decoder import DecoderRunner
from uniserve_models.runner.denoiser import DenoiserRunner
from uniserve_models.runner.encoder import EncoderRunner

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    table = interface.table()
    return table, load_stub()


def test_the_stub_reports_its_image_facts(loaded):
    _, model = loaded
    assert EncoderRunner.processes_images(model)
    assert EncoderRunner.image_towers(model) == ["vit", "vae"]
    assert EncoderRunner.packs_patches(model) is True
    assert EncoderRunner.pixel_dtype(model) == "bfloat16"
    assert EncoderRunner.feature_injection(model) == [
        "direct",
        "temporal_spatial",
    ]
    # The stub places an end token only.
    assert EncoderRunner.injection_tokens(model) == [-1, 1007]
    # Patch size, latent channels, downsample, framing and longest sequence.
    assert DenoiserRunner.image_layout(model) == [16, 3, 16, 2, 1024]
    assert DenoiserRunner.image_unconditional(model) == "start"
    assert DenoiserRunner.prediction_dtype(model) == "bfloat16"
    # The stub decodes RGB patches in [-1, 1].
    assert DecoderRunner.value_range(model) == [-1.0, 1.0]


def test_patch_grids_follow_the_shared_pixel_budget(loaded):
    _, model = loaded
    assert EncoderRunner.patch_grid(model, 48, 64, 1) == [3, 4]
    with pytest.raises(Exception, match="aspect"):
        EncoderRunner.patch_grid(model, 16, 16 * 400, 1)


def _tokenizer():
    vocabulary = {
        "[UNK]": 0,
        "user": 1,
        "assistant": 2,
        "positive": 3,
        "negative": 4,
        "conditioned": 5,
        "unconditional": 6,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


@pytest.mark.parametrize(
    "source,positive,negative,expected",
    [
        ("conditioning", " positive ", "negative", ((1, 3, 2, 5), False)),
        ("conditioning", "  ", "negative", ((), True)),
        (
            "conditioning",
            "\u001cpositive\u0085",
            "negative",
            ((1, 3, 2, 5), False),
        ),
        ("negative_or_start", "positive", " negative ", ((1, 4, 2, 6), False)),
        ("start", "positive", "negative", ((1, 2, 6), False)),
    ],
)
def test_prefixes_frame_their_source_text(source, positive, negative, expected):
    prompt = FlowPrompt(
        user_prefix="user ",
        user_suffix=" assistant ",
        assistant_suffix="",
        conditioned_append=" conditioned",
        unconditional_append=" unconditional",
    )
    tokenizer = _tokenizer()
    assert (
        resolve_prefix(
            prompt,
            lambda: tokenizer,
            source,
            image_prompt=positive,
            negative_prompt=negative,
        )
        == expected
    )


def test_unframed_models_reuse_their_context_and_refuse_overrides(loaded):
    _, model = loaded
    # ``[reuses, *tokens]``: no tokens either way.
    assert DenoiserRunner.guidance_prefix(model, "conditioning", "", "") == [1]
    assert DenoiserRunner.guidance_prefix(model, "start", "", "") == [0]
    with pytest.raises(Exception, match="prompt override"):
        DenoiserRunner.guidance_prefix(model, "conditioning", "positive", "")


def test_decoded_images_fill_their_products_and_staging(loaded):
    table, model = loaded
    latents = torch.linspace(-1.2, 1.2, 8 * 768).reshape(8, 768)
    latents = latents.to(torch.bfloat16)
    staging = torch.zeros(4 * 32 * 32 * 3, dtype=torch.uint8)
    product = torch.zeros(3, 32, 32, dtype=torch.bfloat16)
    runner = DecoderRunner.image(
        model, "image_decoder", 0, -1, Scratch(), latents, staging
    )

    # Two 32x32 images of four 16-pixel patches each; the second publishes
    # its decoded values as a product.
    counts = runner.decode([0, 4, 32, 32, 0, 4, 4, 32, 32, 3072], None, product)
    assert counts == [3072, 3072]
    decoder = model.module.image_decoder
    for index in range(2):
        (decoded,) = decoder.decode(
            (latents[4 * index : 4 * index + 4],),
            sizes=(image.Config(32, 32),),
        )
        expected = quantize_image_hwc(decoded, value_range=(-1.0, 1.0))
        assert torch.equal(
            staging[3072 * index : 3072 * (index + 1)], expected.reshape(-1)
        )
    assert torch.equal(product, decoded)

    # A resident product quantizes like a decoded image.
    assert runner.quantize(product, -1.0, 1.0, 6144) == 3072
    assert torch.equal(staging[6144:9216], staging[3072:6144])
