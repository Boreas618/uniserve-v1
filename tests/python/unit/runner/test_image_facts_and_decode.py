"""Image facts, guidance prefixes and image decoding through the runners."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from tests.python.fixtures.stub import load_stub
from uniserve.bindings.workspace import Scratch
from uniserve.media import image
from uniserve.media.pixels import quantize_image_hwc
from uniserve.processing import FlowPrompt, resolve_prefix
from uniserve.runner.decoder import DecoderRunner
from uniserve.runner.denoiser import DenoiserRunner
from uniserve.runner.encoder import EncoderRunner

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    return load_stub()


def test_the_stub_reports_its_image_facts(loaded):
    model = loaded
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
    # Patch size, latent channels, downsample, framing, longest sequence,
    # the language positions a generated image spans and the most guidance
    # branches a step evaluates (every guidance branch).
    assert DenoiserRunner.image_layout(model) == [16, 3, 16, 2, 1024, 2, 3]
    assert DenoiserRunner.image_unconditional(model) == "start"
    assert DenoiserRunner.prediction_dtype(model) == "bfloat16"
    # The stub decodes RGB patches in [-1, 1].
    assert DecoderRunner.value_range(model) == [-1.0, 1.0]


def test_patch_grids_follow_the_shared_pixel_budget(loaded):
    model = loaded
    assert EncoderRunner.patch_grid(model, 48, 64, 1) == [3, 4]
    with pytest.raises(Exception, match="aspect"):
        EncoderRunner.patch_grid(model, 16, 16 * 400, 1)


def _tokenizer():
    vocabulary = {
        "[UNK]": 0,
        "user": 1,
        "assistant": 2,
        "negative": 4,
        "unconditional": 6,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


@pytest.mark.parametrize(
    "source,negative,expected",
    [
        ("negative_or_start", " negative ", (1, 4, 2, 6)),
        ("negative_or_start", "\u001cnegative\u0085", (1, 4, 2, 6)),
        ("start", "negative", (1, 2, 6)),
    ],
)
def test_prefixes_frame_their_source_text(source, negative, expected):
    prompt = FlowPrompt(
        user_prefix="user ",
        user_suffix=" assistant ",
        assistant_suffix="",
        unconditional_append=" unconditional",
    )
    tokenizer = _tokenizer()
    assert (
        resolve_prefix(
            prompt, lambda: tokenizer, source, negative_prompt=negative
        )
        == expected
    )


def test_the_conditioned_branch_has_no_prefix():
    with pytest.raises(ValueError, match="no prefix"):
        resolve_prefix(None, lambda: None, "conditioning", negative_prompt="")


def test_unframed_models_leave_their_prefixes_empty(loaded):
    model = loaded
    assert DenoiserRunner.guidance_prefix(model, "start") == []
    assert DenoiserRunner.guidance_prefix(model, "negative_or_start") == []


def test_decoded_images_fill_their_products_and_staging(loaded):
    model = loaded
    latents = torch.linspace(-1.2, 1.2, 8 * 768).reshape(8, 768)
    latents = latents.to(torch.bfloat16)
    staging = torch.zeros(4 * 32 * 32 * 3, dtype=torch.uint8)
    product = torch.zeros(3, 32, 32, dtype=torch.bfloat16)
    inputs = torch.zeros(4, 768, dtype=torch.bfloat16)
    values = torch.zeros(3 * 32 * 32, dtype=torch.bfloat16)
    runner = DecoderRunner.image(
        model,
        "image_decoder",
        0,
        -1,
        Scratch(),
        latents,
        staging,
        inputs,
        values,
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

    # One image at a time, as a captured decode replays it: the rows copied
    # into the inputs decode into the values and the head of the staging
    # bytes, and values copied in quantize there.
    staged = staging[3072:6144].clone()
    staging.zero_()
    inputs.copy_(latents[4:8])
    assert runner.decode_one(4, 32, 32) == 3072
    assert torch.equal(values.view(3, 32, 32), decoded)
    assert torch.equal(staging[:3072], staged)
    staging.zero_()
    assert runner.quantize_one(32, 32, -1.0, 1.0) == 3072
    assert torch.equal(staging[:3072], staged)
