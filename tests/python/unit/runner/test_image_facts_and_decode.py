"""Image facts, guidance prefixes and image decoding through the runners."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from tests.python.fixtures.stub import load_stub
from uniserve_models import interface
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
    return table, load_stub(table)


def test_the_stub_reports_its_image_facts(loaded):
    _, model = loaded
    facts = DenoiserRunner.describe(model)
    processor = facts["processor"]
    assert (processor["vit"], processor["vae"]) == (True, True)
    assert processor["patch_tower"] is True
    assert processor["pixel_dtype"] == "bfloat16"
    assert dict(processor["injection"]) == {
        "layout": "direct",
        "positions": "temporal_spatial",
        "start_token": None,
        "end_token": 1007,
    }
    assert dict(facts["denoiser"]) == {
        "patch_size": 16,
        "latent_channels": 3,
        "downsample": 16,
        "framing_tokens": 2,
        "image_unconditional": "start",
        "max_sequence_tokens": 1024,
        "prediction_dtype": "bfloat16",
    }
    # The stub decodes RGB patches in [-1, 1].
    assert list(facts["decoder_range"]) == [-1.0, 1.0]


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
    prepared, _ = table["component.prepare"](
        model,
        {
            "component": "model",
            "path": "",
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
    table["component.bind"](prepared, {"workspaces": {}})
    latents = torch.linspace(-1.2, 1.2, 8 * 768).reshape(8, 768)
    latents = latents.to(torch.bfloat16)
    staging = torch.zeros(4 * 32 * 32 * 3, dtype=torch.uint8)
    arena = torch.zeros(1 << 16, dtype=torch.uint8)
    runner = DecoderRunner.bound(
        prepared, "image_decoder", latents, staging, arena
    )

    # Two 32x32 images of four 16-pixel patches each; the second publishes
    # its decoded values as a product.
    counts = runner.decode(
        [0, 4, 32, 32, 0, 4, 4, 32, 32, 3072],
        [None, [0, "bfloat16", [3, 32, 32]]],
    )
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
    product = interface.regions.view(arena, [0, "bfloat16", [3, 32, 32]])
    assert torch.equal(product, decoded)

    # A resident product quantizes like a decoded image.
    assert (
        runner.quantize([0, "bfloat16", [3, 32, 32]], -1.0, 1.0, 6144) == 3072
    )
    assert torch.equal(staging[6144:9216], staging[3072:6144])
