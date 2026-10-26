"""A model's image facts and the guidance prefixes its framing resolves."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from tests.python.fixtures.stub import load_stub
from uniserve.executor.image import text_settings
from uniserve.executor.image.denoiser import framed_prefix
from uniserve.executor.image.facts import ImageFacts
from uniserve.processing import BranchSource, FlowPrompt, resolve_prefix

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    return load_stub()


def test_the_stub_reports_its_image_facts(loaded):
    facts = ImageFacts.from_loaded(loaded)
    layout = facts.layout
    assert layout is not None
    # Patch size, latent channels, downsample, framing, longest sequence,
    # the language positions a generated image spans and the most guidance
    # branches a step evaluates (every guidance branch).
    assert (
        layout.patch_size,
        layout.latent_channels,
        layout.downsample,
        layout.framing_tokens,
        layout.max_sequence_tokens,
        layout.rope_advance,
        layout.guidance_branches,
    ) == (16, 3, 16, 2, 1024, 2, 3)
    assert layout.image_unconditional is BranchSource.START
    # The stub places features at temporal-spatial positions, ending them
    # with its end token only.
    assert facts.spatial_positions()
    assert facts.injection is not None
    assert (facts.injection.start_token, facts.injection.end_token) == (
        None,
        1007,
    )
    # The text executor's settings: an image policy, three rotary axes.
    assert text_settings(loaded) == (
        '{"image_processor": true, "media_axes": 3}'
    )


def test_patch_grids_follow_the_shared_pixel_budget(loaded):
    facts = ImageFacts.from_loaded(loaded)
    assert facts.grid(48, 64, False, 1) == (3, 4)
    with pytest.raises(Exception, match="aspect"):
        facts.grid(16, 16 * 400, False, 1)


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
    assert framed_prefix(loaded, BranchSource.START) == ()
    assert framed_prefix(loaded, BranchSource.NEGATIVE_OR_START) == ()
