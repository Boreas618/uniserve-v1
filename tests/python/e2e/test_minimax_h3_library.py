"""The MiniMax-H3 library generates a clip from a diffusers root eagerly.

Loads the base checkpoint's text encoder, denoiser, decoders and
post-processor on one GPU as a rank loads them
(``uniserve.loading.rank``), tokenizes a t2va prompt, takes the frame count
of four seconds and the 9:16 canvas from the model package's rules, and runs
``generation.generate``, which drives the runners serving uses. The clip has
those frames at that canvas and a stereo track covering them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.python.fixtures.stub import LOCAL, load_model, read_config
from uniserve.models.minimax_h3 import processing
from uniserve.models.minimax_h3.generation import generate

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]

# The components generation runs, each on this rank alone, and the module
# paths they hold.
_COMPONENTS = ("text_encoder", "transformer", "video_decoder", "audio_decoder")
_MODULES = [
    "text_encoder",
    "transformer",
    "video_decoder",
    "video_postprocessor",
    "audio_decoder",
]


def test_library_generates_a_portrait_clip() -> None:
    root = os.environ.get("UNISERVE_MINIMAX_H3_MODEL")
    if not root or not Path(root).is_dir():
        pytest.fail(
            "UNISERVE_MINIMAX_H3_MODEL must name a MiniMax-H3 diffusers root"
        )
    transformers = pytest.importorskip("transformers")

    # Serving's defaults: the checkpoint's own precision at BF16 and a
    # 512 MiB FlashInfer workspace.
    description = read_config(
        stub=False,
        path=root,
        device="cuda:0",
        components=json.dumps(dict.fromkeys(_COMPONENTS, LOCAL)),
    )
    loaded = load_model(
        description,
        modules=_MODULES,
        device="cuda:0",
        workspace_size=512 << 20,
    )

    # A t2va presentation is the prompt alone, tokenized without special
    # tokens.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        Path(root) / "tokenizer"
    )
    prompt = tokenizer.encode(
        "A lighthouse keeper climbs a spiral staircase at dusk while waves "
        "crash against the rocks below.",
        add_special_tokens=False,
    )
    num_frames = processing.frame_count(4.0)
    canvas = processing.canvas(9, 16)
    generation = generate(
        loaded,
        prompt_token_ids=prompt,
        num_frames=num_frames,
        canvas=canvas,
        seed=7,
    )

    # Four seconds at 24 fps align up to 107 frames of the 768x1344 canvas.
    assert (canvas.width, canvas.height) == (768, 1344)
    assert num_frames == 107
    assert generation.frames.shape == (107, 1344, 768, 3)
    assert generation.frames.dtype == torch.uint8
    # The stereo track spans the clip at the audio decoder's 32 kHz.
    samples, channels = generation.audio.shape
    assert channels == 2
    assert abs(samples / 32_000 - 107 / 24) <= 1 / 24
    assert generation.frames.float().std() > 0
    assert generation.audio.float().abs().max() > 0
