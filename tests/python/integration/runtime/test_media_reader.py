"""The media reader decodes MiniMax-H3 still and audio conditions exactly.

Each case reads the media of a recorded diffusers reference request through
``uniserve.media.reader`` with the sizes the reference conditioned it at, as
the serving worker's host reader does with the sizes the server plans. The
reader applies the reference conditioning's own operations in its order, so
every result is compared bit for bit:

- an image with the diffusers pipeline's prepared image: PIL decoding and a
  LANCZOS resize (and, for a later keyframe, the cover crop);
- a soundtrack with the diffusers normalization of the reference's own
  decode: PyAV planar float at the native rate, truncation there, and one
  torchaudio resample to 32 kHz.

The cases need the reference artifacts (``UNISERVE_MINIMAX_H3_REFERENCE``,
holding ``inputs/media`` and ``reference/diffusers``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from uniserve.media.reader import read_audio, read_image

pytestmark = [
    pytest.mark.integration,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

# The model's audio rate and frame rate.
SAMPLE_RATE = 32_000
FPS = 24


@dataclass(frozen=True, slots=True)
class ImageRead:
    """An image resized to ``resized`` and read in ``window``.

    Sizes are ``(width, height)``; the window is ``(left, top, width,
    height)`` within the resized image.
    """

    resized: tuple[int, int]
    window: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class AudioRead:
    """A soundtrack's native-rate span and the samples it resamples to."""

    native_rate: int
    start_sample: int
    source_samples: int
    samples: int


# The 1344 x 768 canvas every recorded case generates.
_CANVAS = ImageRead((1344, 768), (0, 0, 1344, 768))

# What each recorded run conditioned on, as its metadata.json records: the
# frame count and, in request order, how every condition was prepared. A
# first keyframe is stretched onto the canvas; the later clay-fox keyframe
# covers it at 1376 x 768 and its centre 1344 columns are read. The image
# reference is encoded at its 2048-pixel short edge (3648 x 2048, 7296
# condition rows). The official voice is 44.1 kHz: the 124 frames of five
# seconds keep its first 227,850 native samples, which resample to 165,334
# samples at 32 kHz (two tracks of 207 audio latents).
_CASES = {
    "fl2va_first_8s": (192, (_CANVAS,)),
    "fl2va_first_last_8s": (
        192,
        (_CANVAS, ImageRead((1376, 768), (16, 0, 1344, 768))),
    ),
    "ref2va_image_audio_5s": (
        124,
        (
            ImageRead((3648, 2048), (0, 0, 3648, 2048)),
            AudioRead(44_100, 0, 227_850, 165_334),
        ),
    ),
}


@pytest.fixture(scope="module")
def root() -> Path:
    value = os.environ.get("UNISERVE_MINIMAX_H3_REFERENCE")
    if not value:
        pytest.skip("UNISERVE_MINIMAX_H3_REFERENCE names no reference run")
    return Path(value)


def _paths(root: Path, case: str) -> list[Path]:
    """The media file of each condition of the recorded request."""
    run = root / "reference" / "diffusers" / case / "seed42"
    request = json.loads((run / "request.json").read_text())
    return [
        root / "inputs" / "media" / Path(item["uri"]).name
        for item in request["conditions"]
    ]


@pytest.mark.parametrize("case", list(_CASES))
def test_conditions_read_as_the_reference_conditioning(case, root):
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines.minimax_h3 import references
    from diffusers.modular_pipelines.minimax_h3.before_encoder import (
        MiniMaxH3Ref2VASetupStep,
    )
    from diffusers.utils import load_image

    num_frames, reads = _CASES[case]
    paths = _paths(root, case)
    resize = VaeImageProcessor(vae_scale_factor=16).resize
    for read, path in zip(reads, paths, strict=True):
        if isinstance(read, AudioRead):
            samples = read_audio(
                path.read_bytes(),
                native_rate=read.native_rate,
                start=read.start_sample,
                source_samples=read.source_samples,
                samples=read.samples,
                rate=SAMPLE_RATE,
            )
            # The diffusers setup step normalizes the decoded track,
            # truncated to the generated duration.
            waveform, rate = references._decode_audio_file(path)
            expected = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
                waveform,
                rate,
                SAMPLE_RATE,
                max_duration=num_frames / FPS,
            )
            assert torch.equal(samples, expected.T)
            continue

        pixels = read_image(
            path.read_bytes(), resized=read.resized, window=read.window
        )
        # The pipeline resizes the decoded file with PIL's LANCZOS filter,
        # then crops a later keyframe onto the canvas.
        left, top, width, height = read.window
        expected = np.array(
            resize(
                load_image(str(path)),
                height=read.resized[1],
                width=read.resized[0],
            )
        )[top : top + height, left : left + width]
        assert np.array_equal(pixels[0], expected)


def test_tracks_that_decode_to_another_extent_are_refused(root):
    _, reads = _CASES["ref2va_image_audio_5s"]
    clip = reads[1]
    path = _paths(root, "ref2va_image_audio_5s")[1]
    with pytest.raises(ValueError, match="Hz"):
        read_audio(
            path.read_bytes(),
            native_rate=clip.native_rate + 1,
            start=clip.start_sample,
            source_samples=clip.source_samples,
            samples=clip.samples,
            rate=SAMPLE_RATE,
        )
    with pytest.raises(ValueError, match="planned"):
        read_audio(
            path.read_bytes(),
            native_rate=clip.native_rate,
            start=clip.start_sample,
            source_samples=clip.source_samples,
            samples=clip.samples + 1,
            rate=SAMPLE_RATE,
        )
