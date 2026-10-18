"""The media reader decodes MiniMax-H3 still and audio conditions exactly.

Each case plans a recorded diffusers reference request from its media's
probed facts (``processing.plan_request``, the planner the server's rules
are checked against) and reads every image and soundtrack through
``uniserve.media.reader`` with the plan's values, as the serving
worker's host reader does. The reader applies the reference conditioning's
own operations in its order, so every result is compared bit for bit:

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
from pathlib import Path

import numpy as np
import pytest
import torch

from uniserve.media.reader import read_audio, read_image
from uniserve.models.minimax_h3 import processing

pytestmark = [
    pytest.mark.integration,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

# The model's audio rate and frame rate.
SAMPLE_RATE = 32_000
FPS = 24

_VISION = processing.VisionConfig(
    **json.loads(
        (
            Path(__file__).parents[2] / "fixtures" / "minimax_h3_plan.json"
        ).read_text()
    )["vision"]
)


@pytest.fixture(scope="module")
def root() -> Path:
    value = os.environ.get("UNISERVE_MINIMAX_H3_REFERENCE")
    if not value:
        pytest.skip("UNISERVE_MINIMAX_H3_REFERENCE names no reference run")
    return Path(value)


def _media(root: Path, uri: str) -> Path:
    return root / "inputs" / "media" / Path(uri).name


def _facts(path: Path, kind: processing.ConditionType):
    from diffusers.modular_pipelines.minimax_h3 import references
    from PIL import Image, ImageOps

    if kind is processing.ConditionType.IMAGE:
        with Image.open(path) as value:
            width, height = ImageOps.exif_transpose(value).size
        return processing.ImageFacts(width, height)
    if kind is processing.ConditionType.AUDIO:
        waveform, rate = references._decode_audio_file(path)
        return processing.AudioFacts(rate, int(waveform.shape[1]))
    raise AssertionError("the cases condition on images and audio only")


def _plan(root: Path, case: str):
    """The recorded request's plan and each condition's media file."""
    run = root / "reference" / "diffusers" / case / "seed42"
    request = json.loads((run / "request.json").read_text())
    conditions, paths = [], []
    for item in request["conditions"]:
        kind = processing.ConditionType(item["type"])
        path = _media(root, item["uri"])
        paths.append(path)
        conditions.append(
            processing.Condition(
                type=kind,
                role=processing.Role(item["role"]),
                media=_facts(path, kind),
                frame_index=item.get("frame_index"),
            )
        )
    # The recorded runs come from the reference pipeline, which enlarges
    # small references.
    planned = processing.plan_request(
        processing.Task(request["task"]),
        processing.Target(**request["target"]),
        conditions,
        _VISION,
        references=processing.ReferenceConfig(enlarge=True),
    )
    return planned, paths


def _fit(prepared, canvas) -> tuple[tuple[int, int], tuple[int, ...]]:
    """The reader's ``resized`` and ``window`` of a planned image."""
    if isinstance(prepared, processing.KeyframeFit):
        crop = prepared.cover_crop
        if crop is None:
            return (canvas.width, canvas.height), (
                0,
                0,
                canvas.width,
                canvas.height,
            )
        return (crop.width, crop.height), (
            crop.left,
            crop.top,
            canvas.width,
            canvas.height,
        )
    return (prepared.width, prepared.height), (
        0,
        0,
        prepared.width,
        prepared.height,
    )


@pytest.mark.parametrize(
    "case",
    ["fl2va_first_8s", "fl2va_first_last_8s", "ref2va_image_audio_5s"],
)
def test_conditions_read_as_the_reference_conditioning(case, root):
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines.minimax_h3 import references
    from diffusers.modular_pipelines.minimax_h3.before_encoder import (
        MiniMaxH3Ref2VASetupStep,
    )
    from diffusers.utils import load_image

    planned, paths = _plan(root, case)
    resize = VaeImageProcessor(vae_scale_factor=16).resize
    checked = 0
    for condition, path in zip(planned.conditions, paths, strict=True):
        prepared = condition.prepared
        if isinstance(prepared, processing.AudioClip):
            samples = read_audio(
                path.read_bytes(),
                native_rate=prepared.sample_rate,
                start=prepared.start_sample,
                source_samples=prepared.source_samples,
                samples=prepared.samples,
                rate=SAMPLE_RATE,
            )
            # The diffusers setup step normalizes the decoded track,
            # truncated to the generated duration.
            waveform, rate = references._decode_audio_file(path)
            expected = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
                waveform,
                rate,
                SAMPLE_RATE,
                max_duration=planned.num_frames / FPS,
            )
            assert torch.equal(samples, expected.T)
        else:
            resized, window = _fit(prepared, planned.canvas)
            pixels = read_image(
                path.read_bytes(), resized=resized, window=window
            )
            # The pipeline resizes the decoded file with PIL's LANCZOS
            # filter, then crops a later keyframe onto the canvas.
            left, top, width, height = window
            expected = np.array(
                resize(
                    load_image(str(path)), height=resized[1], width=resized[0]
                )
            )[top : top + height, left : left + width]
            assert np.array_equal(pixels[0], expected)
        checked += 1
    assert checked == len(paths)


def test_tracks_that_decode_to_another_extent_are_refused(root):
    planned, paths = _plan(root, "ref2va_image_audio_5s")
    clip = planned.conditions[1].prepared
    with pytest.raises(ValueError, match="Hz"):
        read_audio(
            paths[1].read_bytes(),
            native_rate=clip.sample_rate + 1,
            start=clip.start_sample,
            source_samples=clip.source_samples,
            samples=clip.samples,
            rate=SAMPLE_RATE,
        )
    with pytest.raises(ValueError, match="planned"):
        read_audio(
            paths[1].read_bytes(),
            native_rate=clip.sample_rate,
            start=clip.start_sample,
            source_samples=clip.source_samples,
            samples=clip.samples + 1,
            rate=SAMPLE_RATE,
        )
