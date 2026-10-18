"""The Python library plans MiniMax-H3 requests as the reference does.

``processing`` calls the request rules of ``uniserve-core`` through the rank
runtime library. Planned requests and their presentations, through that
call, match the vectors in ``minimax_h3_plan.json``, generated from the
diffusers MiniMax-H3 pipeline and the checkpoint's Qwen3-VL processor,
tokenizer and audio VAE; the rules' own tests
(``crates/core/src/video``) check each step against the same vectors.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from uniserve.media import image
from uniserve.models.minimax_h3 import processing
from uniserve.models.minimax_h3.processing import (
    AudioClip,
    AudioFacts,
    Condition,
    ConditionPlan,
    ConditionType,
    ImageFacts,
    ImageVision,
    KeyframeFit,
    PlanError,
    RequestPlan,
    Role,
    Target,
    Task,
    TextSegment,
    VideoClip,
    VideoFacts,
    VideoVision,
    VisionConfig,
    VisionSegment,
)

pytestmark = pytest.mark.unit

_FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "minimax_h3_plan.json"
    ).read_text()
)
_VISION = VisionConfig(**_FIXTURE["vision"])
_ACCEPTED = [case for case in _FIXTURE["requests"] if "expected" in case]
_REJECTED = [case for case in _FIXTURE["requests"] if "error" in case]


def _size(width_height: list[int]) -> image.Config:
    return image.Config(height=width_height[1], width=width_height[0])


def _names(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


def _processor(directory: Path, name: str, config: dict[str, Any]) -> None:
    pixels = config.pop("pixels")
    (directory / name).write_text(
        json.dumps(
            {
                **config,
                "size": dict(
                    zip(("shortest_edge", "longest_edge"), pixels, strict=True)
                ),
            }
        )
    )


def test_vision_config_reads_the_processor_configs(tmp_path):
    vision = _FIXTURE["vision"]
    geometry = {
        name: vision[name]
        for name in ("patch_size", "temporal_patch_size", "merge_size")
    }
    images = (vision["image_min_pixels"], vision["image_max_pixels"])
    videos = (vision["video_min_pixels"], vision["video_max_pixels"])
    _processor(
        tmp_path, "preprocessor_config.json", {**geometry, "pixels": images}
    )
    _processor(
        tmp_path,
        "video_preprocessor_config.json",
        {**geometry, "pixels": videos},
    )
    assert VisionConfig.from_processor(tmp_path) == _VISION

    # The image and video processors must patch alike.
    _processor(
        tmp_path,
        "video_preprocessor_config.json",
        {**geometry, "merge_size": 1, "pixels": videos},
    )
    with pytest.raises(ValueError, match="patch geometry"):
        VisionConfig.from_processor(tmp_path)


@pytest.mark.parametrize("case", _FIXTURE["canvas_cases"])
def test_canvas(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.canvas(*case["aspect"])
        return
    assert processing.canvas(*case["aspect"]) == _size(case["canvas"])


@pytest.mark.parametrize("case", _FIXTURE["duration_cases"])
def test_duration(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.frame_count(case["seconds"])
        return
    plan = processing.plan_request(
        Task.T2VA, Target(duration_seconds=case["seconds"]), [], _VISION
    )
    assert processing.frame_count(case["seconds"]) == case["num_frames"]
    assert plan.num_frames == case["num_frames"]
    assert plan.latent_frames == case["latent_frames"]
    assert plan.audio_latents == case["audio_latents"]


def _audio_facts(track: dict[str, Any]) -> AudioFacts:
    # The channel count is not a planning fact: mono is upmixed to stereo.
    return AudioFacts(track["sample_rate"], track["samples"])


def _media(entry: dict[str, Any]) -> ImageFacts | VideoFacts | AudioFacts:
    media = entry["media"]
    if "width" in media:
        return ImageFacts(media["width"], media["height"])
    if "display" in media:
        track = media.get("soundtrack")
        # The vectors state a video's displayed size as its aspect.
        return VideoFacts(
            display_width=media["display"][0],
            display_height=media["display"][1],
            width=media["display"][0],
            height=media["display"][1],
            frame_rate=Fraction(*media["frame_rate"]),
            frames=media["frames"],
            soundtrack=None if track is None else _audio_facts(track),
        )
    return _audio_facts(media)


def _plan(case: dict[str, Any]) -> RequestPlan:
    target = case["target"]
    conditions = [
        Condition(
            type=ConditionType(entry["type"]),
            role=Role(entry["role"]),
            media=_media(entry),
            frame_index=entry.get("frame_index"),
            start_seconds=entry.get("start_time_seconds"),
        )
        for entry in case["conditions"]
    ]
    # The vectors come from the released pipeline, which enlarges small
    # references.
    return processing.plan_request(
        Task(case["task"]),
        Target(
            aspect_ratio=target["aspect_ratio"],
            duration_seconds=target.get("duration_seconds"),
            short_edge=target["short_edge"],
        ),
        conditions,
        _VISION,
        references=processing.ReferenceConfig(enlarge=True),
    )


def _audio(clip: AudioClip) -> dict[str, int]:
    return {
        "start_sample": clip.start_sample,
        "source_samples": clip.source_samples,
        "samples": clip.samples,
        "latents": clip.latents,
    }


def _condition(plan: ConditionPlan) -> dict[str, Any]:
    """Express a condition plan in the vectors' layout."""
    prepared, seen = plan.prepared, plan.vision
    entry: dict[str, Any] = {
        "index": plan.index,
        "video_rows": plan.video_rows,
        "audio_rows": plan.audio_rows,
        "vision": None,
    }
    if isinstance(seen, ImageVision):
        entry["vision"] = {"grid": list(seen.grid), "tokens": seen.tokens}
    elif isinstance(seen, VideoVision):
        entry["vision"] = {
            "grid": list(seen.grid),
            "block_tokens": seen.block_tokens,
            "frame_indices": list(seen.frame_indices),
            "block_timestamps": list(seen.block_timestamps),
        }

    if isinstance(prepared, KeyframeFit):
        crop = prepared.cover_crop
        entry |= {
            "kind": "keyframe",
            "position": prepared.position.value,
            "cover_crop": None
            if crop is None
            else {
                "width": crop.width,
                "height": crop.height,
                "left": crop.left,
                "top": crop.top,
            },
        }
    elif isinstance(prepared, image.Config):
        entry |= {"kind": "image", "resize": [prepared.width, prepared.height]}
    elif isinstance(prepared, AudioClip):
        entry |= {"kind": "audio", "clip": _audio(prepared)}
    else:
        assert isinstance(prepared, VideoClip)
        track = prepared.soundtrack
        entry |= {
            "kind": "video",
            "canvas": [prepared.canvas.width, prepared.canvas.height],
            "start_frame": prepared.start_frame,
            "clip_frames": prepared.frames,
            "vae_frames": prepared.vae_frames,
            "latent_frames": prepared.latent_frames,
            "soundtrack": None if track is None else _audio(track),
        }
    return entry


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_request_plan(case):
    plan = _plan(case)
    expected = case["expected"]
    assert plan.canvas == _size(expected["canvas"])
    assert plan.num_frames == expected["num_frames"]
    assert plan.latent_frames == expected["latent_frames"]
    assert plan.audio_latents == expected["audio_latents"]
    assert plan.target_video_rows == expected["target_video_rows"]
    assert plan.target_audio_rows == expected["target_audio_rows"]
    conditions = [_condition(item) for item in plan.conditions]
    assert conditions == expected["conditions"]
    assert plan.condition_video_rows == expected["condition_video_rows"]
    assert plan.condition_audio_rows == expected["condition_audio_rows"]
    # Together with the presentation's rows, the planned rows fill the
    # reference's packed sequence exactly.
    planned = (
        plan.target_video_rows
        + plan.target_audio_rows
        + plan.condition_video_rows
        + plan.condition_audio_rows
    )
    assert planned + expected["text_rows"] == expected["sequence_rows"]


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_presentation_segments(case):
    segments = processing.presentation_segments(_plan(case), case["prompt"])
    expected = [
        VisionSegment(f"<|{item['vision']}_pad|>", item["tokens"])
        if "vision" in item
        else TextSegment(item["text"])
        for item in case["expected"]["segments"]
    ]
    assert segments == expected


class _RecordedTokenizer:
    """The checkpoint tokenizer's recorded output on the vectors' texts."""

    def __init__(self) -> None:
        self.ids = dict(_FIXTURE["tokens"])
        self.texts = {
            item["text"]: item["token_ids"]
            for case in _ACCEPTED
            for item in case["expected"]["segments"]
            if "text" in item
        }

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.texts[text])

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids[token]


def _joined(case: dict[str, Any]) -> tuple[list[int], list[int]]:
    """The expected token ids and tags, rebuilt from the segments."""
    tokens = _FIXTURE["tokens"]
    token_ids: list[int] = []
    tags: list[int] = []
    for item in case["expected"]["segments"]:
        if "vision" in item:
            pad = tokens[f"<|{item['vision']}_pad|>"]
            block = [
                tokens["<|vision_start|>"],
                *([pad] * item["tokens"]),
                tokens["<|vision_end|>"],
            ]
            token_ids += block
            tags += [0] * len(block)
        else:
            token_ids += item["token_ids"]
            tags += [1] * len(item["token_ids"])
    return token_ids, tags


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_presentation_tokens(case):
    presentation = processing.present(
        _RecordedTokenizer(), _plan(case), case["prompt"]
    )
    tokens = (list(presentation.token_ids), list(presentation.tags))
    assert tokens == _joined(case)


@pytest.fixture(scope="module")
def checkpoint_tokenizer():
    """The tokenizer of the checkpoint ``UNISERVE_MINIMAX_H3_MODEL`` names."""
    root = os.environ.get("UNISERVE_MINIMAX_H3_MODEL")
    if not root:
        pytest.skip("UNISERVE_MINIMAX_H3_MODEL names no MiniMax-H3 checkpoint")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(Path(root) / "tokenizer")


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_presentation_with_checkpoint_tokenizer(case, checkpoint_tokenizer):
    presentation = processing.present(
        checkpoint_tokenizer, _plan(case), case["prompt"]
    )
    tokens = (list(presentation.token_ids), list(presentation.tags))
    assert tokens == _joined(case)


def test_prompt_must_be_plain_text():
    case = _ACCEPTED[0]
    tokenizer = _RecordedTokenizer()
    tokenizer.texts["<|image_pad|>"] = [_FIXTURE["tokens"]["<|image_pad|>"]]
    for prompt in ("<|image_pad|>", ""):
        with pytest.raises(PlanError) as error:
            processing.present(tokenizer, _plan(case), prompt)
        assert error.value.field == "prompt"


@pytest.mark.parametrize("case", _REJECTED, ids=_names(_REJECTED))
def test_rejected_request(case):
    # Planning rejects most requests; the presentation rejects the prompt.
    with pytest.raises(PlanError) as error:
        processing.present(_RecordedTokenizer(), _plan(case), case["prompt"])
    assert error.value.field == case["error"]["field"]


def test_checkpoint_canvases_restrict_the_target():
    served = [image.Config(height=768, width=1344)]
    plan = processing.plan_request(
        Task.T2VA, Target(duration_seconds=5.0), [], _VISION, canvases=served
    )
    assert plan.canvas == served[0]
    with pytest.raises(PlanError) as error:
        processing.plan_request(
            Task.T2VA,
            Target(aspect_ratio="9:16", duration_seconds=5.0),
            [],
            _VISION,
            canvases=served,
        )
    assert error.value.field == "target.aspect_ratio"


def test_only_served_tasks_are_accepted():
    with pytest.raises(PlanError) as error:
        processing.plan_request(
            Task.T2VA,
            Target(duration_seconds=5.0),
            [],
            _VISION,
            tasks=(Task.REF2VA,),
        )
    assert error.value.field == "task"


def test_max_seconds_bounds_the_duration():
    with pytest.raises(PlanError) as error:
        processing.plan_request(
            Task.T2VA,
            Target(duration_seconds=10.0),
            [],
            _VISION,
            max_seconds=8.0,
        )
    assert error.value.field == "target.duration_seconds"


def test_references_keep_their_size_unless_enlarged():
    """A reference image keeps its size unless it exceeds its encoding size.

    By default a 1920x1080 image keeps its own size on the 32-pixel grid,
    1080 rounding to 1088; enlarging scales it to the 2048-pixel short edge
    as the released pipeline does. A 4096x2304 image is reduced to that edge
    either way.
    """

    def planned(media, references):
        plan = processing.plan_request(
            Task.REF2VA,
            Target(duration_seconds=5.0),
            [
                Condition(
                    type=ConditionType.IMAGE, role=Role.REFERENCE, media=media
                )
            ],
            _VISION,
            references=references,
        )
        return plan.conditions[0].prepared

    default, enlarged = (
        processing.ReferenceConfig(),
        processing.ReferenceConfig(enlarge=True),
    )
    small, large = ImageFacts(1920, 1080), ImageFacts(4096, 2304)
    assert planned(small, default) == image.Config(height=1088, width=1920)
    reduced = image.Config(height=2048, width=3648)
    assert planned(small, enlarged) == reduced
    for references in (default, enlarged):
        assert planned(large, references) == reduced
