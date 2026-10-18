"""Normalize MiniMax-H3 checkpoints into one typed configuration.

``read_config`` recognizes the checkpoint layout (``checkpoint.detect``),
reads the JSON sidecars of every component, validates the FastVideo
inference contract where one exists, and returns one immutable ``Config``
before any module is constructed. The checkpoint's own configuration
defines each network's dimensions; the modules are parameterised by it. The
reader refuses only what the packing and reconstruction code hard-codes:
the 16x spatial and 4x temporal video compression with its 17-frame clip
and the audio hop of the timeline. A checkpoint also states each
denoiser's schedule and attention (``DenoiserConfig``): the base release's
uniform schedule and dense attention, or a FastH3 student's distilled
schedule and sparse attention as its ``fastvideo_inference.json`` states
them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, TypeGuard, get_origin, get_type_hints

from uniserve.media import image
from uniserve.models import qwen3_vl

from . import audio_vae, video_vae
from .checkpoint import (
    DENOISER_DIRECTORIES,
    DENOISER_TASKS,
    Kind,
    Layout,
    detect,
)
from .encoder import RETAINED_LAYERS, TextEncoderConfig
from .packing import (
    AUDIO_HOP,
    AUDIO_SAMPLE_RATE,
    CLIP_FRAMES,
    CLIP_LATENTS,
    VIDEO_TILE_SHAPES,
)
from .processing import buckets

# DMD rungs are unshifted noise levels on the 1000-step training clock.
TRAINING_CLOCK = 1000.0

# A diffusers root evaluates the released sigma grid of 50 points: 49 network
# evaluations and the clean endpoint.
UNIFORM_GRID_POINTS = 50

# FastH3 DMD exports are trained on, and generate only, the 12 training
# buckets of the request rules: the named aspect ratios at the 768p and 480p
# resolutions. A deployment prepares a subset of them.
DMD_CANVASES = buckets()

# A FastVideo PDD student's packed sequence capacity in rows; the checkpoint
# owns this bound and serving options may only tighten it.
PDD_MAX_SEQUENCE_ROWS = 131_072


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Define H3 widths, layers, attention, modulation and output heads.

    ``output_heads`` is 1 for an ordinary DiT and the fine-grid interval
    count for a parallel-decoding (PDD) student, whose output projections
    hold one head-major prediction per interval.
    """

    hidden_size: int
    num_attention_heads: int
    head_dim: int
    num_hidden_layers: int
    num_refiner_layers: int
    intermediate_size: int
    video_channels: int
    audio_channels: int
    text_dim: int
    frequency_dim: int
    time_hidden_dim: int
    time_dim: int
    rope_frequency_dim: int
    rope_theta: float
    norm_eps: float
    qk_norm_eps: float
    output_heads: int

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_attention_heads",
            "head_dim",
            "num_hidden_layers",
            "num_refiner_layers",
            "intermediate_size",
            "video_channels",
            "audio_channels",
            "text_dim",
            "frequency_dim",
            "time_hidden_dim",
            "time_dim",
            "rope_frequency_dim",
            "output_heads",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"H3 transformer {name} must be a positive integer"
                )
        for name in ("rope_theta", "norm_eps", "qk_norm_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"H3 transformer {name} must be finite and positive"
                )
        # Each of the three rotary axes (time, height, width) rotates
        # 2 * rope_frequency_dim channels of a head; the rest stay unrotated.
        if (
            self.frequency_dim % 2
            or self.rope_frequency_dim * 6 > self.head_dim
        ):
            raise ValueError(
                "H3 transformer rotary dimensions must fit its attention "
                "num_attention_heads"
            )


def _check_shifts(video_shift: object, audio_shift: object) -> None:
    for name, value in (
        ("video_shift", video_shift),
        ("audio_shift", audio_shift),
    ):
        if not _is_positive_number(value):
            raise ValueError(f"H3 {name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class UniformGrid:
    """The released full-step schedule.

    ``points`` sigma points, ``linspace(1, 0, points)`` in FP32, shifted per
    modality as ``sigma = s u / (1 + (s - 1) u)``; the network evaluates
    every point but the clean endpoint.
    """

    points: int
    video_shift: float
    audio_shift: float

    def __post_init__(self) -> None:
        if not _is_integer(self.points) or self.points < 2:
            raise ValueError("H3 uniform grid requires at least two points")
        _check_shifts(self.video_shift, self.audio_shift)


@dataclass(frozen=True, slots=True)
class DmdLadder:
    """A DMD student's trained rungs.

    Each rung is an unshifted noise level on the ``clock`` (the 1000-step
    training clock), shifted once per modality; the clean endpoint follows
    the last rung. The student starts from unit noise.
    """

    rungs: tuple[int, ...]
    video_shift: float
    audio_shift: float
    clock: float = TRAINING_CLOCK

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rungs, tuple)
            or not self.rungs
            or any(
                not _is_integer(rung) or not 0 < rung <= self.clock
                for rung in self.rungs
            )
            or any(
                left <= right for left, right in zip(self.rungs, self.rungs[1:])
            )
        ):
            raise ValueError(
                "H3 DMD rungs must be strictly decreasing integers within "
                "the training clock"
            )
        _check_shifts(self.video_shift, self.audio_shift)
        if not _is_positive_number(self.clock):
            raise ValueError("H3 DMD clock must be finite and positive")


@dataclass(frozen=True, slots=True)
class PddGrid:
    """A parallel-decoding student's fine grid and runtime blocks.

    The fine grid ``linspace(max_t, 0, intervals + 1)`` lives on the
    unshifted base clock; each modality reaches its noise level through the
    shift ``f_s(u) = s u M / (u (s - 1) + M)`` whose fixed point is
    ``M = max_t``. ``nodes`` are the fine-grid indices of the block
    boundaries, one network evaluation per block.
    """

    intervals: int
    nodes: tuple[int, ...]
    video_shift: float
    audio_shift: float
    max_t: float

    def __post_init__(self) -> None:
        if not _is_integer(self.intervals) or self.intervals < 2:
            raise ValueError("H3 PDD grid requires at least two intervals")
        if (
            not isinstance(self.nodes, tuple)
            or len(self.nodes) < 2
            or any(not _is_integer(node) for node in self.nodes)
            or self.nodes[0] != 0
            or self.nodes[-1] != self.intervals
            or any(
                left >= right for left, right in zip(self.nodes, self.nodes[1:])
            )
        ):
            raise ValueError(
                "H3 PDD nodes must increase strictly from 0 to the interval "
                "count"
            )
        _check_shifts(self.video_shift, self.audio_shift)
        if not _is_positive_number(self.max_t) or self.max_t > 1:
            raise ValueError("H3 PDD max_t must lie in (0, 1]")


Schedule: TypeAlias = UniformGrid | DmdLadder | PddGrid


@dataclass(frozen=True, slots=True)
class DenseAttention:
    """Full bidirectional attention over the packed sequence."""


@dataclass(frozen=True, slots=True)
class SparseAttention:
    """Video sparse attention over ``tile``-row spatiotemporal tiles.

    Each video query tile keeps ``ceil((1 - sparsity) n)`` of the ``n`` key
    tiles of the generated video. Every reference video is a segment of its
    own, of which a query keeps the fraction ``reference_keep``; a network
    without reference videos states none (``packing.SegmentPacking``).
    """

    tile: int
    sparsity: float
    reference_keep: float | None = None

    def __post_init__(self) -> None:
        # A tile holds one video tile shape of the segment packing.
        if self.tile not in VIDEO_TILE_SHAPES:
            raise ValueError(
                "H3 sparse attention tiles hold "
                f"{' or '.join(map(str, VIDEO_TILE_SHAPES))} rows"
            )
        for name in ("sparsity",) + (
            () if self.reference_keep is None else ("reference_keep",)
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"H3 sparse attention {name} must be finite")
        if not 0 <= self.sparsity < 1:
            raise ValueError("H3 VSA sparsity must lie in [0, 1)")
        if self.reference_keep is not None and not 0 < self.reference_keep <= 1:
            raise ValueError("H3 reference keep rate must lie in (0, 1]")


Attention: TypeAlias = DenseAttention | SparseAttention


@dataclass(frozen=True, slots=True)
class DenoiserConfig:
    """One DiT a deployment can place, with the requests it serves.

    Attributes:
        transformer: The network dimensions.
        schedule: The checkpoint's fixed schedule; requests cannot change it.
        attention: Dense or sparse attention.
        tasks: The tasks this DiT serves, in canonical order.
        canvases: The only canvases the checkpoint generates, or None when it
            generates every canvas of the canvas rules.
        max_sequence_rows: The checkpoint's packed sequence capacity, or None
            when only memory and serving options bound it.
    """

    transformer: TransformerConfig
    schedule: Schedule
    attention: Attention
    tasks: tuple[str, ...]
    canvases: tuple[image.Config, ...] | None
    max_sequence_rows: int | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tasks, tuple)
            or not self.tasks
            or any(
                not any(task in served for served in DENOISER_TASKS.values())
                for task in self.tasks
            )
            or len(set(self.tasks)) != len(self.tasks)
        ):
            raise ValueError("H3 denoiser tasks must name distinct H3 tasks")
        if self.canvases is not None and (
            not isinstance(self.canvases, tuple)
            or not self.canvases
            or any(
                not isinstance(value, image.Config) for value in self.canvases
            )
        ):
            raise ValueError("H3 denoiser canvases must be image.Config values")
        if self.max_sequence_rows is not None and (
            not _is_integer(self.max_sequence_rows)
            or self.max_sequence_rows < 1
        ):
            raise ValueError("H3 sequence capacity must be a positive integer")
        heads = self.transformer.output_heads
        if isinstance(self.schedule, PddGrid) != (heads > 1) or (
            isinstance(self.schedule, PddGrid)
            and heads != self.schedule.intervals
        ):
            raise ValueError(
                "H3 output heads must equal the PDD interval count, and only "
                "a PDD student has more than one"
            )


@dataclass(frozen=True, slots=True)
class Config:
    """Compose the checkpoint's encoders, denoisers and VAEs.

    ``denoisers`` maps each denoising component the checkpoint holds
    (``denoiser`` for the ``transformer`` partition, ``reference_denoiser``
    for ``transformer_ref``) to its configuration; a deployment places one.
    """

    text_encoder: TextEncoderConfig
    denoisers: Mapping[str, DenoiserConfig]
    video_vae: video_vae.Config
    audio_vae: audio_vae.Config

    def __post_init__(self) -> None:
        if not isinstance(self.denoisers, Mapping) or not self.denoisers:
            raise ValueError("H3 checkpoints hold at least one denoiser")
        if set(self.denoisers) - set(DENOISER_DIRECTORIES):
            raise ValueError(
                "H3 denoisers must be named denoiser or reference_denoiser"
            )
        object.__setattr__(
            self, "denoisers", MappingProxyType(dict(self.denoisers))
        )
        for denoiser in self.denoisers.values():
            transformer = denoiser.transformer
            if self.text_encoder.hidden_size != transformer.text_dim:
                raise ValueError(
                    "H3 text features must match denoiser conditioning width"
                )
            if self.video_vae.latent_channels != transformer.video_channels:
                raise ValueError(
                    "H3 video latent channels must match the denoiser"
                )
            if self.audio_vae.latent_channels != transformer.audio_channels:
                raise ValueError(
                    "H3 audio latent channels must match the denoiser"
                )


# Checkpoint config field names mapped to the typed config fields they fill.
# ``normalize`` reads the checkpoint through them.
TRANSFORMER_FIELDS = {
    "num_attention_heads": "num_attention_heads",
    "attention_head_dim": "head_dim",
    "hidden_size": "hidden_size",
    "num_layers": "num_hidden_layers",
    "num_refiner_layers": "num_refiner_layers",
    "ffn_dim": "intermediate_size",
    "in_channels": "video_channels",
    "audio_in_channels": "audio_channels",
    "text_dim": "text_dim",
    "freq_dim": "frequency_dim",
    "time_embed_hidden_dim": "time_hidden_dim",
    "time_embed_dim": "time_dim",
    "rope_freq_dim": "rope_frequency_dim",
    "rope_theta": "rope_theta",
    "norm_eps": "norm_eps",
    "qk_norm_eps": "qk_norm_eps",
}
TEXT_FIELDS = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_hidden_layers": "num_checkpoint_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "head_dim": "head_dim",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "max_position_embeddings": "max_position_embeddings",
}


def _reject(field: str, expected: object, value: object) -> ValueError:
    return ValueError(
        f"unsupported MiniMax-H3 checkpoint: {field} must be {expected!r}, "
        f"got {value!r}"
    )


def _check_contract_fields(
    contract: Mapping[str, Any], required: Mapping[str, object]
) -> None:
    for name, expected in required.items():
        if contract.get(name) != expected:
            raise _reject(name, expected, contract.get(name))


def _contract_shifts(
    contract: Mapping[str, Any], shifts: Mapping[str, float]
) -> None:
    """Require a contract that restates a scheduler shift to agree with it."""
    for modality in ("video", "audio"):
        name = f"{modality}_scheduler_shift"
        if name in contract and contract[name] != shifts[modality]:
            raise ValueError(
                f"unsupported MiniMax-H3 checkpoint: {name}="
                f"{contract[name]!r} disagrees with the {modality} scheduler "
                f"shift {shifts[modality]!r}"
            )


def _sparsity(contract: Mapping[str, Any]) -> float:
    value = contract.get("vsa_sparsity")
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: vsa_sparsity must lie in "
            f"[0, 1), got {value!r}"
        )
    return float(value)


def _dmd_denoiser(
    contract: Mapping[str, Any],
    transformer: TransformerConfig,
    shifts: Mapping[str, float],
) -> DenoiserConfig:
    """Validate a FastH3 DMD contract and describe its denoiser.

    The export's trained DMD rungs are the denoising ladder: a uniform grid
    over the same number of points is not a substitute for them.

    Raises:
        ValueError: The contract is not a text-to-video-and-audio DMD export
            this implementation serves, naming the offending field.
    """
    _check_contract_fields(
        contract, {"task": "t2av", "attention_backend": "VIDEO_SPARSE_ATTN_H3"}
    )
    # The model has no unconditional branch and the sparse-attention kernel
    # tiles 64 rows.
    for name, expected in (("guidance_scale", 1.0), ("vsa_tile_size", 64)):
        value = contract.get(name)
        if not _is_positive_number(value) or value != expected:
            raise _reject(name, expected, value)

    rungs = contract.get("dmd_denoising_steps")
    if (
        not isinstance(rungs, list)
        or not rungs
        or any(
            not _is_integer(rung) or not 0 < rung <= TRAINING_CLOCK
            for rung in rungs
        )
        or any(left <= right for left, right in zip(rungs, rungs[1:]))
    ):
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: dmd_denoising_steps must be "
            "strictly decreasing integers in (0, 1000], got "
            f"{rungs!r}"
        )
    # `num_inference_steps` counts sigma-grid points, including the clean
    # endpoint the solver reaches after the last rung.
    for name, expected in (
        ("transformer_forwards", len(rungs)),
        ("num_inference_steps", len(rungs) + 1),
    ):
        value = contract.get(name)
        if not _is_integer(value) or value != expected:
            raise ValueError(
                f"unsupported MiniMax-H3 checkpoint: {name} must be "
                f"{expected} for {len(rungs)} DMD rungs, got {value!r}"
            )
    sparsity = _sparsity(contract)
    _contract_shifts(contract, shifts)
    return DenoiserConfig(
        transformer=transformer,
        schedule=DmdLadder(
            rungs=tuple(rungs),
            video_shift=shifts["video"],
            audio_shift=shifts["audio"],
        ),
        attention=SparseAttention(tile=64, sparsity=sparsity),
        tasks=("t2va",),
        canvases=DMD_CANVASES,
        max_sequence_rows=None,
    )


def _pdd_denoiser(
    contract: Mapping[str, Any],
    transformer: TransformerConfig,
    shifts: Mapping[str, float],
    component: str,
) -> DenoiserConfig:
    """Validate a FastVideo PDD contract and describe its denoiser.

    Raises:
        ValueError: The contract is not a reference-conditioned PDD student
            this implementation serves, naming the offending field.
    """
    _check_contract_fields(
        contract,
        {
            "model_type": "ref2va",
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "conditioning": "fixed_ordered_references_target_only_flow",
            "vsa_ref_policy": "p2_multi_region",
        },
    )
    if component != "reference_denoiser":
        raise _reject(
            "transformer_component",
            "transformer_ref",
            contract.get("transformer_component"),
        )
    for name, expected in (("guidance_scale", 1.0), ("vsa_tile_size", 128)):
        value = contract.get(name)
        if not _is_positive_number(value) or value != expected:
            raise _reject(name, expected, value)

    intervals = contract.get("pdd_steps")
    if not _is_integer(intervals) or intervals < 2:
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: pdd_steps must be an integer "
            f">= 2, got {intervals!r}"
        )
    if transformer.output_heads != intervals:
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: the transformer's pdd_steps "
            f"{transformer.output_heads} disagrees with the contract's "
            f"{intervals}"
        )
    nodes = contract.get("pdd_step_indices")
    if not isinstance(nodes, list) or any(
        not _is_integer(node) for node in nodes
    ):
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: pdd_step_indices must list "
            f"integers, got {nodes!r}"
        )
    # FastVideo's `num_inference_steps` counts network evaluations, one per
    # block of the fine grid.
    for name in ("transformer_forwards", "num_inference_steps"):
        value = contract.get(name)
        if not _is_integer(value) or value != len(nodes) - 1:
            raise ValueError(
                f"unsupported MiniMax-H3 checkpoint: {name} must be "
                f"{len(nodes) - 1} for {len(nodes) - 1} PDD blocks, "
                f"got {value!r}"
            )
    max_t = contract.get("grid_max_t")
    if not _is_positive_number(max_t) or max_t > 1:
        raise _reject("grid_max_t", "a value in (0, 1]", max_t)
    keep = contract.get("vsa_ref_keep_rate")
    if not _is_positive_number(keep) or keep > 1:
        raise _reject("vsa_ref_keep_rate", "a value in (0, 1]", keep)
    sparsity = _sparsity(contract)
    _contract_shifts(contract, shifts)
    return DenoiserConfig(
        transformer=transformer,
        schedule=PddGrid(
            intervals=intervals,
            nodes=tuple(nodes),
            video_shift=shifts["video"],
            audio_shift=shifts["audio"],
            max_t=float(max_t),
        ),
        attention=SparseAttention(
            tile=128, sparsity=sparsity, reference_keep=float(keep)
        ),
        tasks=DENOISER_TASKS[component],
        canvases=None,
        max_sequence_rows=PDD_MAX_SEQUENCE_ROWS,
    )


def _transformer(values: Mapping[str, Any], *, heads: int) -> TransformerConfig:
    missing = set(TRANSFORMER_FIELDS) - values.keys()
    if missing:
        raise ValueError(
            "MiniMax-H3 transformer is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    if values.get("patch_size") != [1, 2, 2]:
        raise ValueError("MiniMax-H3 transformer patch_size must be [1, 2, 2]")
    if values.get("final_norm_eps") != values.get("norm_eps"):
        raise ValueError(
            "MiniMax-H3 transformer final_norm_eps must equal norm_eps"
        )
    return TransformerConfig(
        output_heads=heads,
        **{
            target: values[source]
            for source, target in TRANSFORMER_FIELDS.items()
        },
    )


def _pixels(preprocessor: Mapping[str, Any]) -> qwen3_vl.PixelConfig:
    """Read the pixel normalization of the checkpoint's image processor."""
    missing = {"image_mean", "image_std"} - preprocessor.keys()
    if missing:
        raise ValueError(
            "MiniMax-H3 processor preprocessor_config is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    mean, std = preprocessor["image_mean"], preprocessor["image_std"]
    if not isinstance(mean, list) or not isinstance(std, list):
        raise ValueError(
            "MiniMax-H3 processor image_mean and image_std must be lists"
        )
    return qwen3_vl.PixelConfig(mean=tuple(mean), std=tuple(std))


def _text_encoder(
    metadata: Mapping[str, Any], preprocessor: Mapping[str, Any]
) -> TextEncoderConfig:
    text = metadata.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("MiniMax-H3 text encoder requires text_config")
    for name, expected in (("hidden_act", "silu"), ("attention_bias", False)):
        if text.get(name) != expected:
            raise ValueError(f"unsupported MiniMax-H3 text encoder {name}")
    if metadata.get("tie_word_embeddings", False):
        raise ValueError(
            "MiniMax-H3 checkpoint requires an independent vocabulary head"
        )
    # The language layers rotate by interleaved M-RoPE; its sections are
    # the checkpoint's.
    rope = text.get("rope_scaling")
    if (
        not isinstance(rope, dict)
        or rope.get("rope_type") != "default"
        or rope.get("mrope_interleaved") is not True
        or not isinstance(rope.get("mrope_section"), list)
    ):
        raise ValueError("unsupported MiniMax-H3 text encoder rope_scaling")
    vision = metadata.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("MiniMax-H3 text encoder requires vision_config")
    missing = set(TEXT_FIELDS) - text.keys()
    if missing:
        raise ValueError(
            "MiniMax-H3 text_encoder.text_config is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    missing = {"image_token_id", "video_token_id"} - metadata.keys()
    if missing:
        raise ValueError(
            "MiniMax-H3 text_encoder config is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    return TextEncoderConfig(
        **{target: text[source] for source, target in TEXT_FIELDS.items()},
        num_retained_layers=RETAINED_LAYERS,
        mrope_sections=tuple(rope["mrope_section"]),
        image_token_id=metadata["image_token_id"],
        video_token_id=metadata["video_token_id"],
        vision=qwen3_vl.read_vision_config(vision),
        pixels=_pixels(preprocessor),
    )


def _sequence_fields(config: type) -> set[str]:
    """Name the fields ``config`` annotates as tuples."""
    return {
        name
        for name, hint in get_type_hints(config).items()
        if get_origin(hint) is tuple
    }


def _video_vae(values: Mapping[str, Any]) -> video_vae.Config:
    result = {}
    sequences = _sequence_fields(video_vae.Config)
    for field in fields(video_vae.Config):
        if field.name not in values:
            raise ValueError(
                f"MiniMax-H3 video_vae is missing field {field.name}"
            )
        value = values[field.name]
        if field.name in sequences:
            if not isinstance(value, (tuple, list)):
                raise ValueError(
                    f"MiniMax-H3 video_vae.{field.name} must be a sequence"
                )
            value = tuple(value)
        result[field.name] = value
    config = video_vae.Config(**result)

    # Packing and reconstruction hard-code the clip timeline: a 17-frame
    # clip encodes to 5 latent frames, 3 of which the final clip drops, at
    # 16x spatial and 4x temporal compression.
    clip = (
        config.clip_length,
        config.clip_latents,
        config.clip_latents - config.token_drop,
        config.spatial_compression,
        config.temporal_compression,
    )
    if clip != (CLIP_FRAMES, CLIP_LATENTS, 2, 16, 4):
        raise ValueError(
            "MiniMax-H3 video_vae must encode 17-frame clips to 5 latent "
            "frames, dropping 3 of them from the final clip, at 16x spatial "
            "and 4x temporal compression"
        )
    return config


def _audio_vae(values: Mapping[str, Any]) -> audio_vae.Config:
    result = {}
    sequences = _sequence_fields(audio_vae.Config)
    for field in fields(audio_vae.Config):
        if field.name not in values:
            raise ValueError(
                f"MiniMax-H3 audio_vae is missing field {field.name}"
            )
        value = values[field.name]
        if field.name in sequences and not isinstance(value, (tuple, list)):
            raise ValueError(
                f"MiniMax-H3 audio_vae.{field.name} must be a sequence"
            )
        if field.name == "resblock_dilation_sizes":
            if any(not isinstance(row, (tuple, list)) for row in value):
                raise ValueError(
                    "MiniMax-H3 audio_vae.resblock_dilation_sizes "
                    "must contain sequences"
                )
            value = tuple(tuple(row) for row in value)
        elif field.name in sequences:
            value = tuple(value)
        result[field.name] = value
    config = audio_vae.Config(**result)

    # Packing counts audio latents in hops of the timeline.
    if (
        math.prod(config.encoder_rates) != AUDIO_HOP
        or math.prod(config.decoder_rates) != AUDIO_HOP
    ):
        raise ValueError(
            f"MiniMax-H3 audio_vae must hop {AUDIO_HOP} samples per latent"
        )
    return config


def _shifts(metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    shifts = {}
    for modality, name in (
        ("video", "scheduler"),
        ("audio", "audio_scheduler"),
    ):
        if "shift" not in metadata[name]:
            raise ValueError(f"MiniMax-H3 {name} is missing field shift")
        shifts[modality] = metadata[name]["shift"]
    return shifts


def normalize(
    layout: Layout, metadata: Mapping[str, Mapping[str, Any]]
) -> Config:
    """Normalize recognized checkpoint sidecars without allocating resources.

    ``metadata`` holds the parsed JSON of ``text_encoder``, ``processor``
    (the image processor's ``preprocessor_config.json``), ``video_vae``,
    ``audio_vae``, ``scheduler``, ``audio_scheduler`` and one entry per
    denoising component the layout holds (its ``config.json``).
    """
    for name in (
        "text_encoder",
        "processor",
        "video_vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
        *layout.denoisers,
    ):
        if not isinstance(metadata.get(name), Mapping):
            raise ValueError(f"MiniMax-H3 requires {name} metadata")
    # Soundtracks are resampled to the timeline's AUDIO_SAMPLE_RATE, so the
    # audio VAE must run at it.
    if metadata["audio_vae"].get("sampling_rate") != AUDIO_SAMPLE_RATE:
        raise ValueError(
            f"MiniMax-H3 audio output requires a {AUDIO_SAMPLE_RATE} Hz "
            "sampling clock"
        )
    shifts = _shifts(metadata)
    denoisers: dict[str, DenoiserConfig] = {}
    for component in layout.denoisers:
        values = metadata[component]
        heads = values.get("pdd_steps", 1)
        if not _is_integer(heads) or heads < 1:
            raise ValueError(
                "unsupported MiniMax-H3 checkpoint: transformer pdd_steps "
                f"must be a positive integer, got {heads!r}"
            )
        transformer = _transformer(values, heads=heads)
        if layout.kind is Kind.FASTVIDEO_EXPORT:
            assert layout.contract is not None
            denoisers[component] = _dmd_denoiser(
                layout.contract, transformer, shifts
            )
        elif layout.kind is Kind.COMPONENT_EXPORT:
            assert layout.contract is not None
            denoisers[component] = _pdd_denoiser(
                layout.contract, transformer, shifts, component
            )
        else:
            if heads != 1:
                raise ValueError(
                    "unsupported MiniMax-H3 checkpoint: a diffusers root DiT "
                    "has one output head"
                )
            denoisers[component] = DenoiserConfig(
                transformer=transformer,
                schedule=UniformGrid(
                    points=UNIFORM_GRID_POINTS,
                    video_shift=shifts["video"],
                    audio_shift=shifts["audio"],
                ),
                attention=DenseAttention(),
                tasks=DENOISER_TASKS[component],
                canvases=None,
                max_sequence_rows=None,
            )
    return Config(
        text_encoder=_text_encoder(
            metadata["text_encoder"], metadata["processor"]
        ),
        denoisers=denoisers,
        video_vae=_video_vae(metadata["video_vae"]),
        audio_vae=_audio_vae(metadata["audio_vae"]),
    )


# Component sidecars, relative to the directory that holds the component. A
# component export holds its schedulers and draws the others from its base.
_SIDECARS = {
    "text_encoder": "text_encoder/config.json",
    "processor": "processor/preprocessor_config.json",
    "video_vae": "vae/config.json",
    "audio_vae": "audio_vae/config.json",
    "scheduler": "scheduler/scheduler_config.json",
    "audio_scheduler": "audio_scheduler/scheduler_config.json",
}
_EXPORT_SIDECARS = frozenset({"scheduler", "audio_scheduler"})


def read_config(root: Path, io, *, sources, base: Path | None = None) -> Config:
    """Read all architecture sidecars before any numerical module construction.

    ``root`` is the checkpoint directory, whose sidecars the loader has
    already fetched; ``io`` and ``sources`` (the resolved ``config_sources``,
    of which H3 declares none) are part of the package interface. ``base``
    supplies the base components of a component export, resolved by the loader;
    the export's text encoder and VAE sidecars are read from it, and its
    schedulers and DiT partition from ``root``. Every other layout has no
    base.

    An unreadable sidecar raises ``OSError``; invalid JSON, an unsupported
    checkpoint, or a base that does not match the layout raises
    ``ValueError``.
    """
    layout = detect(root)
    if (layout.kind is Kind.COMPONENT_EXPORT) != (base is not None):
        raise ValueError(
            "a MiniMax-H3 component export, and only one, reads its other "
            "components from its pinned base checkpoint"
        )
    metadata: dict[str, Any] = {}
    for name, relative in _SIDECARS.items():
        directory = root if base is None or name in _EXPORT_SIDECARS else base
        metadata[name] = json.loads(
            (directory / relative).read_text(encoding="utf-8")
        )
    for component, subdirectory in layout.denoisers.items():
        metadata[component] = json.loads(
            (root / subdirectory / "config.json").read_text(encoding="utf-8")
        )
    return normalize(layout, metadata)


# Checkpoint tensor headers needed to resolve architecture before module
# selection. H3's architecture comes entirely from JSON sidecars.
config_sources = ()
