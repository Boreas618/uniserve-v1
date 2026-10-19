"""Normalize MiniMax-H3 checkpoints into one typed configuration.

``read_config`` recognizes the checkpoint (``checkpoint.detect``), reads the
JSON sidecars of every component, and returns one immutable ``Config``
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
from .checkpoint import TASK_PARTITIONS, Layout, detect
from .encoder import RETAINED_LAYERS, TextEncoderConfig
from .packing import (
    AUDIO_HOP,
    AUDIO_SAMPLE_RATE,
    CLIP_FRAMES,
    CLIP_LATENTS,
    VIDEO_TILE_SHAPES,
)
from .processing import canvas

# DMD rungs are unshifted noise levels on the 1000-step training schedule.
NUM_TRAIN_TIMESTEPS = 1000.0

# The base release evaluates the released sigma grid of 50 points: 49
# network evaluations and the clean endpoint.
UNIFORM_GRID_POINTS = 50

# The aspect ratios a request names a canvas by besides ``auto``, as
# ``(width, height)``, landscape first.
NAMED_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))

# The 768p resolution class: the canvas rule's short edge, at which the
# released checkpoint generates the rule's canvas of every named ratio. A
# request selects these canvases by stating it as its target short edge.
CANVAS_SHORT_EDGE = 768

# The canvas of every named aspect ratio at the 768p class, keyed as a
# request selects it: ``(short_edge, "W:H")``. The base release generates
# these by the canvas rule (``processing.canvas``).
NAMED_CANVASES: Mapping[tuple[int, str], image.Config] = MappingProxyType(
    {
        (CANVAS_SHORT_EDGE, f"{width}:{height}"): canvas(width, height)
        for width, height in NAMED_ASPECT_RATIOS
    }
)

# FastH3 DMD students are trained on, and generate only, 12 buckets: the
# named aspect ratios at the 768p and 480p classes. The 768p buckets are the
# canvas rule's canvases; the 480p buckets, (height, width) below in the same
# ratio order, are the training table's own rather than a rescaled rule
# (21:9 is 992x416, not 960x416). A deployment prepares a subset of them.
DMD_CANVASES: Mapping[tuple[int, str], image.Config] = MappingProxyType(
    {
        **NAMED_CANVASES,
        (480, "21:9"): image.Config(416, 992),
        (480, "16:9"): image.Config(480, 832),
        (480, "4:3"): image.Config(480, 640),
        (480, "1:1"): image.Config(480, 480),
        (480, "3:4"): image.Config(640, 480),
        (480, "9:16"): image.Config(832, 480),
    }
)


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

    Each rung is an unshifted noise level of the ``num_train_timesteps``-step
    training schedule (1000 for H3), shifted once per modality; the clean
    endpoint follows the last rung. The student starts from unit noise.
    """

    rungs: tuple[int, ...]
    video_shift: float
    audio_shift: float
    num_train_timesteps: float = NUM_TRAIN_TIMESTEPS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rungs, tuple)
            or not self.rungs
            or any(
                not _is_integer(rung)
                or not 0 < rung <= self.num_train_timesteps
                for rung in self.rungs
            )
            or any(
                left <= right for left, right in zip(self.rungs, self.rungs[1:])
            )
        ):
            raise ValueError(
                "H3 DMD rungs must be strictly decreasing integers within "
                "num_train_timesteps"
            )
        _check_shifts(self.video_shift, self.audio_shift)
        if not _is_positive_number(self.num_train_timesteps):
            raise ValueError(
                "H3 DMD num_train_timesteps must be finite and positive"
            )


@dataclass(frozen=True, slots=True)
class PddGrid:
    """A parallel-decoding student's fine grid and runtime blocks.

    The fine grid ``linspace(max_t, 0, intervals + 1)`` covers the unshifted
    times ``[0, max_t]``; each modality reaches its noise level through the
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
        canvases: The only canvases the checkpoint generates, keyed by the
            ``(short_edge, "W:H")`` a request selects them with, or None
            when it generates the canvas rule's canvases.
    """

    transformer: TransformerConfig
    schedule: Schedule
    attention: Attention
    tasks: tuple[str, ...]
    canvases: Mapping[tuple[int, str], image.Config] | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tasks, tuple)
            or not self.tasks
            or any(task not in TASK_PARTITIONS for task in self.tasks)
            or len(set(self.tasks)) != len(self.tasks)
        ):
            raise ValueError("H3 denoiser tasks must name distinct H3 tasks")
        if self.canvases is not None and (
            not isinstance(self.canvases, Mapping)
            or not self.canvases
            or any(
                not isinstance(value, image.Config)
                for value in self.canvases.values()
            )
            # A canvas is named once, so a raster's name is its own.
            or len(set(self.canvases.values())) != len(self.canvases)
        ):
            raise ValueError(
                "H3 denoiser canvases must name distinct image.Config values"
            )
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

    ``denoisers`` maps each DiT partition the checkpoint holds
    (``transformer`` or ``transformer_ref``) to its configuration; a
    deployment places one.
    """

    text_encoder: TextEncoderConfig
    denoisers: Mapping[str, DenoiserConfig]
    video_vae: video_vae.Config
    audio_vae: audio_vae.Config

    def __post_init__(self) -> None:
        if not isinstance(self.denoisers, Mapping) or not self.denoisers:
            raise ValueError("H3 checkpoints hold at least one denoiser")
        if set(self.denoisers) - set(TASK_PARTITIONS.values()):
            raise ValueError(
                "H3 denoisers must be named transformer or transformer_ref"
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


def _contract_field(contract: Mapping[str, Any], name: str) -> Any:
    """Return a field a FastH3 contract must state.

    Raises:
        ValueError: The contract omits the field.
    """
    if name not in contract:
        raise ValueError(
            "unsupported FastH3 checkpoint: fastvideo_inference.json "
            f"states no {name}"
        )
    return contract[name]


def _fasth3_attention(contract: Mapping[str, Any]) -> SparseAttention:
    """Describe a FastH3 student's sparse attention from its contract.

    A ref2va student states the keep rate of its reference videos
    (``vsa_ref_keep_rate``); a t2va student attends the generated video
    alone. ``SparseAttention`` refuses a tile no packing serves and a
    sparsity or keep rate outside its range.
    """
    return SparseAttention(
        tile=_contract_field(contract, "vsa_tile_size"),
        sparsity=_contract_field(contract, "vsa_sparsity"),
        reference_keep=contract.get("vsa_ref_keep_rate"),
    )


def _fasth3_schedule(
    contract: Mapping[str, Any],
    transformer: TransformerConfig,
    shifts: Mapping[str, float],
) -> Schedule:
    """Build a FastH3 student's schedule from its contract.

    A DMD student evaluates its trained rungs (``dmd_denoising_steps``) on
    the 1000-step training schedule. A parallel-decoding student has one
    output head per interval of its fine grid over ``grid_max_t`` and
    advances one block of it per evaluation, the blocks bounded by
    ``pdd_step_indices``. Each schedule refuses values outside its own
    domain.

    Raises:
        ValueError: The contract states neither schedule.
    """

    # A JSON list becomes the tuple a schedule holds; any other value
    # reaches the schedule as it is, which refuses it.
    def listed(value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    if "dmd_denoising_steps" in contract:
        return DmdLadder(
            rungs=listed(contract["dmd_denoising_steps"]),
            video_shift=shifts["video"],
            audio_shift=shifts["audio"],
        )
    if "pdd_step_indices" in contract:
        return PddGrid(
            intervals=transformer.output_heads,
            nodes=listed(contract["pdd_step_indices"]),
            video_shift=shifts["video"],
            audio_shift=shifts["audio"],
            max_t=_contract_field(contract, "grid_max_t"),
        )
    raise ValueError(
        "unsupported FastH3 checkpoint: fastvideo_inference.json states "
        "neither dmd_denoising_steps nor pdd_step_indices"
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
    ``audio_vae``, ``scheduler``, ``audio_scheduler`` and one entry per DiT
    partition the layout holds (its ``config.json``).
    """
    for name in (
        "text_encoder",
        "processor",
        "video_vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
        *layout.partitions,
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
    contract = layout.contract
    denoisers: dict[str, DenoiserConfig] = {}
    for partition, tasks in layout.partitions.items():
        values = metadata[partition]
        heads = values.get("pdd_steps", 1)
        if not _is_integer(heads) or heads < 1:
            raise ValueError(
                "unsupported MiniMax-H3 checkpoint: transformer pdd_steps "
                f"must be a positive integer, got {heads!r}"
            )
        transformer = _transformer(values, heads=heads)
        schedule: Schedule
        attention: Attention
        if contract is None:
            # The base release: the uniform grid and dense attention.
            schedule = UniformGrid(
                points=UNIFORM_GRID_POINTS,
                video_shift=shifts["video"],
                audio_shift=shifts["audio"],
            )
            attention = DenseAttention()
        else:
            schedule = _fasth3_schedule(contract, transformer, shifts)
            attention = _fasth3_attention(contract)
        denoisers[partition] = DenoiserConfig(
            transformer=transformer,
            schedule=schedule,
            attention=attention,
            tasks=tasks,
            canvases=DMD_CANVASES if isinstance(schedule, DmdLadder) else None,
        )
    return Config(
        text_encoder=_text_encoder(
            metadata["text_encoder"], metadata["processor"]
        ),
        denoisers=denoisers,
        video_vae=_video_vae(metadata["video_vae"]),
        audio_vae=_audio_vae(metadata["audio_vae"]),
    )


# Component sidecars, relative to the checkpoint root.
_SIDECARS = {
    "text_encoder": "text_encoder/config.json",
    "processor": "processor/preprocessor_config.json",
    "video_vae": "vae/config.json",
    "audio_vae": "audio_vae/config.json",
    "scheduler": "scheduler/scheduler_config.json",
    "audio_scheduler": "audio_scheduler/scheduler_config.json",
}


def read_config(root: Path, io, *, sources, base: Path | None = None) -> Config:
    """Read all architecture sidecars before any numerical module construction.

    ``root`` is the checkpoint directory, whose sidecars the loader has
    already fetched; ``io`` and ``sources`` (the resolved ``config_sources``,
    of which H3 declares none) are part of the package interface. ``base``
    is the snapshot of the base revision the checkpoint pins
    (``checkpoint.base_checkpoint``), resolved by the loader; a sidecar the
    checkpoint omits is read from it.

    An unreadable or missing sidecar raises ``OSError``; invalid JSON or an
    unsupported checkpoint raises ``ValueError``.
    """
    layout = detect(root)
    metadata: dict[str, Any] = {}
    for name, relative in (
        *_SIDECARS.items(),
        *(
            (partition, f"{partition}/config.json")
            for partition in layout.partitions
        ),
    ):
        directory = (
            root if base is None or (root / relative).is_file() else base
        )
        metadata[name] = json.loads(
            (directory / relative).read_text(encoding="utf-8")
        )
    return normalize(layout, metadata)


# Checkpoint tensor headers needed to resolve architecture before module
# selection. H3's architecture comes entirely from JSON sidecars.
config_sources = ()
