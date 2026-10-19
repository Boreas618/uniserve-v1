"""MiniMax-H3 configurations of the released checkpoint families.

The released architecture values are stated once, by the sidecars of the
checkpoint fixture ``fixtures/models/fasth3``; the factories below read them
with the production reader and apply per-test overrides.
"""

from dataclasses import replace
from functools import cache
from pathlib import Path

from torch import nn

from uniserve.loading import Config as IOConfig
from uniserve.media import image
from uniserve.models import ComponentEntry, EntryPoint
from uniserve.models.minimax_h3 import (
    Config,
    DenoiserConfig,
    DenseAttention,
    DmdLadder,
    PddGrid,
    SparseAttention,
    TransformerConfig,
    UniformGrid,
    VideoPostprocessor,
    audio_vae,
    video_vae,
)
from uniserve.models.minimax_h3.config import read_config
from uniserve.models.minimax_h3.encoder import TextEncoderConfig

CHECKPOINT = Path(__file__).parent / "models/fasth3"

# The 16:9 canvas FastH3 DMD exports generate at 768p.
WIDE = image.Config(768, 1344)


@cache
def _released() -> Config:
    return read_config(CHECKPOINT, IOConfig(), sources={})


def transformer_config(**overrides) -> TransformerConfig:
    """The released DiT dimensions, with ``overrides`` applied."""
    transformer = _released().denoisers["transformer"].transformer
    return replace(transformer, **overrides)


def text_encoder_config(**overrides) -> TextEncoderConfig:
    """The released text encoder dimensions, with ``overrides`` applied."""
    return replace(_released().text_encoder, **overrides)


def video_vae_config(**overrides) -> video_vae.Config:
    """The released video VAE configuration, with ``overrides`` applied."""
    return replace(_released().video_vae, **overrides)


def audio_vae_config(**overrides) -> audio_vae.Config:
    """The released audio VAE configuration, with ``overrides`` applied."""
    return replace(_released().audio_vae, **overrides)


class DenoisingModel(nn.Module):
    """Serve a loaded H3 denoiser with its output sampling clock."""

    def __init__(self, denoiser):
        super().__init__()
        self.config = denoiser.config
        self.denoiser = denoiser
        self.video_postprocessor = VideoPostprocessor(frame_rate=24)


def entry_points(config):
    return {
        "denoiser": ComponentEntry("denoiser", (EntryPoint("forward"),)),
    }


def dmd_denoiser(
    transformer: TransformerConfig | None = None,
    *,
    rungs: tuple[int, ...] = (999, 749, 500, 250),
    video_shift: float = 12.0,
    sparsity: float = 0.9,
) -> DenoiserConfig:
    """A FastH3 DMD student: sparse tile-64 attention, t2va and fl2va, 16:9.

    It serves the tasks of the ``transformer`` partition it replaces.
    """
    return DenoiserConfig(
        transformer=transformer or transformer_config(),
        schedule=DmdLadder(
            rungs=rungs, video_shift=video_shift, audio_shift=3.0
        ),
        attention=SparseAttention(tile=64, sparsity=sparsity),
        tasks=("t2va", "fl2va"),
        canvases={(768, "16:9"): WIDE},
    )


def base_denoiser(
    transformer: TransformerConfig | None = None,
    *,
    tasks: tuple[str, ...] = ("t2va", "fl2va"),
    points: int = 50,
) -> DenoiserConfig:
    """A released diffusers DiT: dense attention over the uniform grid."""
    return DenoiserConfig(
        transformer=transformer or transformer_config(),
        schedule=UniformGrid(points=points, video_shift=12.0, audio_shift=3.0),
        attention=DenseAttention(),
        tasks=tasks,
        canvases=None,
    )


def omniref_denoiser(
    transformer: TransformerConfig | None = None,
) -> DenoiserConfig:
    """A FastH3 OmniRef PDD student: 32 heads, multi-segment tile-128 VSA."""
    return DenoiserConfig(
        transformer=replace(
            transformer or transformer_config(), output_heads=32
        ),
        schedule=PddGrid(
            intervals=32,
            nodes=(0, 4, 8, 12, 16, 20, 24, 28, 32),
            video_shift=12.0,
            audio_shift=3.0,
            max_t=0.999,
        ),
        attention=SparseAttention(tile=128, sparsity=0.9, reference_keep=0.1),
        tasks=("ref2va",),
        canvases=None,
    )


def fasth3_config() -> Config:
    """A FastH3 DMD export: one denoiser serving t2va and fl2va."""
    return Config(
        text_encoder=text_encoder_config(),
        denoisers={"transformer": dmd_denoiser()},
        video_vae=video_vae_config(),
        audio_vae=audio_vae_config(),
    )


def base_config() -> Config:
    """The base release: both DiT partitions."""
    return Config(
        text_encoder=text_encoder_config(),
        denoisers={
            "transformer": base_denoiser(),
            "transformer_ref": base_denoiser(tasks=("ref2va",)),
        },
        video_vae=video_vae_config(),
        audio_vae=audio_vae_config(),
    )
