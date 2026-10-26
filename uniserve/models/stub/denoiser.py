"""Deterministic image prediction and diffusion schedules.

``Denoiser`` predicts exactly zero for every image sample while using the
library's real schedule, guidance and Euler solver. A zero prediction leaves
each sample unchanged through every solver step, and the noise scale is the
constant 1, so a generated image decodes to its initial normal noise draw.
"""

from __future__ import annotations

import torch

from uniserve.diffusion import (
    AdditiveGuidance,
    EulerSolver,
    NoiseScale,
    make_schedule,
)
from uniserve.models import (
    ImageDenoiser,
    TransformerDecoder,
)
from uniserve.models.inputs import PackedImageInput
from uniserve.nn.attention import Attention
from uniserve.tensors import OutputLayout, TensorOutput

from .config import Config


class Denoiser(ImageDenoiser):
    """Zero-valued image prediction over the real diffusion solver machinery.

    Latents are canonical patch rows of ``patch_size**2 * 3`` values, one row
    per ``patch_size`` pixel square: the layout ``Model.latent_encoder``
    produces and ``Model.image_decoder`` consumes.
    """

    framing_tokens = 2
    rope_advance = 2

    @property
    def max_sequence_tokens(self) -> int:
        return 1024

    def __init__(self, config: Config, backbone: TransformerDecoder):
        super().__init__(
            patch_size=config.patch_size,
            latent_channels=3,
            downsample=config.patch_size,
            noise_scale=NoiseScale(1.0, "constant", 1.0, 1.0),
            prediction_dtype=torch.bfloat16,
            solver=EulerSolver(),
        )
        self.backbone = backbone

    def make_schedules(self, steps, *, shift, device):
        return {
            "image": make_schedule(
                steps,
                shift=1.0 if shift is None else shift,
                direction="ascending",
                shift_domain="time",
                device=device,
            )
        }

    def make_guidance(
        self, *, text_scale, image_scale, interval, renorm, renorm_min
    ):
        return AdditiveGuidance(
            text_scale, image_scale, interval, renorm, renorm_min
        )

    def forward(self, inputs: PackedImageInput, *, state, constants, workspace):
        # The packed tokens size the K/V rows written below.
        tokens = inputs.rows.tokens
        width = self.patch_size**2 * self.latent_channels
        if inputs.source.shape[1] != width:
            raise ValueError(
                "simulation latents must be canonical image patch rows"
            )

        # [query tokens, kv heads, head dim] for the one-head, width-one cache.
        values = inputs.source.new_zeros((tokens, 1, 1))

        # The step's zero K/V publish through the backbone's single attention
        # layer, the cache ``model._Layer`` writes for text tokens. The
        # layer's table entry supplies the addresses; an entry without write
        # addresses leaves the cache untouched.
        cache = self.backbone.layers["0"].attention
        if not isinstance(cache, Attention):
            raise ValueError(
                "simulation backbone must publish through its scalar "
                "attention cache layer"
            )
        cache.update_cache(values, values, inputs.attention)

        # The prediction is exactly zero on every token.
        shape = (tokens, width)
        return {
            "image": (
                TensorOutput(
                    inputs.source.new_zeros(shape, dtype=self.prediction_dtype),
                    OutputLayout(
                        shape,
                        self.prediction_dtype,
                        tuple(slice(0, n) for n in shape),
                    ),
                ),
            )
        }
