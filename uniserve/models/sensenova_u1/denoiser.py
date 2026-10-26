"""SenseNova U1 image prediction, guidance and diffusion schedules.

SenseNova denoises in pixel space. A sample is one image stored as canonical
patch rows ``[tokens, 3 * stride**2]``, where ``stride`` is the vision patch
size times the vision downsampling factor, so every row is one backbone token
and holds that token's ``stride x stride`` RGB block in pixel, then channel
order (``uniserve.nn.functional.patchify``). There is no VAE: the model's
``ImageDecoder`` only unfolds the patch rows back into pixels.

Each step re-encodes the current sample through this module's own vision
tower (``vision.TokenEncoder``), adds timestep (and optionally noise-scale)
embeddings, and runs every image token through the ``flow`` experts of the
backbone shared with the text model. The image tokens attend to their
guidance branch's prefix in the prefix cache. A velocity head then turns the
backbone output into the solver's velocity (``flow.Velocity``).

A packed forward is row-local: each token's sample row is its own pixel
block, which the encoder reads alone (``vision.TokenEncoder``),
and the MLP velocity heads act on each token alone. The convolutional pixel
decoder (``use_pixel_head``) is the exception, an eager per-image region
over each image's token grid (``flow.Velocity.decode``).
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.diffusion import (
    AdditiveGuidance,
    EulerSolver,
    Renorm,
    make_schedule,
)
from uniserve.models import ImageDenoiser
from uniserve.models.inputs import PackedImageInput
from uniserve.nn.linear import Linear
from uniserve.nn.routing import IndexedRoutes
from uniserve.nn.timestep import TimestepEmbedding
from uniserve.tensors import OutputLayout, TensorOutput

from . import flow, vision
from .config import Config
from .transformer import Transformer


class Denoiser(ImageDenoiser):
    """Predict image velocity from pixels through the shared backbone.

    The backbone is the text model's, whose ``flow`` experts the image
    tokens take.
    """

    # The prompt's ``<img>`` token sits at the position before the image, the
    # patches share the next one and ``</img>`` takes the one after.
    rope_advance = 2

    @property
    def max_sequence_tokens(self) -> int:
        # ``framing_tokens`` keeps the base default of zero, so this bounds
        # the image's patch tokens alone; the opening ``<img>`` token belongs
        # to the prompt prefix (``processing.flow_prompt``).
        return self.config.max_image_seq_len

    def __init__(self, config: Config, backbone: Transformer):
        # Output pixels per backbone token on each spatial axis. Samples are
        # pixel-space patches, so the patch size and downsampling both equal
        # this stride and a token row carries 3 * stride**2 values.
        stride = config.vision.patch_size * round(
            1 / config.vision.downsample_ratio
        )
        super().__init__(
            patch_size=stride,
            latent_channels=3,
            downsample=stride,
            noise_scale=config.flow.noise,
            prediction_dtype=torch.float32,
            solver=EulerSolver("velocity"),
        )
        # ``backbone`` is the same instance as ``Model.text.backbone``; its
        # weights load once, through the backbone mapping of
        # ``weights.checkpoint_mappings``. The input encoder has its own
        # generation weights, distinct from ``Model.vision_encoder``.
        self.config, self.backbone = config, backbone
        self.input = vision.TokenEncoder(config.vision)
        self.time_embedding = TimestepEmbedding(config.text.hidden_size)
        self.noise_embedding = (
            TimestepEmbedding(config.text.hidden_size)
            if config.flow.add_noise_scale_embedding
            else None
        )

        # Three checkpoint head variants: a convolutional pixel decoder, a deep
        # adaptive time head, or a shallow two-layer MLP, exactly one of which
        # is active per checkpoint. ``use_pixel_head`` takes precedence over
        # the head depth. The decoder's two internal 2x shuffles account for
        # a factor of four of the stride; ``final_upscale`` supplies the rest.
        if config.flow.use_pixel_head:
            head: nn.Module = nn.Identity()
            decoder: nn.Module = flow.Decoder(
                config.text.hidden_size, final_upscale=stride // 4
            )
        elif config.flow.head.num_layers > 2:
            head = flow.Head(
                config.flow.head,
                input_size=config.text.hidden_size,
                output_size=3 * stride**2,
            )
            decoder = nn.Identity()
        else:
            head = nn.Sequential(
                Linear(config.text.hidden_size, config.flow.head.hidden_size),
                nn.GELU(),
                Linear(config.flow.head.hidden_size, 3 * stride**2),
            )
            decoder = nn.Identity()
        self.prediction = flow.Velocity(head, decoder, patch_size=stride)

    def make_schedules(self, steps, *, shift, device):
        # Network time rises from pure noise (0) to the clean image (1), the
        # direction ``flow.Velocity``'s ``1 - t`` denominator assumes. A
        # requested shift applies in the sigma domain; none means no shift.
        return {
            "image": make_schedule(
                steps,
                shift=1.0 if shift is None else shift,
                direction="ascending",
                shift_domain="sigma",
                device=device,
            )
        }

    def make_guidance(
        self,
        *,
        text_scale: float,
        image_scale: float,
        interval: tuple[float, float],
        renorm: Renorm,
        renorm_min: float,
    ):
        return AdditiveGuidance(
            text_scale, image_scale, interval, renorm, renorm_min
        )

    def forward(self, inputs: PackedImageInput, *, state, constants, workspace):
        """Predict every token's velocity for one packed forward.

        Returns ``{"image": (velocity,)}``: a ``TensorOutput`` holding the
        FP32 ``[tokens, 3 * stride**2]`` velocity on the last pipeline
        stage, and ``None`` on every other stage, which receives the
        prediction by broadcast from the last stage. A latent token's row is
        its velocity from its sample row; framing and padding tokens' rows
        are finite and unspecified. ``state``, ``constants`` and
        ``workspace`` are unused.

        Raises:
            ValueError: The source rows are not canonical patch rows, or the
                pixel decoder evaluates without the rows' host regions. The
                backbone and velocity head raise their own shape errors.
        """
        rows = inputs.rows
        if inputs.source.shape[1] != 3 * self.patch_size**2:
            raise ValueError("SenseNova reads canonical pixel patch rows")
        # Each token's sample row; tokens other than latent ones read source
        # row 0, whose results nothing reads.
        samples = inputs.source.index_select(0, rows.sample_index.clamp_min(0))

        # Only the first pipeline stage builds input embeddings; later stages
        # pass None and the backbone receives hidden states from the previous
        # stage. A mesh without a ``pp`` axis selects the rank-local group.
        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        hidden = None
        if pipeline.rank == 0:
            # Each token's pixel block at its (row, column) in its image's
            # token grid.
            hidden = self.input(samples, rows.positions[1:].T)

            # Every token receives its row's timestep and, when the
            # checkpoint enables it, its row's noise scale normalized by the
            # configured maximum, in the samples' dtype.
            hidden = hidden + self.time_embedding(rows.timesteps)
            if self.noise_embedding is not None:
                scales = rows.noise_scale.to(samples.dtype)[rows.token_row]
                hidden = hidden + self.noise_embedding(
                    scales / self.noise_scale.maximum
                )

        # Every packed token takes the flow experts; text tokens exist only
        # as the cached prefix the shared attention reads.
        hidden = self.backbone(
            hidden,
            rows.positions,
            inputs.attention,
            routes=IndexedRoutes("flow"),
        )
        if pipeline.rank != pipeline.size - 1:
            return {"image": (None,)}

        # The velocity's origin is each token's solver sample.
        velocity = self.prediction(
            hidden, rows.timesteps, samples, rows.regions
        )
        shape = (rows.tokens, 3 * self.patch_size**2)
        return {
            "image": (
                TensorOutput(
                    velocity,
                    OutputLayout(
                        shape, velocity.dtype, tuple(slice(0, n) for n in shape)
                    ),
                ),
            )
        }
