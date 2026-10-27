"""SenseNova's adaptive flow head and convolutional image reconstruction.

The denoiser's backbone produces one hidden row per image token. The modules
here turn those rows into a clean-image prediction in canonical patch rows and
convert it into the flow-matching velocity the Euler solver consumes.

The MLP heads act on each token alone, so the velocity of a packed forward
is row-local. The convolutional ``Decoder`` mixes neighbouring tokens of
one image's token grid: it is an eager per-image region, evaluated from the
rows' host regions (``PackedImageRows.regions``), which a captured forward
runs between the graph segments around it.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.diffusion import NoiseScale
from uniserve.nn.functional import patchify
from uniserve.nn.linear import Linear
from uniserve.nn.timestep import TimestepEmbedding


@dataclass(frozen=True)
class HeadConfig:
    """Dimensions of the MLP velocity heads.

    Attributes:
        hidden_size: Residual width of the deep adaptive ``Head``, or the
            hidden width of the shallow two-layer MLP.
        num_layers: Checkpoint head depth. Without ``use_pixel_head``, more
            than two selects ``Head`` with this many residual blocks;
            otherwise ``Denoiser`` builds the shallow MLP.
        mlp_ratio: Expansion of each residual block's MLP; only ``Head``
            reads it.
    """

    hidden_size: int
    num_layers: int
    mlp_ratio: float

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.hidden_size, self.num_layers)
        ):
            raise ValueError(
                "flow head width and depth must be positive integers"
            )
        if (
            isinstance(self.mlp_ratio, bool)
            or not isinstance(self.mlp_ratio, (int, float))
            or not math.isfinite(self.mlp_ratio)
            or self.mlp_ratio <= 0
            or int(self.hidden_size * self.mlp_ratio) < 1
        ):
            raise ValueError(
                "flow head expansion must produce a positive finite width"
            )


@dataclass(frozen=True)
class Config:
    """Velocity head selection and initial-noise scaling.

    Attributes:
        head: MLP head dimensions, unused when ``use_pixel_head`` is set.
        use_pixel_head: Whether the convolutional ``Decoder`` replaces the
            MLP heads.
        add_noise_scale_embedding: Whether the denoiser adds an embedding of
            each image's noise scale to its token inputs.
        noise: Resolution-dependent scale of the initial normal draw, and
            the input of the optional noise-scale embedding.
    """

    head: HeadConfig
    use_pixel_head: bool
    add_noise_scale_embedding: bool
    noise: NoiseScale

    def __post_init__(self):
        if (
            type(self.use_pixel_head) is not bool
            or type(self.add_noise_scale_embedding) is not bool
        ):
            raise ValueError(
                "flow head selection and noise embedding must be boolean"
            )


class _Residual(nn.Module):
    """Apply adaptive shift/scale and a gated residual MLP."""

    function = "models.sensenova_u1.flow._Residual/1"

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        width = int(config.hidden_size * config.mlp_ratio)
        self.mlp = nn.Sequential(
            Linear(config.hidden_size, width),
            nn.SiLU(),
            Linear(width, config.hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), Linear(config.hidden_size, 3 * config.hidden_size)
        )

    def forward(self, hidden, time):
        shift, scale, gate = self.modulation(time).chunk(3, dim=-1)
        return hidden + gate * self.mlp(self.norm(hidden) * (1 + scale) + shift)


class _Output(nn.Module):
    """Project to patch values after adaptive shift/scale normalization."""

    function = "models.sensenova_u1.flow._Output/1"

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), Linear(hidden_size, 2 * hidden_size)
        )
        self.projection = Linear(hidden_size, output_size)

    def forward(self, hidden, time):
        shift, scale = self.modulation(time).chunk(2, dim=-1)
        return self.projection(self.norm(hidden) * (1 + scale) + shift)


class Head(nn.Module):
    """Predict clean patch values using the checkpoint's adaptive time network.

    Maps ``[tokens, input_size]`` rows and ``[tokens]`` timesteps to
    ``[tokens, output_size]``. The head embeds the timestep with its own
    ``TimestepEmbedding``, separate from the one the denoiser adds to the
    backbone input, and uses it to modulate every block.
    """

    function = "models.sensenova_u1.flow.Head/1"

    def __init__(
        self, config: HeadConfig, *, input_size: int, output_size: int
    ):
        super().__init__()
        self.config = config
        self.input = Linear(input_size, config.hidden_size)
        self.time_embedding = TimestepEmbedding(config.hidden_size)
        self.blocks = nn.ModuleList(
            _Residual(config) for _ in range(config.num_layers)
        )
        self.output = _Output(config.hidden_size, output_size)

    def forward(
        self, hidden: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input(hidden)
        time = self.time_embedding(timestep)
        for block in self.blocks:
            hidden = block(hidden, time)
        return self.output(hidden, time)


class Decoder(nn.Module):
    """Reconstruct image pixels with three channel-to-spatial rearrangements.

    Maps ``[1, input_size, rows, columns]`` token features to
    ``[1, out_channels, rows * 4 * final_upscale, columns * 4 *
    final_upscale]`` pixels: two 2x pixel shuffles around a convolution, then
    a convolution to ``out_channels * final_upscale**2`` channels and a final
    ``final_upscale`` shuffle.
    """

    function = "models.sensenova_u1.flow.Decoder/1"

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 1024,
        *,
        out_channels: int = 3,
        final_upscale: int = 8,
    ):
        super().__init__()
        if input_size % 4 or hidden_size % 4:
            raise ValueError(
                "pixel decoder widths must be divisible by "
                "the intermediate 2x2 shuffle"
            )
        self.blocks = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(input_size // 4, hidden_size, 3, padding=1),
            nn.GELU(),
            nn.PixelShuffle(2),
        )
        self.output = nn.Conv2d(
            hidden_size // 4, out_channels * final_upscale**2, 3, padding=1
        )
        self._final_upscale = final_upscale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(
            self.output(self.blocks(hidden)), self._final_upscale
        )


class Velocity(nn.Module):
    """Convert clean image predictions into velocity in canonical patch order.

    The predictor can be an adaptive Head, a shallow ordinary MLP, or
    Identity followed by the convolutional Decoder. The velocity is
    ``(prediction - sample) / (1 - t)`` with ``1 - t`` clamped to at least
    0.02, the reference implementation's default endpoint epsilon.
    """

    function = "models.sensenova_u1.flow.Velocity/1"

    def __init__(self, head: nn.Module, decoder: nn.Module, *, patch_size: int):
        super().__init__()
        self.head, self.decoder, self.patch_size = head, decoder, patch_size

    def forward(
        self,
        hidden: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        regions: tuple[tuple[int, int, int], ...] | None,
    ) -> torch.Tensor:
        """Predict each token's FP32 velocity.

        ``hidden`` is ``[tokens, text_hidden]``, ``timesteps`` the
        ``[tokens]`` FP32 network time of each token's row and ``sample``
        the ``[tokens, 3 * patch_size**2]`` canonical patch rows each token's
        velocity starts from. ``regions`` are the host ``(first token, grid
        rows, grid columns)`` of each image's tokens, which only the
        convolutional decoder reads; tokens outside every region predict
        zero there.

        Raises:
            ValueError: The decoder evaluates without host regions, or a
                region lies outside the tokens.
        """
        if isinstance(self.decoder, Decoder):
            prediction = self.decode(hidden, regions)
        elif isinstance(self.head, Head):
            prediction = self.decoder(self.head(hidden, timesteps))
        else:
            prediction = self.decoder(self.head(hidden))

        # Schedules hold FP32 timesteps, so the division returns FP32 even
        # when the difference is BF16. FP32 is the declared
        # ``prediction_dtype``, the dtype of the buffers in which the other
        # pipeline stages receive the broadcast prediction.
        return (prediction - sample) / (1 - timesteps[:, None]).clamp_min(0.02)

    def decode(
        self,
        hidden: torch.Tensor,
        regions: tuple[tuple[int, int, int], ...] | None,
    ) -> torch.Tensor:
        """Reconstruct each image's clean patch rows with the pixel decoder.

        An eager per-image region: each image's token rows form its
        ``[1, text_hidden, rows, columns]`` grid, which the decoder turns
        into pixels and ``patchify`` back into its tokens' patch rows.
        """
        if regions is None:
            raise ValueError(
                "the pixel decoder is an eager per-image region and reads "
                "the rows' host regions"
            )
        tokens = hidden.shape[0]
        prediction = None
        for first, rows, columns in regions:
            if first < 0 or first + rows * columns > tokens:
                raise ValueError("an image region lies outside the tokens")
            spatial = (
                self.head(hidden[first : first + rows * columns])
                .reshape(1, rows, columns, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            patches = patchify(
                self.decoder(spatial), patch_size=self.patch_size
            )[0]
            if prediction is None:
                prediction = patches.new_zeros((tokens, patches.shape[1]))
            prediction[first : first + rows * columns] = patches
        if prediction is None:
            # No image: every token is padding.
            return hidden.new_zeros((tokens, 3 * self.patch_size**2))
        return prediction
