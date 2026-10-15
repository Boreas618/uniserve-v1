"""Image latent posterior sampling and canonical patch serialization."""

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.functional import patchify, unpatchify

from .layers import DiagonalGaussian


class PatchAutoencoder(nn.Module):
    """Compose numerical encoder/decoder modules with latent patch coordinates.

    downsample is the output-pixel stride of one latent patch token. Scale and
    shift normalize posterior samples before their canonical patch conversion.
    The posterior generator and every input tensor belong to the caller.
    """

    value_range = (0.0, 1.0)

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        posterior: DiagonalGaussian,
        *,
        patch_size: int,
        latent_channels: int,
        latent_dtype: torch.dtype,
        downsample: int,
        scale: float,
        shift: float,
    ):
        super().__init__()
        self.encoder, self.decoder, self.posterior = encoder, decoder, posterior
        if (
            any(
                type(value) is not int or value < 1
                for value in (patch_size, latent_channels, downsample)
            )
            or scale <= 0
        ):
            raise ValueError(
                "latent patch dimensions and scale must be positive"
            )
        self.patch_size, self.latent_channels = patch_size, latent_channels
        self.latent_dtype, self.downsample = latent_dtype, downsample
        self.scale, self.shift = scale, shift

    @property
    def samples_posterior(self) -> bool:
        """Whether encoding samples the posterior rather than taking its
        mean, and so takes noise.
        """  # noqa: D205
        return bool(self.posterior.sample)

    def posterior_shape(self, height: int, width: int) -> tuple[int, int, int]:
        """The ``(channels, height, width)`` posterior of one image of
        ``height`` x ``width`` pixels.

        One latent patch spans ``downsample`` pixels and ``patch_size``
        posterior positions on each axis.
        """  # noqa: D205
        stride = self.downsample // self.patch_size
        return (self.latent_channels, height // stride, width // stride)

    def encode(
        self,
        pixels: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample NCHW pixels into normalized latent patch rows.

        A sampled posterior takes standard normal ``noise`` of shape
        ``[N, *posterior_shape(height, width)]`` in any floating dtype (it
        is converted to the moments' dtype), or draws from ``generator``.
        """
        if pixels.ndim != 4:
            raise ValueError("latent encoding requires NCHW pixels")

        dtype = next(self.encoder.parameters()).dtype
        moments = self.encoder(pixels.to(dtype))
        if noise is not None:
            noise = noise.to(moments.dtype)
        latents = self.scale * (
            self.posterior(moments, generator=generator, noise=noise)
            - self.shift
        )

        # Trim to complete latent patches before serialization.
        height = pixels.shape[-2] // self.downsample * self.patch_size
        width = pixels.shape[-1] // self.downsample * self.patch_size
        return self.patchify(latents[:, :, :height, :width]).to(
            self.latent_dtype
        )

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        return patchify(latents, patch_size=self.patch_size)

    def unpatchify(
        self, patches: torch.Tensor, size: image.Config
    ) -> torch.Tensor:
        latent_size = image.Config(
            size.height // self.downsample * self.patch_size,
            size.width // self.downsample * self.patch_size,
        )
        return unpatchify(
            patches,
            latent_size,
            patch_size=self.patch_size,
            channels=self.latent_channels,
        )

    def decode(self, patches: torch.Tensor, size: image.Config) -> torch.Tensor:
        """Restore patch rows to clamped [0, 1] pixels of the given image
        size.
        """  # noqa: D205
        latents = self.unpatchify(patches, size)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)

        dtype = next(self.decoder.parameters()).dtype
        latents = latents.to(dtype) / self.scale + self.shift
        pixels = self.decoder(latents)
        return (pixels * 0.5 + 0.5).clamp(0, 1)


class RGBDecoder(nn.Module):
    """Restore signed RGB pixels from canonical spatial patch rows."""

    value_range = (-1.0, 1.0)

    def __init__(self, patch_size: int):
        super().__init__()
        if type(patch_size) is not int or patch_size < 1:
            raise ValueError("RGB patch size must be positive")
        self.patch_size = patch_size

    def decode(self, patches: torch.Tensor, size: image.Config) -> torch.Tensor:
        pixels = unpatchify(
            patches, size, patch_size=self.patch_size, channels=3
        )
        return pixels.unsqueeze(0) if pixels.ndim == 3 else pixels
