"""Shared FP32 latent normalization after an ordinary encoder call."""

from __future__ import annotations

import torch
from torch import nn

from .layers import DiagonalGaussian
from .tiled import TiledEncoder


class LatentEncoder(nn.Module):
    """Call the composed numerical encoder and normalize channel statistics.

    The inverse of ``LatentDecoder``. ``encoder`` maps its native input to
    posterior moments, or directly to latents when ``posterior`` is None; a
    ``TiledEncoder`` encodes its raster as overlapping tiles. The posterior
    turns the moments into latents, which pass through ``latent_dtype``, the
    representation in which the VAE's reference keeps its latents, before
    ``(latent - mean) / std`` in FP32. Mean and standard deviation broadcast
    against the latent. Derived constants retain real values during meta
    construction; loading moves their backing with the encoder.
    """

    function = "nn.vae.encoder.LatentEncoder/1"

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        encoder: nn.Module,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        posterior: DiagonalGaussian | None = None,
        latent_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if mean.is_meta or std.is_meta or mean.shape != std.shape:
            raise ValueError("latent statistics require real matching tensors")
        if (
            not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or not bool((std > 0).all())
        ):
            raise ValueError(
                "latent statistics must be finite with positive standard "
                "deviations"
            )
        if not latent_dtype.is_floating_point:
            raise ValueError("latent representation must be floating point")
        self.encoder, self.posterior = encoder, posterior
        self.latent_dtype = latent_dtype
        self.register_buffer("mean", mean.float(), persistent=False)
        self.register_buffer("std", std.float(), persistent=False)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        rows: slice | None = None,
        window: tuple[slice, ...] | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the normalized FP32 latent of ``inputs``.

        ``rows`` asks for a band of latent rows along the height axis alone,
        which a ``TiledEncoder`` computes from the tiles the band depends
        on; None encodes every row. ``window`` then indexes the encoder
        output in native latent coordinates and keeps only that region, such
        as the latent frames of an input that was padded to the encoder's
        temporal extent; the posterior samples the kept region alone.
        ``noise`` is the standard normal draw a sampling posterior adds, with
        the kept mean's shape; without it the posterior draws its own.

        Raises:
            ValueError: ``rows`` is given to an encoder that is not a
                ``TiledEncoder``, or the latent statistics live elsewhere.
        """
        if isinstance(self.encoder, TiledEncoder):
            values = self.encoder.encode(inputs, tiled=True, rows=rows)
        elif rows is not None:
            raise ValueError("a band of latent rows needs a tiled encoder")
        else:
            values = self.encoder(inputs)
        if window is not None:
            values = values[window]
        if self.posterior is not None:
            values = self.posterior(values, noise=noise)
        elif noise is not None:
            raise ValueError("a deterministic encoder takes no posterior noise")
        if (
            values.device != self.mean.device
            or self.std.device != values.device
        ):
            raise ValueError(
                "latent values and normalization statistics must share "
                "their device"
            )

        latents = values.to(self.latent_dtype).float()
        return (latents - self.mean) / self.std
