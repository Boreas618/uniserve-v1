"""An image denoiser's latent layout and the positions of its sequences.

A generated image is denoised as one *framed sequence* per guidance branch:
its latent patch rows in raster order, framed by an opening and a closing
token when the network frames images (``framing_tokens`` two, BAGEL's
start and end markers), otherwise alone (SenseNova). Every token of the
sequence carries three rotary coordinates ``(temporal, row, column)``: the
temporal coordinate is the branch's, and the spatial ones are a latent
patch's raster coordinates in the image's latent grid, zero for framing
tokens.

:class:`ImageLayout` states these facts of a loaded ``ImageDenoiser`` and
derives every per-image extent from them on the host.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from uniserve.models import ImageDenoiser
from uniserve.processing import BranchSource

__all__ = ["ImageLayout"]


@dataclass(frozen=True, slots=True)
class ImageLayout:
    """The latent layout of an image denoiser (``ImageDenoiser``).

    Attributes:
        patch_size: Native latent pixels per patch on each axis.
        latent_channels: Native latent channels.
        downsample: Output pixels per latent patch row on each axis.
        framing_tokens: Tokens framing a generated image: zero, or one
            opening and one closing token.
        image_unconditional: What the image-unconditional guidance branch
            conditions on.
        max_sequence_tokens: The longest image sequence, in latent patch
            rows; framing excluded.
        rope_advance: Language positions a generated image spans in its
            conversation: a framed image's closing token sits this far past
            the position before the image.
        guidance_branches: The most guidance branches one step evaluates.
        prediction_dtype: The dtype of the network's predictions.
    """

    patch_size: int
    latent_channels: int
    downsample: int
    framing_tokens: int
    image_unconditional: BranchSource
    max_sequence_tokens: int
    rope_advance: int
    guidance_branches: int
    prediction_dtype: torch.dtype

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.patch_size,
                self.latent_channels,
                self.downsample,
                self.max_sequence_tokens,
                self.guidance_branches,
            )
        ):
            raise ValueError(
                "an image layout's extents and branch count must be positive"
            )
        if self.framing_tokens not in (0, 2):
            raise ValueError(
                "an image denoiser frames an image with zero or two tokens"
            )
        if type(self.rope_advance) is not int or self.rope_advance < 1:
            raise ValueError("a generated image spans at least one position")

    @classmethod
    def of(cls, denoiser: ImageDenoiser) -> ImageLayout:
        """The layout a loaded image denoiser states."""
        return cls(
            patch_size=int(denoiser.patch_size),
            latent_channels=int(denoiser.latent_channels),
            downsample=int(denoiser.downsample),
            framing_tokens=int(denoiser.framing_tokens),
            image_unconditional=BranchSource(denoiser.image_unconditional),
            max_sequence_tokens=int(denoiser.max_sequence_tokens),
            rope_advance=int(denoiser.rope_advance),
            guidance_branches=int(denoiser.guidance_branches),
            prediction_dtype=denoiser.prediction_dtype,
        )

    @property
    def width(self) -> int:
        """Values per latent patch row."""
        return self.patch_size**2 * self.latent_channels

    @property
    def max_row_tokens(self) -> int:
        """Tokens of the longest framed sequence one row holds."""
        return self.max_sequence_tokens + self.framing_tokens

    def grid(self, height: int, width: int) -> tuple[int, int]:
        """The ``(rows, columns)`` latent patch grid of an image.

        Raises:
            ValueError: The image holds no latent patch.
        """
        rows, columns = height // self.downsample, width // self.downsample
        if not rows * columns:
            raise ValueError(
                "image dimensions must contain at least one latent patch"
            )
        return rows, columns

    def latent_rows(self, height: int, width: int) -> int:
        """Latent patch rows of an image (``ImageDenoiser.latent_shape``)."""
        rows, columns = self.grid(height, width)
        return rows * columns

    def sequence_length(self, height: int, width: int) -> int:
        """Tokens of an image's framed sequence: its latent rows and framing."""
        return self.latent_rows(height, width) + self.framing_tokens

    def positions(self, height: int, width: int, temporal: int) -> np.ndarray:
        """The ``int64 [3, tokens]`` positions of an image's framed sequence.

        Rows are the temporal, row and column axes. The temporal axis is
        ``temporal`` on every token, framing tokens included; the spatial
        axes hold each latent patch's raster coordinates in the latent grid
        and zero on framing tokens.
        """
        rows, columns = self.grid(height, width)
        count = rows * columns
        first = 1 if self.framing_tokens else 0
        positions = np.zeros((3, count + self.framing_tokens), np.int64)
        positions[0] = temporal
        latent = np.arange(count, dtype=np.int64)
        positions[1, first : first + count] = latent // columns
        positions[2, first : first + count] = latent % columns
        return positions

    def conditioning_positions(
        self, height: int, width: int, position: int
    ) -> np.ndarray:
        """The positions of a latent image inserted as conditioning.

        A latent ingest or feedback prefill places the image one position
        past the conditioning position ``position``: its opening framing
        token sits at ``position`` and its closing one ``rope_advance`` past
        it.

        Raises:
            ValueError: The denoiser does not frame images, so a latent image
                cannot be inserted.
        """
        if self.framing_tokens != 2:
            raise ValueError(
                "latent feature export requires framed image conditioning"
            )
        positions = self.positions(height, width, position + 1)
        positions[0, 0] = position
        positions[0, -1] = position + self.rope_advance
        return positions
