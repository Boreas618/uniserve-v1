"""SenseNova's NEO patch projection, axial RoPE and dense spatial reduction.

``Encoder`` is the whole vision tower: a patch convolution, a 2D rotary
rotation of the patch features and a strided convolution that merges each
``factor x factor`` block of patches into one backbone token. The model
instantiates it twice with separate weights: as the network wrapped by
``Model.vision_encoder`` for input images, and, as a ``TokenEncoder``, as
``Denoiser.input`` for the image being denoised.

Both convolutions have a stride equal to their kernel and the rotary
rotation acts on one patch, so a backbone token's features depend only on
its own ``stride x stride`` pixel block and its position in the token grid:
``TokenEncoder`` encodes token blocks with no image extent at all.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.nn.functional import apply_rotary
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.vision.position import (
    build_abs_positions_from_grid_hw,
    merged_squares,
)


def _finite(value: object) -> bool:
    """Return whether ``value`` is a finite real number other than a bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class Config:
    """NEO vision tower dimensions.

    Attributes:
        hidden_size: Patch feature width; the rotary splits it into a column
            half and a row half.
        output_size: Token width after the dense reduction, which the model
            config requires to equal the text hidden size.
        downsample_ratio: Reciprocal of the dense reduction factor per axis.
        patch_size: Input pixels per patch on each axis.
        num_channels: Input image channels.
        rope_theta: Base of the patch rotary frequencies.
    """

    hidden_size: int
    output_size: int
    downsample_ratio: float
    patch_size: int
    num_channels: int
    rope_theta: float

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    self.hidden_size,
                    self.output_size,
                    self.patch_size,
                    self.num_channels,
                )
            )
            or self.hidden_size % 4
        ):
            raise ValueError(
                "NEO patch dimensions must be positive and its rotary "
                "width divisible by four"
            )
        if (
            not _finite(self.downsample_ratio)
            or not 0 < self.downsample_ratio <= 1
            or round(1 / self.downsample_ratio) * self.downsample_ratio != 1
        ):
            raise ValueError(
                "NEO downsampling must be the reciprocal of a positive integer"
            )
        if not _finite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("NEO rotary theta must be finite and positive")


class _Network(nn.Module):
    """The parameters of the NEO vision tower, which both encoders share.

    A patch convolution, the rotary embedding of patch-grid coordinates and
    a strided convolution reducing each ``factor x factor`` block of patches
    to one backbone token.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        factor = round(1 / config.downsample_ratio)
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            config.hidden_size, config.output_size, factor, stride=factor
        )
        self.rotary = RotaryEmbedding(
            config.hidden_size // 2, theta=config.rope_theta
        )
        self.activation = nn.GELU()


class Encoder(_Network):
    """Encode NCHW pixels or flattened CHW patches with explicit image grids.

    Columns rotate the first channel half and rows rotate the second, using
    interleaved pairs in FP32 before returning to the patch convolution dtype.
    Spatial reduction never crosses an image boundary.

    ``pixels`` is either ``[images, channels, H, W]`` or packed patches
    ``[patches, channels * patch_size**2]`` in CHW order, raster order within
    each image. ``grids`` holds the same ``(rows, columns)`` patch grids as
    the host ``grid_shapes``: the device copy yields rotary coordinates
    without a host readback, which keeps that step capturable in a CUDA
    graph, and the host copy drives validation and reshapes. Returns
    ``[tokens, output_size]`` with each image's downsampled grid in raster
    order, images concatenated.
    """

    def forward(
        self,
        pixels: torch.Tensor,
        grids: torch.Tensor,
        grid_shapes: tuple[tuple[int, int], ...] | None,
    ) -> torch.Tensor:
        """Encode images, or patch rows packed in fixed slots.

        With ``grid_shapes`` None the computation reads the grids only as
        device values: ``pixels`` are packed patch rows, ``grids`` cover
        them in order (images and the padding segments of a packed batch,
        a ``(0, 0)`` grid an empty segment, every side a multiple of the
        reduction factor), and the shapes depend only on the number of rows
        and grids, so a padded packing (``PatchEncoder.encode_segments``)
        replays one captured call. Each image's tokens are those it has
        alone.
        """
        counts = (
            None
            if grid_shapes is None
            else tuple(height * width for height, width in grid_shapes)
        )
        if (
            grids.ndim != 2
            or grids.shape[1] != 2
            or grids.dtype not in (torch.int32, torch.int64)
            or (counts is not None and grids.shape[0] != len(counts))
        ):
            raise ValueError(
                "NEO grids require one integer height/width pair per image"
            )
        factor = round(1 / self.config.downsample_ratio)
        if grid_shapes is not None and any(
            min(shape) < 1 or any(axis % factor for axis in shape)
            for shape in grid_shapes
        ):
            raise ValueError(
                "NEO image grids must align with dense spatial downsampling"
            )

        if pixels.ndim == 2:
            # Packed patches arrive as [total_patches, channels*patch*patch].
            if pixels.shape[1] != (
                self.config.num_channels * self.config.patch_size**2
            ) or (counts is not None and pixels.shape[0] != sum(counts)):
                raise ValueError(
                    "NEO packed CHW pixels must cover the declared image grids"
                )
            pixels = pixels.reshape(
                -1,
                self.config.num_channels,
                self.config.patch_size,
                self.config.patch_size,
            )
        # Without host grids only packed patch rows are accepted.
        packed = grid_shapes is None
        if pixels.ndim != 4 or (
            packed and pixels.shape[2:] != (self.config.patch_size,) * 2
        ):
            raise ValueError(
                "NEO pixels must be NCHW images or flattened CHW patches"
            )

        features = self.activation(
            self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
        )
        # [patches, hidden, h, w] -> [total_patches, hidden] across all images.
        features = features.permute(0, 2, 3, 1).reshape(
            -1, self.config.hidden_size
        )
        if counts is not None and features.shape[0] != sum(counts):
            raise ValueError("NEO image grids must cover all projected patches")
        if packed and features.shape[0] % factor**2:
            raise ValueError(
                "NEO packed patch rows must hold whole reduction squares"
            )

        columns, rows = build_abs_positions_from_grid_hw(
            grids, total=features.shape[0]
        )
        parts = []
        for hidden, coordinates, axis in zip(
            features.float().chunk(2, dim=-1),
            (columns, rows),
            (1, 0),
            strict=True,
        ):
            # Without host grids, every coordinate lies below the row count.
            cosine, sine = self.rotary(
                coordinates,
                dtype=torch.float32,
                sequence_length=features.shape[0]
                if grid_shapes is None
                else max((shape[axis] for shape in grid_shapes), default=0),
            )
            parts.append(
                apply_rotary(
                    hidden.unsqueeze(1), cosine, sine, rotation="interleaved"
                ).squeeze(1)
            )
        features = torch.cat(parts, dim=-1).to(features.dtype)

        if grid_shapes is None:
            # Each token's factor x factor square of patch rows, gathered from
            # the device grids, reduced by the same convolution.
            squares = merged_squares(
                grids, features.shape[0] // factor**2, factor
            )
            blocks = (
                features[squares]
                .reshape(-1, factor, factor, self.config.hidden_size)
                .permute(0, 3, 1, 2)
            )
            return self.dense_embedding(blocks).reshape(
                -1, self.config.output_size
            )
        if not grid_shapes:
            return features.new_empty((0, self.config.output_size))

        # Equal grids reduce in one batched convolution; mixed grids loop.
        if all(shape == grid_shapes[0] for shape in grid_shapes):
            height, width = grid_shapes[0]
            spatial = features.reshape(
                len(grid_shapes), height, width, -1
            ).permute(0, 3, 1, 2)
            return (
                self.dense_embedding(spatial)
                .permute(0, 2, 3, 1)
                .reshape(-1, self.config.output_size)
            )
        outputs = []
        for values, (height, width) in zip(
            features.split([height * width for height, width in grid_shapes]),
            grid_shapes,
            strict=True,
        ):
            spatial = values.reshape(1, height, width, -1).permute(0, 3, 1, 2)
            outputs.append(
                self.dense_embedding(spatial)
                .permute(0, 2, 3, 1)
                .reshape(-1, self.config.output_size)
            )
        return torch.cat(outputs)


class TokenEncoder(_Network):
    """The vision tower over packed token blocks rather than whole images.

    The same network as :class:`Encoder`, with the same parameters, whose
    call encodes each backbone token from its own pixel block, so any
    packing of images' tokens encodes in one call of static shape. A module
    call, it reaches a tower placed on another device through the execution
    context's transfers.
    """

    def forward(
        self, token_pixels: torch.Tensor, token_rc: torch.Tensor
    ) -> torch.Tensor:
        """Encode each backbone token from its own pixel block.

        ``token_pixels`` is ``[tokens, channels * stride**2]``: each token's
        ``stride x stride`` pixel block (``stride`` the patch size times the
        reduction factor) in pixel, then channel order, the layout of
        ``uniserve.nn.functional.patchify``. ``token_rc`` is ``[tokens, 2]``
        integer ``(row, column)`` coordinates of each token in its image's
        token grid. Returns ``[tokens, output_size]`` features, each equal
        to what ``Encoder.forward`` gives that token of its whole image: the
        token's ``factor x factor`` patches rotate by their patch-grid
        coordinates ``(row * factor + i, column * factor + j)``. Tokens are
        independent, so any packing of any images' tokens, with arbitrary
        finite padding tokens, encodes in one call of static shape.
        """
        config = self.config
        factor = round(1 / config.downsample_ratio)
        patch, channels = config.patch_size, config.num_channels
        tokens = token_pixels.shape[0]
        if (
            token_pixels.ndim != 2
            or token_pixels.shape[1] != channels * (factor * patch) ** 2
            or token_rc.shape != (tokens, 2)
        ):
            raise ValueError(
                "NEO token blocks require [tokens, channels * stride**2] "
                "pixels and [tokens, 2] grid coordinates"
            )

        # [tokens, (i, a), (j, b), channels] pixel blocks, with i, j a patch
        # of the block and a, b a pixel of the patch, into one CHW patch per
        # (token, i, j) in raster order.
        patches = (
            token_pixels.reshape(tokens, factor, patch, factor, patch, channels)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(-1, channels, patch, patch)
        )
        features = self.activation(
            self.patch_embedding(patches.to(self.patch_embedding.weight.dtype))
        ).reshape(-1, config.hidden_size)

        # Patch-grid coordinates of every (token, i, j) patch.
        offsets = torch.arange(factor, device=token_rc.device)
        rows = token_rc[:, 0, None, None] * factor + offsets[None, :, None]
        columns = token_rc[:, 1, None, None] * factor + offsets[None, None, :]
        parts = []
        for hidden, coordinates in zip(
            features.float().chunk(2, dim=-1),
            (
                columns.expand(-1, factor, factor).reshape(-1),
                rows.expand(-1, factor, factor).reshape(-1),
            ),
            strict=True,
        ):
            cosine, sine = self.rotary(
                coordinates,
                dtype=torch.float32,
                sequence_length=features.shape[0],
            )
            parts.append(
                apply_rotary(
                    hidden.unsqueeze(1), cosine, sine, rotation="interleaved"
                ).squeeze(1)
            )
        features = torch.cat(parts, dim=-1).to(features.dtype)

        # Each token's factor x factor patches reduce to one feature row.
        blocks = features.reshape(
            tokens, factor, factor, config.hidden_size
        ).permute(0, 3, 1, 2)
        return self.dense_embedding(blocks).reshape(tokens, config.output_size)
