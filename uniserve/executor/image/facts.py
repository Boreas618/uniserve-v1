"""The image facts a text prompt is laid out with: vision blocks and grids.

A context prefill places each input image's encoder features as one block
of its sequence, and a feedback prefill places a generated image's features
after the image. The model's :class:`Injection` fixes the block's tokens and
positions:

- its layout (``uniserve.processing.FeatureLayout``): ``DIRECT`` places the
  features alone; ``FRAMED`` puts a start marker before them and an end
  marker after. A block that closes a generated image always ends with the
  end marker.
- its positions (``uniserve.processing.PositionLayout``): ``TEMPORAL``
  gives every token the conditioning position; ``SEQUENTIAL`` consecutive
  positions from it, as text tokens take; ``TEMPORAL_SPATIAL`` three axes
  (temporal, row, column), the feature tokens taking their raster
  coordinates in the feature grid and markers zero spatial coordinates. A
  closing block's tokens sit one temporal position further, and its end
  marker two past the conditioning position.

Feature tokens carry the placeholder token id 1; the text executor replaces
their embeddings with the features (the block's ``features``).

Spatial positions need an image's patch grid, which the model's vision
transform derives from the image's size and its request's input image
count. :class:`ImageFacts` asks it once per key and keeps it, with the other
facts the text executor lays out prompts with: whether text rows carry
three rotary axes and how many positions a generated image spans.
"""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from uniserve.executor.image.layout import ImageLayout
from uniserve.executor.text.context import VisionBlock
from uniserve.loading.rank import capability
from uniserve.models import ImageDenoiser
from uniserve.processing import FeatureLayout, PatchTransform, PositionLayout

__all__ = [
    "ImageFacts",
    "Injection",
    "PatchGrid",
    "feature_grid",
]

# The token id of a feature token, whose embedding the features replace.
_PLACEHOLDER = 1

# A model's patch grid of an image: ``(height, width, input_images)`` to
# ``(rows, columns)``, ``input_images`` 0 for a generated image.
PatchGrid = Callable[[int, int, int], Sequence[int]]


def feature_grid(
    grid: tuple[int, int] | None, feature_tokens: int
) -> tuple[int, int]:
    """The ``(rows, columns)`` grid of features pooled from a patch grid.

    The encoder pools a square of patches into each feature, so each axis
    of the raw patch grid ``grid`` is divided by the square root of the
    pooling that maps its patch count onto ``feature_tokens``.

    Raises:
        ValueError: There is no patch grid, or the feature count does not
            divide it into a square pooling.
    """
    if grid is None:
        raise ValueError(
            "temporal-spatial feature injection requires a patch image "
            "transform"
        )
    if feature_tokens < 1:
        raise ValueError("vision features must have shape [tokens, hidden]")
    raw_rows, raw_columns = grid
    pooling, remainder = divmod(raw_rows * raw_columns, feature_tokens)
    factor = math.isqrt(pooling)
    if remainder or factor == 0 or factor * factor != pooling:
        raise ValueError(
            "vision feature count does not align with its patch grid"
        )
    rows, columns = raw_rows // factor, raw_columns // factor
    if rows * columns != feature_tokens:
        raise ValueError("vision output grid is not integral")
    return rows, columns


@dataclass(frozen=True, slots=True)
class Injection:
    """A model's feature injection, with its marker tokens resolved.

    Attributes:
        layout: How the features enter the sequence.
        positions: The position coordinates of the block's tokens.
        start_token: The start marker's token id, or None when the model
            places none.
        end_token: The end marker's token id, or None when the model places
            none.
    """

    layout: FeatureLayout
    positions: PositionLayout
    start_token: int | None
    end_token: int | None

    def block(
        self,
        feature_tokens: int,
        position: int,
        close_image: bool,
        grid: tuple[int, int] | None,
    ) -> VisionBlock:
        """Lay out a block of ``feature_tokens`` features at ``position``.

        Args:
            feature_tokens: The image's feature count.
            position: The conditioning position the block follows.
            close_image: The block closes a generated image, so it ends
                with the end marker and sits one temporal position later.
            grid: The image's raw ``(rows, columns)`` patch grid before the
                encoder's pooling (:meth:`ImageFacts.grid`), required by
                ``TEMPORAL_SPATIAL`` positions.

        Raises:
            ValueError: The block has no features, a marker it places is
                unresolved, or the feature count does not pool the patch
                grid squarely.
        """
        if feature_tokens < 1:
            raise ValueError("vision features must have shape [tokens, hidden]")
        leading = self.layout is FeatureLayout.FRAMED
        trailing = leading or close_image
        tokens = int(leading) + feature_tokens + int(trailing)

        token_ids = [_PLACEHOLDER] * tokens
        if leading:
            token_ids[0] = _marker(self.start_token)
        if trailing:
            token_ids[-1] = _marker(self.end_token)
        begin = int(leading)
        features = range(begin, begin + feature_tokens)

        # One position per token on a single axis, or ``[3, tokens]``
        # temporal, row and column positions.
        if self.positions is PositionLayout.TEMPORAL:
            positions = np.full(tokens, position, np.int64)
        elif self.positions is PositionLayout.SEQUENTIAL:
            positions = np.arange(position, position + tokens, dtype=np.int64)
        else:
            # TEMPORAL_SPATIAL: feature tokens take their raster coordinates
            # in the pooled feature grid; markers keep zero spatial ones.
            _, columns = feature_grid(grid, feature_tokens)
            positions = np.zeros((3, tokens), np.int64)
            positions[0] = position + int(close_image)
            raster = np.arange(feature_tokens, dtype=np.int64)
            positions[1, begin : begin + feature_tokens] = raster // columns
            positions[2, begin : begin + feature_tokens] = raster % columns
            if close_image:
                # The end marker closing a generated image sits two past
                # the conditioning position, one past the image's tokens.
                positions[0, -1] = position + 2
        positions.flags.writeable = False
        return VisionBlock(
            np.asarray(token_ids, dtype=np.int64), features, positions
        )


def _marker(token: int | None) -> int:
    if token is None:
        raise ValueError(
            "feature injection requires a resolved marker token id"
        )
    return token


def _token(token: int | None) -> int | None:
    # A negative id stands for a marker the model does not place.
    return None if token is None or token < 0 else int(token)


def _vit_grid(
    processor: Any, height: int, width: int, input_images: int
) -> tuple[int, int]:
    """The understanding transform's ``(rows, columns)`` patch grid.

    ``processor`` is the model's image policy; ``input_images`` is the
    request's input image count for a request input image, which a shared
    pixel budget divides, or 0 for a generated image, which keeps the
    single-image pixel bound.

    Raises:
        ValueError: The model has no patch transform, or its resize policy
            rejects the dimensions.
    """
    transform = None if processor is None else processor.vit
    if not isinstance(transform, PatchTransform):
        raise ValueError(
            "temporal-spatial feature injection requires a patch image "
            "transform"
        )
    return transform.grid_shape(height, width, input_images or None)


class ImageFacts:
    """The image facts of a loaded model that text prompts are laid out with.

    Attributes:
        injection: How vision features enter a prompt, or None for a model
            that places none.
        layout: The image denoiser's latent layout, or None for a model
            without one.
    """

    __slots__ = ("_grids", "_patch_grid", "injection", "layout")

    def __init__(
        self,
        injection: Injection | None,
        layout: ImageLayout | None,
        patch_grid: PatchGrid,
    ) -> None:
        """State a model's image facts.

        Args:
            injection: The model's feature injection, or None.
            layout: The model's image denoiser layout, or None.
            patch_grid: The model's patch grid of an image, called as
                ``patch_grid(height, width, input_images)``; it may raise
                for an image its vision transform refuses.
        """
        self.injection = injection
        self.layout = layout
        self._patch_grid = patch_grid
        # The grid of every key asked so far. Prompts are laid out on one
        # thread, so the cache takes no lock.
        self._grids: dict[tuple[int, int, int], tuple[int, int]] = {}

    @classmethod
    def from_loaded(cls, loaded: Any) -> ImageFacts:
        """The image facts of a loaded model.

        The injection is the image policy's (``image_processor``), the
        layout the model's ``ImageDenoiser``'s, and the patch grid the
        policy's understanding transform's.
        """
        processor = loaded.image_processor
        declared = None if processor is None else processor.feature_injection
        injection = (
            None
            if declared is None
            else Injection(
                layout=FeatureLayout(declared.layout),
                positions=PositionLayout(declared.positions),
                start_token=_token(declared.start_token_id),
                end_token=_token(declared.end_token_id),
            )
        )
        # The capability is abstract; the lookup returns its concrete module.
        kind: type[Any] = ImageDenoiser
        denoiser = capability(loaded.module, kind)
        layout = None if denoiser is None else ImageLayout.of(denoiser)
        return cls(injection, layout, functools.partial(_vit_grid, processor))

    @property
    def rope_advance(self) -> int:
        """Language positions a generated image spans in its conversation.

        The image denoiser's (``ImageLayout.rope_advance``); 1 for a model
        without one.
        """
        return 1 if self.layout is None else self.layout.rope_advance

    def spatial_positions(self) -> bool:
        """Whether text rows carry three rotary axes.

        True for a model with an image denoiser, whose sequences place
        latent images spatially, or one placing vision features at
        temporal-spatial positions; other models read one temporal axis.
        """
        return self.layout is not None or (
            self.injection is not None
            and self.injection.positions is PositionLayout.TEMPORAL_SPATIAL
        )

    def grid(
        self, height: int, width: int, close_image: bool, input_images: int
    ) -> tuple[int, int]:
        """The vision transform's ``(rows, columns)`` patch grid of an image.

        A generated image (``close_image``) keeps the single-image pixel
        bound, so its grid ignores its request's ``input_images``; a
        request's input image takes the request's input image count. The
        grid of each ``(height, width, count)`` is asked once and kept.

        Raises:
            ValueError: The patch grid function refuses the image, or
                reports a grid that is not two positive extents.
        """
        key = (height, width, 0 if close_image else input_images)
        grid = self._grids.get(key)
        if grid is not None:
            return grid

        answer = self._patch_grid(*key)
        try:
            rows, columns = (operator.index(extent) for extent in answer)
        except (TypeError, ValueError):
            rows = columns = 0
        if rows < 1 or columns < 1:
            raise ValueError(
                f"the patch grid of {key} is not two positive extents: "
                f"{answer!r}"
            )
        grid = self._grids[key] = (rows, columns)
        return grid
