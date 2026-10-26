"""The rows of a packed image denoiser forward and their device expansion.

A *row* is one image's framed sequence in a denoiser forward: one guidance
branch of a stepping image, or one latent image written as conditioning
(:class:`ImageRow`). A forward packs its rows' tokens back to back, so the
host describes it per row, in O(rows) work: :class:`RowColumns` holds one
int64 column per :class:`RowField` and one float32 column per
:class:`RowValue`. :meth:`RowColumns.padded` fits the rows into a bucket of
``R`` rows and ``T`` tokens: the live rows first, then empty rows of no
tokens, then one padding row that holds every remaining token.

:func:`expand_rows` derives the per-token metadata a network reads
(``uniserve.models.inputs.PackedImageRows``) and the prediction row of every
token from the device copy of the columns, at static shapes, so it is the
first work of every captured forward. :func:`pack_attention` builds the
rows' attention row column.

Padding and empty rows are inert: they read no sample (``sample_index``
-1), predict into the dump row, attend to no prefix, write no cache entry
and carry timestep and noise scale zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from uniserve.executor.image.layout import ImageLayout
from uniserve.executor.text.rows import pack_rows
from uniserve.models.inputs import PackedImageRows, TokenKind

__all__ = [
    "ExpandedRows",
    "ImageRow",
    "RowColumns",
    "RowField",
    "RowValue",
    "expand_rows",
    "pack_attention",
]


class RowField(IntEnum):
    """The int64 columns of :class:`RowColumns`, one value per row."""

    # The row's first token in the packed forward, and its token count.
    TOKEN_START = 0
    TOKENS = 1
    # Framing tokens around the row's latents: 0, or 2 (one opening and one
    # closing token).
    FRAMING = 2
    # The row's image: its index among the images of the batch.
    IMAGE = 3
    # The row's first latent row in the forward's source, and its latent
    # rows; zero for padding and empty rows.
    SAMPLE = 4
    UNITS = 5
    # The image's latent grid; zero for padding and empty rows.
    GRID_ROWS = 6
    GRID_COLUMNS = 7
    # Temporal positions of the opening token, the latent tokens and the
    # closing token.
    OPEN_POSITION = 8
    LATENT_POSITION = 9
    CLOSE_POSITION = 10
    # The row's first prediction row.
    PREDICTION = 11
    # The cache slot the row attends to, its visible prefix there, and
    # whether the row writes its keys and values after the prefix.
    SLOT = 12
    PREFIX = 13
    WRITE = 14


class RowValue(IntEnum):
    """The float32 columns of :class:`RowColumns`, one value per row."""

    # The network time of the row's step; zero for conditioning.
    TIMESTEP = 0
    # The scale of the image's initial noise draw.
    NOISE_SCALE = 1


@dataclass(frozen=True, slots=True)
class ImageRow:
    """One live row of a packed forward.

    Attributes:
        image: The row's image among the images of the batch.
        sample: The image's first latent row in the forward's source.
        height: Image height in pixels.
        width: Image width in pixels.
        positions: Temporal positions of the opening token, the latent
            tokens and the closing token. A denoising branch places all
            three at the branch's temporal position; a latent image written
            as conditioning at ``(p, p + 1, p + rope_advance)``
            (``ImageLayout.conditioning_positions``).
        timestep: The network time the row is evaluated at.
        noise_scale: The scale of the image's initial noise draw.
        prediction: The row's first prediction row.
        slot: The cache slot the row attends to.
        prefix: The slot's visible prefix the row attends to.
        write: Whether the row writes its keys and values to the slot.
    """

    image: int
    sample: int
    height: int
    width: int
    positions: tuple[int, int, int]
    timestep: float
    noise_scale: float = 0.0
    prediction: int = 0
    slot: int = 0
    prefix: int = 0
    write: bool = False


@dataclass(frozen=True, slots=True)
class RowColumns:
    """The host columns of a packed forward's rows.

    Attributes:
        fields: ``int64 [len(RowField), R]`` the :class:`RowField` columns.
        values: ``float32 [len(RowValue), R]`` the :class:`RowValue`
            columns.
        live: The live rows, which lead; the rest are empty rows, then the
            padding row.
    """

    fields: np.ndarray
    values: np.ndarray
    live: int

    def __post_init__(self) -> None:
        rows = self.fields.shape[1] if self.fields.ndim == 2 else -1
        if (
            self.fields.shape != (len(RowField), rows)
            or self.values.shape != (len(RowValue), rows)
            or self.fields.dtype != np.int64
            or self.values.dtype != np.float32
        ):
            raise ValueError(
                "row columns hold one int64 column per field and one "
                "float32 column per value, all of one row count"
            )
        if not 0 <= self.live <= rows:
            raise ValueError("live rows must lie within the row columns")
        tokens = self.column(RowField.TOKENS)
        starts = np.cumsum(tokens) - tokens
        if np.any(tokens < 0) or np.any(
            self.column(RowField.TOKEN_START) != starts
        ):
            raise ValueError("rows must pack their tokens back to back")

    @classmethod
    def of(cls, layout: ImageLayout, rows: Sequence[ImageRow]) -> RowColumns:
        """Columns of live rows packed back to back, without padding.

        Raises:
            ValueError: An image holds no latent patch or exceeds the
                denoiser's longest sequence.
        """
        fields = np.zeros((len(RowField), len(rows)), np.int64)
        values = np.zeros((len(RowValue), len(rows)), np.float32)
        start = 0
        for index, row in enumerate(rows):
            grid = layout.grid(row.height, row.width)
            units = grid[0] * grid[1]
            if units > layout.max_sequence_tokens:
                raise ValueError(
                    f"a {row.height}x{row.width} image exceeds the "
                    f"denoiser's {layout.max_sequence_tokens} latent rows"
                )
            tokens = units + layout.framing_tokens
            fields[:, index] = (
                start,
                tokens,
                layout.framing_tokens,
                row.image,
                row.sample,
                units,
                *grid,
                *row.positions,
                row.prediction,
                row.slot,
                row.prefix,
                int(row.write),
            )
            values[:, index] = (row.timestep, row.noise_scale)
            start += tokens
        return cls(fields, values, len(rows))

    @property
    def rows(self) -> int:
        """Rows of the columns, live, empty and padding."""
        return self.fields.shape[1]

    @property
    def tokens(self) -> int:
        """Tokens of every row together."""
        return int(self.column(RowField.TOKENS).sum())

    @property
    def live_tokens(self) -> int:
        """Tokens of the live rows."""
        return int(self.column(RowField.TOKENS)[: self.live].sum())

    def column(self, field: RowField) -> np.ndarray:
        """One field's ``int64 [R]`` column."""
        return self.fields[field]

    def padded(self, tokens: int, rows: int, row_tokens: int) -> RowColumns:
        """The live rows fitted into a bucket of ``rows`` rows and ``tokens``.

        Empty rows of no tokens follow the live rows, and the last row is
        the padding row, holding the bucket's remaining tokens. Padding and
        empty rows attend to slot 0 without prefix, write nothing, read no
        latent and carry zero positions, timestep and noise scale.
        ``row_tokens`` is the longest sequence one row holds
        (``ImageLayout.max_row_tokens``), which bounds every row's queries
        in a captured forward, the padding row's too.

        Raises:
            ValueError: The bucket has no row or no token to spare (a bucket
                serves fewer rows and tokens than it holds), or its padding
                exceeds a row's longest sequence.
        """
        live_tokens = self.live_tokens
        if rows <= self.live or tokens <= live_tokens:
            raise ValueError(
                f"a bucket of {rows} rows and {tokens} tokens cannot pad "
                f"{self.live} rows of {live_tokens} tokens"
            )
        if tokens - live_tokens > row_tokens:
            raise ValueError(
                f"{tokens - live_tokens} padding tokens exceed the "
                f"{row_tokens} tokens one row holds"
            )
        fields = np.zeros((len(RowField), rows), np.int64)
        values = np.zeros((len(RowValue), rows), np.float32)
        fields[:, : self.live] = self.fields[:, : self.live]
        values[:, : self.live] = self.values[:, : self.live]
        fields[RowField.TOKENS, -1] = tokens - live_tokens
        fields[RowField.TOKEN_START] = np.concatenate(
            ([0], np.cumsum(fields[RowField.TOKENS])[:-1])
        )
        return RowColumns(fields, values, self.live)

    def regions(self) -> tuple[tuple[int, int, int], ...]:
        """Host ``(first latent token, grid rows, grid columns)`` per row.

        One entry per row holding latents, in row order: what an eager
        per-image region (``PackedImageRows.regions``) reads.
        """
        fields = self.fields
        start = fields[RowField.TOKEN_START] + (fields[RowField.FRAMING] > 0)
        return tuple(
            (
                int(start[row]),
                int(fields[RowField.GRID_ROWS, row]),
                int(fields[RowField.GRID_COLUMNS, row]),
            )
            for row in np.flatnonzero(fields[RowField.UNITS])
        )

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The columns as host ``(int64 fields, float32 values)`` tensors."""
        return torch.from_numpy(self.fields), torch.from_numpy(self.values)


@dataclass(frozen=True, slots=True)
class ExpandedRows:
    """The per-token metadata of a forward's rows.

    Attributes:
        rows: What the network reads (``PackedImageRows``).
        prediction_index: ``int64 [T]`` each token's prediction row: a
            latent token's row of its image's predictions, the dump row for
            every other token.
    """

    rows: PackedImageRows
    prediction_index: torch.Tensor


def expand_rows(
    fields: torch.Tensor,
    values: torch.Tensor,
    *,
    tokens: int,
    dump: int,
    regions: tuple[tuple[int, int, int], ...] | None = None,
) -> ExpandedRows:
    """Expand the rows' columns into per-token metadata on their device.

    ``fields`` and ``values`` are the device copies of
    :class:`RowColumns`' columns; their rows pack exactly ``tokens`` tokens
    back to back. Within a row, token ``local`` is the opening token when
    the row is framed and ``local`` is 0, the closing token when it is
    framed and ``local`` is its last token, and otherwise latent row ``j =
    local - framed`` while ``j`` lies below the row's latent rows; every
    other token is padding. Latent row ``j`` reads source row ``sample + j``
    at grid coordinates ``(j // columns, j % columns)`` and predicts into
    row ``prediction + j``; other tokens read nothing (-1), sit at spatial
    coordinates zero and predict into ``dump``.

    Every shape follows from ``tokens`` and the row count, and no device
    value is read on the host, so the expansion is captured with the
    forward it feeds. ``regions`` are the host regions the network's eager
    per-image work reads (:meth:`RowColumns.regions`), when it has any.
    """
    device = fields.device
    rows = fields.shape[1]
    column = {field: fields[field] for field in RowField}

    # Each token's row, and its index within the row.
    token_row = torch.repeat_interleave(
        torch.arange(rows, device=device),
        column[RowField.TOKENS],
        output_size=tokens,
    )
    local = (
        torch.arange(tokens, device=device)
        - column[RowField.TOKEN_START][token_row]
    )
    framed = column[RowField.FRAMING][token_row] > 0
    latent_row = local - framed.long()
    latent = (latent_row >= 0) & (
        latent_row < column[RowField.UNITS][token_row]
    )
    opening = framed & (local == 0)
    closing = framed & (local == column[RowField.TOKENS][token_row] - 1)

    kind = torch.full_like(local, TokenKind.PAD)
    kind = torch.where(opening, TokenKind.OPEN, kind)
    kind = torch.where(closing, TokenKind.CLOSE, kind)
    kind = torch.where(latent, TokenKind.LATENT, kind)

    # Raster coordinates in the latent grid; rows without latents have no
    # columns, which the clamp keeps from dividing by zero.
    columns = column[RowField.GRID_COLUMNS][token_row].clamp_min(1)
    zero = torch.zeros_like(local)
    temporal = torch.where(
        opening,
        column[RowField.OPEN_POSITION][token_row],
        torch.where(
            closing,
            column[RowField.CLOSE_POSITION][token_row],
            torch.where(
                latent, column[RowField.LATENT_POSITION][token_row], zero
            ),
        ),
    )
    positions = torch.stack(
        (
            temporal,
            torch.where(latent, latent_row // columns, zero),
            torch.where(latent, latent_row % columns, zero),
        )
    )

    sample_index = torch.where(
        latent, column[RowField.SAMPLE][token_row] + latent_row, -1
    )
    prediction_index = torch.where(
        latent, column[RowField.PREDICTION][token_row] + latent_row, dump
    )
    timesteps = torch.where(
        kind != TokenKind.PAD,
        values[RowValue.TIMESTEP][token_row],
        values.new_zeros(()),
    )
    row_grid = torch.stack(
        (column[RowField.GRID_ROWS], column[RowField.GRID_COLUMNS]), dim=1
    )
    return ExpandedRows(
        PackedImageRows(
            sample_index=sample_index,
            kind=kind,
            positions=positions,
            timesteps=timesteps,
            token_row=token_row,
            row_grid=row_grid,
            noise_scale=values[RowValue.NOISE_SCALE].clone(),
            regions=regions,
        ),
        prediction_index,
    )


def pack_attention(
    columns: RowColumns,
    pages: Sequence[np.ndarray],
    stride: int,
    tokens: int,
) -> np.ndarray:
    """The attention row column of the rows (``AttentionMetadata.gather``).

    Each row attends to its slot's visible prefix with its tokens as
    queries, and writes them to the slot when its ``WRITE`` field is set.
    ``pages`` holds each block table's selected page counts per live row;
    padding and empty rows select none.

    Args:
        columns: The rows, padded to their bucket.
        pages: Each table's page counts of the live rows.
        stride: Section words: the binding's rows plus one.
        tokens: Token-row words: the binding's tokens.
    """
    return pack_rows(
        slots=columns.column(RowField.SLOT),
        prefixes=columns.column(RowField.PREFIX),
        queries=columns.column(RowField.TOKENS),
        writes=columns.column(RowField.WRITE).astype(bool),
        pages=[np.asarray(counts)[: columns.live] for counts in pages],
        stride=stride,
        tokens=tokens,
    )
