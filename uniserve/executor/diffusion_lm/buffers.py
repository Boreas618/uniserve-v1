"""The device columns and resident banks of a diffusion LM lane.

:class:`Dimensions` sizes one lane's columns from its bounds and the
model's facts; :class:`Columns` holds them in one zeroed device arena, which
every captured graph addresses for the lane's life. ``R`` is the packed
sequences of one forward, ``T`` its tokens, ``Tb`` the numerical block
tables, ``W`` the widest table's page columns, ``H`` the hidden width, ``C``
the block length, ``K`` the candidates one ``readout`` selects
(:data:`~.shapes.CANDIDATES`) and ``Rd`` the blocks one denoising step
holds:

| Column | Element | Shape |
| --- | --- | --- |
| ``input_ids``, ``positions`` | int64 | ``[T]`` |
| ``row_columns`` | int64 | ``[(6 + Tb) * (R + 1) + T]`` |
| ``slot_rows`` | int64 | ``[T]`` |
| ``finished_rows`` | int64 | ``[256]`` |
| ``candidate_index``, ``candidate_words`` | int64 | ``[K]`` |
| ``candidate_logprobs`` | float32 | ``[K]`` |
| ``attend_state.i`` (final stage) | compute | ``[T, attend_widths[i]]`` |
| ``finished`` (final stage) | compute | ``[T, H]`` |
| ``denoise.columns`` (generating) | int64 | ``[4, Rd]`` |
| ``denoise.relays`` (generating) | int64 | ``[Rd]`` |
| ``denoise.tokens`` (generating) | int64 | ``[Rd, C]`` |
| ``denoise.history`` (generating) | int64 | ``[Rd, depth, C]`` |
| ``denoise.self_conditioning`` (generating) | embedding | ``[Rd, C, H]`` |
| ``denoise.results`` (generating) | int64 | ``[Rd, 1 + C]`` |
| ``denoise.continued`` (generating) | uint8 | ``[Rd]`` |

The host writes a forward's tokens, positions and packed attention rows
(``row_columns``, which ``AttentionMetadata.gather`` expands on the device
into the attention metadata beside the columns), a readout's slot rows,
finished rows and flat candidate indices, and a denoising step's columns
(slot, seed bits, block and step per row) and relay index. The forwards
write the attend state, the finished rows, the candidate log-probabilities
(whose 32-bit patterns ``candidate_words`` widens to output words) and a
step's results and continuation flags.

:class:`Banks` holds a generating lane's resident per-slot state: each
slot's block, the argmax blocks the stopping rule compares, the
self-conditioning embedding of its latest distribution and whether its block
continues. Slot zero is the padding sentinel: the denoiser's commits skip
it, so it stays zero, and a padding row of a step gathers it
(:meth:`Banks.gather`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from uniserve.executor.arena import (
    COLUMN_ALIGNMENT,
    aligned,
)
from uniserve.nn.attention.gather import ROW_SECTIONS
from uniserve.runner.attention import AttentionMetadata

from .shapes import CANDIDATES, SLOT_BUCKETS

__all__ = ["BankShape", "Banks", "Columns", "Dimensions"]

# A column: its name, element type and shape.
Column = tuple[str, torch.dtype, tuple[int, ...]]


def _nbytes(dtype: torch.dtype, shape: Sequence[int]) -> int:
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def _arena_bytes(columns: Sequence[Column]) -> int:
    return sum(
        aligned(max(_nbytes(dtype, shape), 1)) for _, dtype, shape in columns
    )


def _allocate(
    columns: Sequence[Column], device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Allocate ``columns`` in one zeroed arena on the current stream.

    Each column starts at a ``COLUMN_ALIGNMENT`` boundary
    (:mod:`uniserve.executor.arena`), so its element view of the arena's
    bytes is aligned for any element type.
    """
    arena = torch.zeros(_arena_bytes(columns), dtype=torch.uint8, device=device)
    views: dict[str, torch.Tensor] = {}
    offset = 0
    for name, dtype, shape in columns:
        size = _nbytes(dtype, shape)
        views[name] = arena[offset : offset + size].view(dtype).view(shape)
        offset += aligned(max(size, 1))
    return arena, views


@dataclass(frozen=True, slots=True)
class Dimensions:
    """The dimensions of one diffusion LM lane's columns.

    Attributes:
        rows: Packed sequences of one forward ``R``, padding included.
        tokens: Packed tokens of one forward ``T``, padding included.
        tables: Numerical block tables ``Tb``.
        width: The widest table's page columns ``W``.
        hidden: The hidden width ``H``.
        attend_widths: Per-token widths of the attend state, on the final
            pipeline stage; empty elsewhere, where no ``finish`` runs.
        compute_dtype: The attend state's and finished rows' element type.
        length: Tokens of one block ``C``.
        denoise_rows: Blocks one denoising step holds ``Rd``; zero on a
            lane that serves readouts only.
        depth: The argmax blocks a slot's stopping rule compares (the
            served stability threshold).
        embedding_dtype: The self-conditioning embeddings' element type.
    """

    rows: int
    tokens: int
    tables: int
    width: int
    hidden: int
    attend_widths: tuple[int, ...]
    compute_dtype: torch.dtype
    length: int
    denoise_rows: int
    depth: int
    embedding_dtype: torch.dtype

    def row_words(self) -> int:
        """Return the words of the packed attention rows' column."""
        return (ROW_SECTIONS + self.tables) * (self.rows + 1) + self.tokens

    def columns(self) -> list[Column]:
        """Return every column: its name, element type and shape."""
        tokens, i64 = self.tokens, torch.int64
        columns: list[Column] = [
            ("input_ids", i64, (tokens,)),
            ("positions", i64, (tokens,)),
            ("row_columns", i64, (self.row_words(),)),
            ("slot_rows", i64, (tokens,)),
            ("finished_rows", i64, (SLOT_BUCKETS[-1],)),
            ("candidate_index", i64, (CANDIDATES,)),
            ("candidate_logprobs", torch.float32, (CANDIDATES,)),
            ("candidate_words", i64, (CANDIDATES,)),
        ]
        for index, width in enumerate(self.attend_widths):
            columns.append(
                (f"attend_state.{index}", self.compute_dtype, (tokens, width))
            )
        if self.attend_widths:
            columns.append(
                ("finished", self.compute_dtype, (tokens, self.hidden))
            )
        if self.denoise_rows:
            rows, length = self.denoise_rows, self.length
            columns += [
                ("denoise.columns", i64, (4, rows)),
                ("denoise.relays", i64, (rows,)),
                ("denoise.tokens", i64, (rows, length)),
                ("denoise.history", i64, (rows, self.depth, length)),
                (
                    "denoise.self_conditioning",
                    self.embedding_dtype,
                    (rows, length, self.hidden),
                ),
                ("denoise.results", i64, (rows, 1 + length)),
                ("denoise.continued", torch.uint8, (rows,)),
            ]
        return columns

    def bytes(self) -> int:
        """Return the device bytes of the columns and the attention metadata."""
        return _arena_bytes(self.columns()) + AttentionMetadata.device_bytes(
            self.rows, self.tokens, self.tables, self.width
        )


class Columns:
    """A lane's columns in one zeroed device arena.

    Captured graphs address the columns, so the arena lives as long as the
    lane. The arena is allocated on the current stream.

    Args:
        dimensions: The columns' dimensions.
        device: The device the lane computes on.
    """

    def __init__(self, dimensions: Dimensions, device: torch.device) -> None:
        self.dimensions = dimensions
        self.arena, self._views = _allocate(dimensions.columns(), device)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._views[name]

    @property
    def attend_state(self) -> list[torch.Tensor]:
        """The attend state ``attend`` writes and ``finish`` completes."""
        return [
            self._views[f"attend_state.{index}"]
            for index in range(len(self.dimensions.attend_widths))
        ]


@dataclass(frozen=True, slots=True)
class BankShape:
    """The resident banks of a generating lane.

    Attributes:
        slots: Request slots, the padding slot zero excluded.
        length: Tokens of one block ``C``.
        depth: The argmax blocks a slot's stopping rule compares.
        hidden: The self-conditioning embedding width ``H``.
        dtype: The self-conditioning embeddings' element type.
    """

    slots: int
    length: int
    depth: int
    hidden: int
    dtype: torch.dtype

    def columns(self) -> list[Column]:
        """Return each bank: its name, element type and shape, one row per
        slot and the padding slot.
        """  # noqa: D205
        rows, length = self.slots + 1, self.length
        return [
            ("canvas", torch.int64, (rows, length)),
            ("history", torch.int64, (rows, self.depth, length)),
            ("self_conditioning", self.dtype, (rows, length, self.hidden)),
            ("live", torch.uint8, (rows,)),
        ]

    def slot_bytes(self) -> int:
        """Return the bank bytes of one slot: one row of every bank."""
        return sum(
            _nbytes(dtype, shape[1:]) for _, dtype, shape in self.columns()
        )

    def fixed_bytes(self) -> int:
        """Return what the banks allocate beyond :meth:`slot_bytes` per slot.

        The padding slot's row and, at most, one alignment of every bank.
        """
        return self.slot_bytes() + len(self.columns()) * COLUMN_ALIGNMENT


class Banks:
    """A generating lane's resident per-slot block state, zeroed.

    Slot zero is the padding sentinel and stays zero: the denoiser's commits
    skip it. The banks are allocated on the current stream in one arena;
    the denoiser receives them once, at construction.

    Args:
        shape: The banks' dimensions.
        device: The device the lane computes on.
    """

    def __init__(self, shape: BankShape, device: torch.device) -> None:
        _, views = _allocate(shape.columns(), device)
        self.canvas = views["canvas"]
        self.history = views["history"]
        self.self_conditioning = views["self_conditioning"]
        self.live = views["live"]

    def gather(
        self,
        slots: torch.Tensor,
        rows: int,
        canvas: torch.Tensor,
        history: torch.Tensor,
        self_conditioning: torch.Tensor,
    ) -> None:
        """Gather the rows of ``slots[:rows]`` into a step's state rows.

        Row ``i`` of ``canvas``, ``history`` and ``self_conditioning``
        receives bank row ``slots[i]`` on the current stream, behind the
        earlier commits to the banks. A padding row names slot zero and
        reads its zero state. The caller checked the slots on the host
        before staging them and holds ``rows`` within the state rows.
        """
        index = slots[:rows]
        for bank, target in (
            (self.canvas, canvas),
            (self.history, history),
            (self.self_conditioning, self_conditioning),
        ):
            torch.index_select(bank, 0, index, out=target[:rows])
