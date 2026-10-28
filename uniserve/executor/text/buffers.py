"""Bounds, device columns and host staging of a text computation's runners.

:class:`TextBounds` derives every bound of a computation's runners from the
worker configuration and the prefix cache's groups: the row and token
capacity of their columns, the block tables and the page width each graph
addresses, and the decode and prefill graph buckets. It is derived once for
the manifest (without the fitted pool, so it bounds what binding allocates)
and again at binding.

:class:`Dimensions` sizes one runner's columns; :class:`InputBuffers` holds
them in one device arena, which every captured graph addresses for the
runner's life. ``R`` is the row capacity, ``T`` the token capacity, ``Tb``
the numerical block tables, ``B`` the int32 words of a token set, ``L`` the
logprob words a forward may report and ``P`` the prompt rows a forward may
score, padded to whole calls:

| Column | Element | Shape |
| --- | --- | --- |
| ``decode_inputs`` (:class:`DecodeInputs`) | bytes | ``R * 109`` |
| ``select_inputs`` (:class:`SelectInputs`) | bytes | ``R * 86`` |
| ``input_ids`` | int64 | ``[T]`` |
| ``positions`` (axis-major) | int64 | ``[3, T]`` |
| ``row_columns`` | int64 | ``[(6 + Tb) * (R + 1) + T]`` |
| ``causal_values`` | int32 | ``[R]`` |
| ``logit_rows`` | int64 | ``[T]`` |
| ``candidate_rows``, ``candidate_mask_rows`` | int64 | ``[T]`` |
| ``candidate_draws`` | float32 | ``[T]`` |
| ``candidate_params`` | float32 | ``[T, 8]`` |
| ``select_drafts`` | int64 | ``[T]`` |
| ``select_masks`` | int32 | ``[R, B]`` |
| ``token_sets`` (finish, transitions) | int32 | ``[2, R, B]`` |
| ``set_indices``, ``set_values`` | int64, int32 | ``[S]`` |
| ``logprob_tasks``, ``logprob_counts`` | int64 | ``[R]`` |
| ``verified_logprobs`` (task, candidate row, top count) | int64 | ``[3, T]`` |
| ``prompt_rows`` (field-major) | int64 | ``[8, P]`` |
| ``prompt_block`` | int32 | ``[1]`` |
| ``commit`` (field-major) | int64 | ``[8, R]`` |
| ``slot_writes`` (:class:`SlotWrite` pairs) | int64 | ``[4, R]`` |
| ``relays`` (:class:`Relay` pairs) | int64 | ``[8, R]`` |
| ``logits`` | model | ``[R, local vocab]`` |
| ``completion`` | int64 | ``[4, R]`` |
| ``continuation``, ``transition`` | uint8 | ``[R]`` |
| ``tagged``, ``points``, ``scored`` | int64 | ``[R]`` |
| ``logprobs`` | int64 | ``[L + 1]`` |
| ``prompt_tail`` | model | ``[R + 1, vocab]`` |
| ``hidden`` | model | ``[T, hidden]`` |
| ``input_embeddings`` (multimodal) | model | ``[T, hidden]`` |
| ``embedding_mask`` (multimodal) | uint8 | ``[T]`` |

A runner that drafts (``D`` draft tokens a call proposes at most, the
deployment's draft limit) adds the columns of its draft pass
(:meth:`~uniserve.runner.causal_lm.CausalLMRunner.draft`), whose chained
steps' windows hold ``Wd = 2 * D * R`` tokens at most:

| Column | Element | Shape |
| --- | --- | --- |
| ``draft_row_columns`` | int64 | as ``row_columns`` |
| ``draft_window_columns`` | int64 | as ``row_columns`` |
| ``draft_input_ids``, ``draft_positions`` | int64 | ``[T]`` |
| ``draft_hidden_rows`` | int64 | ``[T]`` |
| ``draft_pairs``, ``draft_proposing`` | int64 | ``[2, R]`` |
| ``draft_window_tokens`` | int64 | ``[R]`` |
| ``draft_window_owners``, ``draft_window_positions`` | int64 | ``[Wd]`` |
| ``proposals`` | int64 | ``[R, D]`` |

The host writes the inputs: a decode's rows, a selection's tasks, a
prefill's tokens, positions and row columns, the candidate rows of a
verification, the distinct token sets rows mask with, the nonzero words
(``S`` of them at most) of each task's finish and transition sets, prompt
and commit rows, and index pairs: a decode's host tokens and positions by
slot, a prefill's relayed tokens, the relayed outputs and prompt tails. The
forwards and selections write the outputs: candidate logits, completion
fields, relay values, logprob words (the last word takes the scores no one
reports), retained prompt logits (the last row is spare) and hidden states.

The two input regions regroup the host-written columns a decode or a
selection reads, so each costs one host-to-device copy: a region's layout
depends on its bucket ``b`` (all of ``b`` rows' slots, then all their finish
tokens, ...), and the graph of bucket ``b`` addresses that layout's views.

:class:`Staging` carries host values to the device columns: one page-locked
arena per staged step, reused only once the copies that read it completed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import torch

from uniserve.executor.arena import aligned
from uniserve.sampling.state import COMMIT_FIELDS
from uniserve.sampling.top_k import PARAMETER_COLUMNS

from .prompt import PROMPT_FIELDS
from .protocol import COMPLETION_FIELDS
from .rows import ROW_SECTIONS
from .sampling import (
    FINISH_IDS,
    bitmask_words,
    logprob_capacity,
    prompt_block,
)
from .shapes import (
    PrefillShape,
    PrefillVariant,
    TextShapes,
    prefill_row_buckets,
    prefill_shapes,
)

__all__ = [
    "DECODE_ROW_BYTES",
    "SELECT_ROW_BYTES",
    "SET_SCATTER_WORDS",
    "DecodeInputs",
    "Dimensions",
    "InputBuffers",
    "PageGroup",
    "pool_rows",
    "Relay",
    "SelectInputs",
    "SlotWrite",
    "Stage",
    "Staging",
    "TableShape",
    "TextBounds",
]

# Bytes one decode row stages: its slot (int64), finish tokens (int64 each),
# draw (float32), parameters (float32 each) and forced finish (uint8).
DECODE_ROW_BYTES = 8 + 8 * FINISH_IDS + 4 + 4 * PARAMETER_COLUMNS + 1

# Bytes one selection task stages: slot and gate, logits row, sampler, set
# row and terminal draft prefix (int64 each), parameters and draw (float32)
# and its bias and forced-finish flags (uint8 each).
SELECT_ROW_BYTES = 8 * 6 + 4 * PARAMETER_COLUMNS + 4 + 2

# Nonzero token-set words per task the scatter column holds; a selection
# with more copies its sets whole.
SET_SCATTER_WORDS = 16

# The least bytes of a page-locked staging arena, so that small stages share
# one arena size instead of replacing arenas as they grow.
_MIN_ARENA_BYTES = 1 << 16


class SlotWrite(IntEnum):
    """The ``[2, R]`` index-pair blocks of the ``slot_writes`` column.

    Each pair is ``(slot, value)``: a decode row's host value, written into
    its slot's state before the forward gathers it.
    """

    TOKENS = 0
    POSITIONS = 1


class Relay(IntEnum):
    """The ``[2, R]`` index-pair blocks of the ``relays`` column."""

    # ``(token offset, slot)``: a prefill row's first token, the slot's
    # committed next token.
    PREFILL_TOKENS = 0
    # ``(call, task)``: sampled tokens and transition decisions copied into
    # the worker's staging by call.
    TOKENS = 1
    TRANSITIONS = 2
    # ``(tail row, slot)``: retained prompt logits copied into the slot.
    PROMPT_TAILS = 3


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True, slots=True)
class PageGroup:
    """One prefix cache group's page shape, as the worker plans it.

    Attributes:
        block_size: Tokens per page.
        units_per_page: Pool units (and block tables) one page spans.
        window: The group's attention window, ``None`` for full attention.
    """

    block_size: int
    units_per_page: int
    window: int | None

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> PageGroup:
        """Decode one entry of the worker's ``cache_groups``."""
        window = value.get("window")
        return cls(
            block_size=int(value["block_size"]),
            units_per_page=int(value["units_per_page"]),
            window=None if window is None else int(window),
        )

    def table_width(
        self,
        max_sequence_tokens: int,
        max_query_tokens: int,
        units: int | None = None,
    ) -> int:
        """Return the most pages a row selects from one of the group's tables.

        A full-attention row selects its whole reservation; a windowed row
        its window and queries plus the partial page they start in. With the
        cache pool's ``units`` (sentinel included), a table selects no more
        pages than the pool holds.
        """
        resident = -(-max(max_sequence_tokens, 1) // self.block_size)
        width = resident
        if self.window is not None:
            reach = self.window + max(max_query_tokens, 1)
            width = min(resident, -(-reach // self.block_size) + 1)
        if units is not None:
            width = min(width, max((units - 1) // self.units_per_page, 1))
        return width


def pool_rows(groups: Sequence[PageGroup], units: int) -> int:
    """Rows of a page of every group that a pool of ``units`` holds.

    ``units`` counts the pool's sentinel unit, which no row takes.
    """
    row_units = max(sum(group.units_per_page for group in groups), 1)
    return max(units - 1, 0) // row_units


@dataclass(frozen=True, slots=True)
class TableShape:
    """One numerical block table: its group's page layout."""

    group: int
    block_size: int
    windowed: bool


@dataclass(frozen=True, slots=True)
class TextBounds:
    """Every bound of a computation's text runners.

    Attributes:
        rows: Row capacity of the columns: a batch's rows, or a wider
            prefill or decode bucket.
        tokens: Token capacity of the columns.
        groups: The prefix cache groups, in group order.
        tables: The numerical block tables, group-major.
        widths: Pages per table every graph addresses.
        decode: The decode row buckets, in configured order.
        prefill: The prefill captures.
    """

    rows: int
    tokens: int
    groups: tuple[PageGroup, ...]
    tables: tuple[TableShape, ...]
    widths: tuple[int, ...]
    decode: tuple[int, ...]
    prefill: tuple[PrefillShape, ...]

    @classmethod
    def derive(
        cls,
        *,
        max_batch_calls: int,
        max_batch_tokens: int,
        max_model_len: int,
        slots: int,
        groups: Sequence[PageGroup],
        units: int | None,
        graphs: bool,
        decode_sizes: Sequence[int],
        prefill_token_sizes: Sequence[int],
        prefill_outputs: bool,
        media_axes: int | None,
    ) -> TextBounds:
        """Derive the bounds.

        Args:
            max_batch_calls: The most calls of a batch.
            max_batch_tokens: The most tokens of a batch.
            max_model_len: The most tokens of a request.
            slots: The request slots.
            groups: The prefix cache's groups.
            units: The cache pool's units, sentinel included, once fitted;
                ``None`` bounds the graph shapes by the configuration
                alone, as the manifest does before planning.
            graphs: CUDA graphs serve the computation.
            decode_sizes: The configured decode row buckets.
            prefill_token_sizes: The configured prefill token buckets.
            prefill_outputs: Prefill graphs keep their outputs.
            media_axes: The rotary axes of prompts placing vision blocks,
                for a model that places them: their prefills are captured
                as a variant of their own.

        Raises:
            ValueError: No cache group.
        """
        if not groups:
            raise ValueError(
                "a text computation reads a prefix cache of some group"
            )
        max_rows = min(max_batch_calls, max(slots, 1))
        tokens = int(max_batch_tokens)
        tables: list[TableShape] = []
        widths: list[int] = []
        for index, group in enumerate(groups):
            width = group.table_width(max_model_len, tokens, units)
            for _ in range(group.units_per_page):
                tables.append(
                    TableShape(
                        index, group.block_size, group.window is not None
                    )
                )
                widths.append(int(width))

        # A decode batch holds a page of every group per row, so the pool
        # bounds the rows it admits; the configured sizes up to that bound
        # are captured, and the bound itself, so that every decode batch
        # the engine forms fits a bucket.
        decode_rows = (
            max_rows
            if units is None
            else min(max_rows, pool_rows(groups, units))
        )
        decode: list[int] = []
        if graphs and decode_rows > 0:
            decode = [
                int(size) for size in decode_sizes if 0 < size <= decode_rows
            ]
            if not decode or decode[-1] != decode_rows:
                decode.append(decode_rows)
        variants = [PrefillVariant.TEXT]
        if media_axes is not None:
            variants.append(PrefillVariant.media(int(media_axes)))
        prefill: list[PrefillShape] = []
        if graphs:
            pages = [
                (group.block_size, group.units_per_page) for group in groups
            ]
            prefill = prefill_shapes(
                prefill_token_sizes,
                prefill_row_buckets(decode_rows),
                decode_rows,
                tokens,
                variants,
                bool(prefill_outputs),
                None if units is None else (pages, units - 1),
            )
        rows = max(
            [max_rows] + [shape.row_bucket for shape in prefill] + list(decode)
        )
        tokens = max([tokens] + list(decode))
        return cls(
            rows=rows,
            tokens=tokens,
            groups=tuple(groups),
            tables=tuple(tables),
            widths=tuple(widths),
            decode=tuple(decode),
            prefill=tuple(prefill),
        )

    @property
    def width(self) -> int:
        """Return the most pages a row selects from one table."""
        return max(max(self.widths, default=1), 1)

    @property
    def pages(self) -> list[tuple[int, int]]:
        """Return each group's ``(tokens per page, units per page)``."""
        return [
            (group.block_size, group.units_per_page) for group in self.groups
        ]

    def shapes(self) -> TextShapes:
        """Return the computation's graph catalogs."""
        return TextShapes(self.decode, self.prefill)

    def dimensions(
        self,
        *,
        vocab: int,
        local_vocab: int,
        hidden: int,
        embeddings: bool,
        dtype: torch.dtype,
        drafts: int = 0,
    ) -> Dimensions:
        """Return the column dimensions of a runner of these bounds."""
        return Dimensions(
            rows=self.rows,
            tokens=self.tokens,
            tables=len(self.tables),
            width=self.width,
            local_vocab=int(local_vocab),
            vocab=int(vocab),
            hidden=int(hidden),
            embeddings=bool(embeddings),
            dtype=dtype,
            drafts=int(drafts),
        )


@dataclass(frozen=True, slots=True)
class Dimensions:
    """The dimensions of one runner's columns.

    Attributes:
        rows: Row capacity ``R``.
        tokens: Token capacity ``T``.
        tables: Numerical block tables ``Tb``.
        width: Pages a row selects from one table, at most.
        local_vocab: Vocabulary columns of this rank's logits.
        vocab: The global vocabulary sampling and prompt scoring read.
        hidden: The model's hidden width.
        embeddings: Vision features replace token embeddings.
        dtype: The model's floating dtype.
        drafts: Draft tokens a call proposes at most; zero for a runner
            that does not draft.
    """

    rows: int
    tokens: int
    tables: int
    width: int
    local_vocab: int
    vocab: int
    hidden: int
    embeddings: bool
    dtype: torch.dtype
    drafts: int = 0

    def row_words(self) -> int:
        """Return the row gather's column words."""
        return (ROW_SECTIONS + self.tables) * (self.rows + 1) + self.tokens

    def logprob_words(self) -> int:
        """Return the logprob words a forward may report."""
        return logprob_capacity(self.tokens)

    def set_words(self) -> int:
        """Return the int32 words of one token set."""
        return bitmask_words(self.vocab)

    def prompt_block(self, top: int) -> int:
        """Return the prompt rows a ``score_prompt`` call scores at ``top``."""
        return prompt_block(top, self.vocab, self.tokens)

    def prompt_rows(self) -> int:
        """Return the prompt rows a forward may score, padded to a block."""
        return self.tokens + self.rows + self.prompt_block(0)

    def scatter_words(self) -> int:
        """Return the nonzero token-set words one selection scatters."""
        return 2 * self.rows * SET_SCATTER_WORDS

    def columns(self) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
        """Return every column: its name, element type and shape."""
        rows, tokens, words = self.rows, self.tokens, self.set_words()
        columns: list[tuple[str, torch.dtype, tuple[int, ...]]] = [
            ("decode_inputs", torch.uint8, (rows * DECODE_ROW_BYTES,)),
            ("select_inputs", torch.uint8, (rows * SELECT_ROW_BYTES,)),
            ("input_ids", torch.int64, (tokens,)),
            ("positions", torch.int64, (3, tokens)),
            ("row_columns", torch.int64, (self.row_words(),)),
            ("causal_values", torch.int32, (rows,)),
            ("logit_rows", torch.int64, (tokens,)),
            ("candidate_rows", torch.int64, (tokens,)),
            ("candidate_draws", torch.float32, (tokens,)),
            ("candidate_params", torch.float32, (tokens, PARAMETER_COLUMNS)),
            ("candidate_mask_rows", torch.int64, (tokens,)),
            ("select_drafts", torch.int64, (tokens,)),
            ("select_masks", torch.int32, (rows, words)),
            ("token_sets", torch.int32, (2, rows, words)),
            ("set_indices", torch.int64, (self.scatter_words(),)),
            ("set_values", torch.int32, (self.scatter_words(),)),
            ("logprob_tasks", torch.int64, (rows,)),
            ("logprob_counts", torch.int64, (rows,)),
            # A verification reports a row per position it may commit.
            ("verified_logprobs", torch.int64, (3, tokens)),
            ("prompt_rows", torch.int64, (PROMPT_FIELDS, self.prompt_rows())),
            ("prompt_block", torch.int32, (1,)),
            ("commit", torch.int64, (COMMIT_FIELDS, rows)),
            ("slot_writes", torch.int64, (2 * len(SlotWrite), rows)),
            ("relays", torch.int64, (2 * len(Relay), rows)),
            ("logits", self.dtype, (rows, self.local_vocab)),
            ("completion", torch.int64, (COMPLETION_FIELDS, rows)),
            ("continuation", torch.uint8, (rows,)),
            ("transition", torch.uint8, (rows,)),
            ("tagged", torch.int64, (rows,)),
            ("points", torch.int64, (rows,)),
            ("scored", torch.int64, (rows,)),
            # The reportable words, then one that takes the scores no one
            # reports.
            ("logprobs", torch.int64, (self.logprob_words() + 1,)),
            ("prompt_tail", self.dtype, (rows + 1, self.vocab)),
            ("hidden", self.dtype, (tokens, self.hidden)),
        ]
        if self.embeddings:
            columns.append(
                ("input_embeddings", self.dtype, (tokens, self.hidden))
            )
            columns.append(("embedding_mask", torch.uint8, (tokens,)))
        if self.drafts:
            windows = self.window_tokens()
            columns += [
                ("draft_row_columns", torch.int64, (self.row_words(),)),
                ("draft_window_columns", torch.int64, (self.row_words(),)),
                ("draft_input_ids", torch.int64, (tokens,)),
                ("draft_positions", torch.int64, (tokens,)),
                ("draft_hidden_rows", torch.int64, (tokens,)),
                ("draft_pairs", torch.int64, (2, rows)),
                ("draft_proposing", torch.int64, (2, rows)),
                ("draft_window_tokens", torch.int64, (rows,)),
                ("draft_window_owners", torch.int64, (windows,)),
                ("draft_window_positions", torch.int64, (windows,)),
                ("proposals", torch.int64, (rows, self.drafts)),
            ]
        return columns

    def window_tokens(self) -> int:
        """Return the tokens of the drafter's chained-step windows at most:
        a proposer's window spans its drafts and its proposals.
        """  # noqa: D205
        return 2 * self.drafts * self.rows

    def bytes(self) -> int:
        """Return the device bytes of the columns."""
        return sum(
            aligned(max(int(np.prod(shape)), 1) * _element_size(dtype))
            for _, dtype, shape in self.columns()
        )


@dataclass(frozen=True, slots=True)
class DecodeInputs:
    """The views of a decode bucket of ``rows`` rows in its input region.

    Attributes:
        slots: ``int64 [rows]`` each row's request slot, 0 for padding.
        finish_ids: ``int64 [rows, FINISH_IDS]`` finish tokens, -1 padded.
        draws: ``float32 [rows]`` each row's draw.
        params: ``float32 [rows, PARAMETER_COLUMNS]`` parameter rows.
        force: ``uint8 [rows]`` rows forced to finish.
        region: The bucket's bytes of the region.
    """

    slots: torch.Tensor
    finish_ids: torch.Tensor
    draws: torch.Tensor
    params: torch.Tensor
    force: torch.Tensor
    region: torch.Tensor


def decode_layout(
    rows: int,
) -> dict[str, tuple[int, np.dtype, tuple[int, ...]]]:
    """Return the decode region's fields of a bucket: offset, dtype, shape."""
    return {
        "slots": (0, np.dtype(np.int64), (rows,)),
        "finish_ids": (8 * rows, np.dtype(np.int64), (rows, FINISH_IDS)),
        "draws": (8 * (1 + FINISH_IDS) * rows, np.dtype(np.float32), (rows,)),
        "params": (
            (8 * (1 + FINISH_IDS) + 4) * rows,
            np.dtype(np.float32),
            (rows, PARAMETER_COLUMNS),
        ),
        "force": (
            (8 * (1 + FINISH_IDS) + 4 + 4 * PARAMETER_COLUMNS) * rows,
            np.dtype(np.uint8),
            (rows,),
        ),
    }


@dataclass(frozen=True, slots=True)
class SelectInputs:
    """The views of a selection's task bucket of ``tasks`` tasks.

    Attributes:
        tasks: ``int64 [2, tasks]`` each task's slot and gate.
        rows: ``int64 [tasks]`` each task's logits row.
        paths: ``int64 [tasks]`` each task's sampler.
        mask_rows: ``int64 [tasks]`` each task's token set, or -1.
        terminal: ``int64 [tasks]`` each task's terminal draft prefix.
        params: ``float32 [tasks, PARAMETER_COLUMNS]`` parameter rows.
        draws: ``float32 [tasks]`` draws.
        biased: ``uint8 [tasks]`` tasks adding their slot's logit biases.
        force: ``uint8 [tasks]`` tasks forced to finish.
        region: The bucket's bytes of the region.
    """

    tasks: torch.Tensor
    rows: torch.Tensor
    paths: torch.Tensor
    mask_rows: torch.Tensor
    terminal: torch.Tensor
    params: torch.Tensor
    draws: torch.Tensor
    biased: torch.Tensor
    force: torch.Tensor
    region: torch.Tensor


def select_layout(
    tasks: int,
) -> dict[str, tuple[int, np.dtype, tuple[int, ...]]]:
    """Return the select region's fields of a bucket: offset, dtype, shape."""
    i64, f32, u8 = np.dtype(np.int64), np.dtype(np.float32), np.dtype(np.uint8)
    return {
        "tasks": (0, i64, (2, tasks)),
        "rows": (16 * tasks, i64, (tasks,)),
        "paths": (24 * tasks, i64, (tasks,)),
        "mask_rows": (32 * tasks, i64, (tasks,)),
        "terminal": (40 * tasks, i64, (tasks,)),
        "params": (48 * tasks, f32, (tasks, PARAMETER_COLUMNS)),
        "draws": ((48 + 4 * PARAMETER_COLUMNS) * tasks, f32, (tasks,)),
        "biased": ((52 + 4 * PARAMETER_COLUMNS) * tasks, u8, (tasks,)),
        "force": ((53 + 4 * PARAMETER_COLUMNS) * tasks, u8, (tasks,)),
    }


_TORCH = {
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.uint8): torch.uint8,
}


def _region_views(
    region: torch.Tensor,
    layout: dict[str, tuple[int, np.dtype, tuple[int, ...]]],
) -> dict[str, torch.Tensor]:
    views = {}
    for name, (offset, dtype, shape) in layout.items():
        count = int(np.prod(shape))
        views[name] = (
            region[offset : offset + count * dtype.itemsize]
            .view(_TORCH[dtype])
            .view(shape)
        )
    return views


def host_region(
    layout: dict[str, tuple[int, np.dtype, tuple[int, ...]]], size: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a host byte region of ``size`` bytes and its field views."""
    region = np.zeros(size, dtype=np.uint8)
    views = {}
    for name, (offset, dtype, shape) in layout.items():
        count = int(np.prod(shape))
        views[name] = (
            region[offset : offset + count * dtype.itemsize]
            .view(dtype)
            .reshape(shape)
        )
    return region, views


class InputBuffers:
    """A runner's columns in one zeroed device arena.

    Captured graphs address the columns, so the arena lives as long as the
    runner. The region views of each bucket are created once.

    Args:
        dimensions: The columns' dimensions.
        device: The device the runner computes on.
    """

    def __init__(self, dimensions: Dimensions, device: torch.device) -> None:
        self.dimensions = dimensions
        self.arena = torch.zeros(
            dimensions.bytes(), dtype=torch.uint8, device=device
        )
        self._columns: dict[str, torch.Tensor] = {}
        offset = 0
        for name, dtype, shape in dimensions.columns():
            count = max(int(np.prod(shape)), 1)
            size = count * _element_size(dtype)
            view = self.arena[offset : offset + size].view(dtype)
            self._columns[name] = view[: int(np.prod(shape))].view(shape)
            offset += aligned(size)
        self._decode: dict[int, DecodeInputs] = {}
        self._select: dict[int, SelectInputs] = {}

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._columns[name]

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def decode(self, rows: int) -> DecodeInputs:
        """Return the decode input views of a bucket of ``rows`` rows."""
        found = self._decode.get(rows)
        if found is None:
            region = self["decode_inputs"][: rows * DECODE_ROW_BYTES]
            found = DecodeInputs(
                **_region_views(region, decode_layout(rows)), region=region
            )
            self._decode[rows] = found
        return found

    def select(self, tasks: int) -> SelectInputs:
        """Return the select input views of a bucket of ``tasks`` tasks."""
        found = self._select.get(tasks)
        if found is None:
            region = self["select_inputs"][: tasks * SELECT_ROW_BYTES]
            found = SelectInputs(
                **_region_views(region, select_layout(tasks)), region=region
            )
            self._select[tasks] = found
        return found


@dataclass
class Stage:
    """The host values of one step, gathered before their copies.

    :meth:`put` names a device destination and its host value; the values
    are copied into one page-locked arena and their copies enqueued by
    :meth:`Staging.submit`.
    """

    entries: list[tuple[torch.Tensor, np.ndarray]] = field(default_factory=list)

    def put(self, target: torch.Tensor, value: np.ndarray) -> None:
        """Copy ``value`` into ``target`` (contiguous, same element count)."""
        value = np.ascontiguousarray(value)
        if value.nbytes != target.numel() * target.element_size():
            raise AssertionError("a staged value disagrees with its column")
        if value.nbytes:
            self.entries.append((target, value))


class Staging:
    """Page-locked arenas carrying a computation's host values to its columns.

    An arena serves one submitted :class:`Stage`: its values are copied in,
    one host-to-device copy per value is enqueued on the stream, and an
    event recorded behind them. The arena is reused only once its event
    completed, because the copies read it when the device reaches them,
    which may be well after the host returned. Arenas grow to the largest
    stage and their number to the stages in flight. Columns in host memory
    take their values at once, without an arena.

    Args:
        device: The device the columns live on.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._free: list[torch.Tensor] = []
        self._busy: list[tuple[torch.Tensor, torch.cuda.Event]] = []

    def _arena(self, size: int) -> torch.Tensor:
        # Arenas whose copies completed return to the free list.
        busy = []
        for arena, event in self._busy:
            if event.query():
                self._free.append(arena)
            else:
                busy.append((arena, event))
        self._busy = busy
        for index, arena in enumerate(self._free):
            if arena.numel() >= size:
                return self._free.pop(index)
        if self._free:
            # Replace the smallest free arena by one that holds the stage.
            self._free.sort(key=lambda arena: arena.numel())
            self._free.pop(0)
        return torch.empty(
            max(size, _MIN_ARENA_BYTES), dtype=torch.uint8, pin_memory=True
        )

    def submit(self, stage: Stage, stream: torch.cuda.Stream | None) -> None:
        """Enqueue ``stage``'s copies on ``stream`` behind its prior work."""
        if not stage.entries:
            return
        if self.device.type != "cuda":
            for target, value in stage.entries:
                source = torch.from_numpy(value.view(np.uint8).reshape(-1))
                target.copy_(source.view(target.dtype).view(target.shape))
            stage.entries.clear()
            return
        offsets = []
        size = 0
        for _, value in stage.entries:
            offsets.append(size)
            size = aligned(size + value.nbytes)
        arena = self._arena(size)
        host = arena.numpy()
        with torch.cuda.stream(stream):
            for (target, value), offset in zip(
                stage.entries, offsets, strict=True
            ):
                host[offset : offset + value.nbytes] = value.view(
                    np.uint8
                ).reshape(-1)
                source = arena[offset : offset + value.nbytes].view(
                    target.dtype
                )
                target.copy_(source.view(target.shape), non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        self._busy.append((arena, event))
        stage.entries.clear()

    def close(self) -> None:
        """Drop the arenas after the streams drained."""
        self._free.clear()
        self._busy.clear()
