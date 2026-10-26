"""Prefill calls that place image features in a request's sequence.

A *context* prefill (:attr:`~.protocol.CallFlag.WRITES_CONTEXT`) holds a
prompt with vision blocks: its rows alternate runs of prompt text (causal)
and one block per vision input (non-causal), at increasing prefix cache and
position coordinates; a block may advance the rotary position by less than
its token count. A *visual-state* prefill
(:attr:`~.protocol.CallFlag.WRITES_VISUAL_STATE`) places exactly one feature
after a generated or ingested image: a vision feature closes the image as
one block row; a latent feature (an image denoiser's conditioning) writes
its framed latent sequence's K/V through the image denoiser instead of a
text row.

Every row of these calls carries explicit positions on the three rotary
axes (:attr:`~.rows.RowSource.PLACED`; text and markers take spatial
zeros), and their features replace the block's placeholder embeddings.
Under CUDA graphs they replay the media variant's prefill buckets, whose
graphs read each row's causality and the embedding columns on the device.
The logical position after the call is the temporal position after its last
row; a visual-state row advances it by the image denoiser's rope advance
when it completes a feedback.

The image half of the executor answers what the rows need through two
protocols the text lanes hold: :class:`ImageFacts` (the model's feature
injection, patch grids and rotary layout) and :class:`ImageDenoiser` (the
latent sequence of a conditioning image and the write of its K/V).

The module also assembles a forward's rows from calls of either kind
(:func:`host_rows`, :func:`placed_rows`, :func:`take_rows`,
:func:`concat_rows`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from uniserve.processing import PositionLayout

from .rows import Rows, RowSource, TokenSelection

__all__ = [
    "ConditionRow",
    "ImageDenoiser",
    "ImageFacts",
    "Injection",
    "MediaCall",
    "PlacedRow",
    "VisionBlock",
    "VisionInput",
    "block_row",
    "concat_rows",
    "context",
    "host_rows",
    "next_position",
    "placed_rows",
    "take_rows",
    "text_row",
    "visual",
]


@dataclass(frozen=True, slots=True)
class VisionBlock:
    """One vision block's slots in a language sequence.

    Attributes:
        token_ids: ``int64 [n]`` marker tokens and placeholder 1 at feature
            slots.
        features: The feature slots, whose embeddings the features replace.
        positions: ``int64 [n]`` one position per slot (temporal or
            sequential layouts), or ``int64 [3, n]`` temporal, row and
            column positions.
    """

    token_ids: np.ndarray
    features: range
    positions: np.ndarray

    def __len__(self) -> int:
        return len(self.token_ids)


class Injection(Protocol):
    """A model's feature injection policy with its markers resolved."""

    @property
    def positions(self) -> PositionLayout:
        """The position coordinates of inserted features."""

    def block(
        self,
        feature_tokens: int,
        position: int,
        close_image: bool,
        grid: tuple[int, int] | None,
    ) -> VisionBlock:
        """Lay out a block of ``feature_tokens`` features at ``position``.

        ``grid`` is the image's raw patch grid ``(rows, columns)``, which
        temporal-spatial positions require; a block that closes a generated
        image ends with the end marker.

        Raises:
            ValueError: The block has no feature, a marker is unresolved, or
                the features do not pool the grid.
        """


class ImageFacts(Protocol):
    """What a text lane asks the image half about vision blocks."""

    def spatial_positions(self) -> bool:
        """Whether the model's rows carry three rotary axes."""

    @property
    def injection(self) -> Injection | None:
        """The model's feature injection, ``None`` without one."""

    def grid(
        self, height: int, width: int, close_image: bool, input_images: int
    ) -> tuple[int, int]:
        """The raw patch grid ``(rows, columns)`` of an image's features.

        A generated image (``close_image``) keeps the single-image pixel
        bound; a request's input image counts its ``input_images``.
        """

    @property
    def rope_advance(self) -> int:
        """Positions a completed generated image spans; 1 without a denoiser."""


@dataclass(frozen=True, slots=True)
class ConditionRow:
    """One latent image whose K/V the image denoiser writes as conditioning.

    Attributes:
        features: The latent feature tensor the worker leased.
        height, width: The image's pixel size.
        slot: The request slot whose cache receives the K/V.
        prefix: The visible cache tokens before the latent sequence.
        position: The language position the sequence is conditioned at.
    """

    features: torch.Tensor
    height: int
    width: int
    slot: int
    prefix: int
    position: int


class ImageDenoiser(Protocol):
    """What a text lane asks the image denoiser about latent conditioning."""

    def sequence_length(self, height: int, width: int) -> int:
        """Tokens of an image's framed latent sequence."""

    def conditioning_positions(
        self, height: int, width: int, start: int
    ) -> np.ndarray:
        """The sequence's ``int64 [3, tokens]`` positions from ``start``."""

    def condition_latents(
        self,
        rows: Sequence[ConditionRow],
        pages: np.ndarray,
        stream: torch.cuda.Stream | None,
    ) -> None:
        """Write the K/V of ``rows`` into their slots' caches.

        ``pages`` is ``int64 [2 * G, len(rows)]``: per cache group ``g``,
        each row's first selected page (row ``2g``) and page count (row
        ``2g + 1``). The work orders after what ``stream`` (the text lane's,
        ``None`` on the host) enqueued before, and ``stream`` orders after
        it.
        """


@dataclass(frozen=True, slots=True)
class VisionInput:
    """One vision block of a context call (the step JSON's ``vision``).

    Attributes:
        offset: The block's place among the call's input tokens.
        features: ``[tokens, hidden]`` encoder features.
        height, width: The image's pixel size.
    """

    offset: int
    features: torch.Tensor
    height: int
    width: int

    @classmethod
    def from_json(
        cls, value: Mapping[str, Any], features: Sequence[torch.Tensor]
    ) -> VisionInput:
        """Decode a ``vision`` entry naming its tensor among ``features``."""
        return cls(
            offset=int(value["offset"]),
            features=features[int(value["tensor"])],
            height=int(value["height"]),
            width=int(value["width"]),
        )


@dataclass(frozen=True, slots=True)
class PlacedRow:
    """One row at explicit rotary positions.

    Attributes:
        slot: The request slot.
        tokens: ``int64 [n]`` the row's tokens.
        positions: ``int64 [3, n]`` temporal, row and column positions.
        prefix: Visible cache tokens before the row.
        selection: The logits the row projects
            (:class:`~.rows.TokenSelection`).
        causal: The row attends causally; a vision block attends to all of
            itself.
        spatial: The row carries three meaningful rotary axes.
    """

    slot: int
    tokens: np.ndarray
    positions: np.ndarray
    prefix: int
    selection: int
    causal: bool
    spatial: bool

    @property
    def query(self) -> int:
        """Return the row's query tokens."""
        return len(self.tokens)


@dataclass(slots=True)
class MediaCall:
    """The rows of a call placing image features and how it advances.

    Attributes:
        rows: Every row of the call, in sequence order; its last row is the
            one it samples from. Empty for a latent conditioning, whose
            K/V the image denoiser writes.
        embeddings: ``(token, features)`` of each vision block: its first
            feature token, counted from the call's first token.
        position: The logical position after the call.
        continued: The position the slot continues from when it samples.
        sample: The semantic position of the call's sample.
        latent: The latent conditioning the image denoiser writes.
        last: The row the call's progress is computed from: its last row,
            or a latent conditioning's sequence.
    """

    rows: list[PlacedRow]
    embeddings: list[tuple[int, torch.Tensor]]
    position: int
    continued: int
    sample: int
    latent: ConditionRow | None
    last: PlacedRow = field(repr=False)


def next_position(row: PlacedRow) -> int:
    """Return the temporal position after ``row``."""
    if row.query == 0:
        raise AssertionError("a context row has no positions")
    return int(np.max(row.positions[0])) + 1


def text_row(
    slot: int,
    tokens: Sequence[int] | np.ndarray,
    prefix: int,
    position: int,
    selection: int,
    spatial: bool,
) -> PlacedRow:
    """Return the causal row of prompt ``tokens`` after ``prefix`` tokens.

    Its tokens take consecutive temporal positions from ``position`` and
    zero spatial ones.
    """
    values = np.asarray(tokens, dtype=np.int64)
    count = len(values)
    positions = np.zeros((3, count), dtype=np.int64)
    positions[0] = position + np.arange(count, dtype=np.int64)
    return PlacedRow(
        slot=int(slot),
        tokens=values,
        positions=positions,
        prefix=int(prefix),
        selection=int(selection),
        causal=True,
        spatial=bool(spatial),
    )


def block_row(
    block: VisionBlock,
    slot: int,
    prefix: int,
    spatial: bool,
    selection: int,
) -> PlacedRow:
    """Return the row of ``block`` after ``prefix`` visible tokens of ``slot``.

    Its markers and placeholders take the block's positions and attend to
    all of the block. ``spatial`` gives it three rotary axes.
    """
    count = len(block)
    values = np.asarray(block.positions, dtype=np.int64)
    if values.ndim == 1:
        positions = np.zeros((3, count), dtype=np.int64)
        positions[0] = values
    else:
        positions = values.reshape(3, count)
    tokens = np.asarray(block.token_ids, dtype=np.int64)
    if np.any((tokens < 0) | (tokens > np.iinfo(np.uint32).max)):
        raise ValueError("vision marker overflows")
    return PlacedRow(
        slot=int(slot),
        tokens=tokens,
        positions=positions,
        prefix=int(prefix),
        selection=int(selection),
        causal=False,
        spatial=bool(spatial),
    )


def _vision_row(
    facts: ImageFacts | None,
    vision: VisionInput,
    slot: int,
    prefix: int,
    position: int,
    close_image: bool,
    selection: int,
    allocated: int,
    input_images: int,
) -> tuple[PlacedRow, int]:
    """Return the block row of ``vision`` at ``position``.

    Returns:
        The row and its first feature slot.
    """
    if facts is None:
        raise NotImplementedError("vision input requires the image runtime")
    injection = facts.injection
    if injection is None:
        raise ValueError(
            "vision state preparation requires declared feature injection"
        )
    if prefix > allocated:
        raise ValueError(
            "vision row prefix lies outside the request's cache extent"
        )
    if not close_image and input_images == 0:
        raise ValueError("request admission declares no input images")
    if vision.features.ndim != 2:
        raise ValueError("vision features must have shape [tokens, hidden]")
    tokens = int(vision.features.shape[0])
    # Spatial positions place the features on the image's patch grid.
    grid = None
    if injection.positions == PositionLayout.TEMPORAL_SPATIAL:
        grid = facts.grid(
            vision.height, vision.width, close_image, input_images
        )
    block = injection.block(tokens, position, close_image, grid)
    row = block_row(block, slot, prefix, facts.spatial_positions(), selection)
    return row, block.features.start


def context(
    facts: ImageFacts | None,
    *,
    slot: int,
    tokens: np.ndarray,
    vision: Sequence[VisionInput],
    logical: int,
    visible: int,
    allocated: int,
    input_images: int,
    scores: bool,
    token_output: bool,
) -> MediaCall:
    """Return the rows of a context prefill: its prompt runs and blocks.

    A call that ``scores`` its prompt projects every text token and each
    block's last token; otherwise only the call's last row projects, when
    it has a token output.

    Args:
        facts: The image half's facts.
        slot: The request slot.
        tokens: ``int64 [n]`` the call's input tokens.
        vision: The call's vision blocks, ordered by offset.
        logical: The request's logical position before the call.
        visible: The visible cache tokens before the call.
        allocated: The slot's allocated cache length.
        input_images: The request's admitted input images.
        scores: The request scores its prompt.
        token_output: The call samples a token.

    Raises:
        ValueError: The blocks do not lie in order within the tokens, or a
            block cannot be laid out.
    """
    spatial = facts is not None and facts.spatial_positions()
    position = int(logical)
    prefix = int(visible)
    rows: list[PlacedRow] = []
    embeddings: list[tuple[int, torch.Tensor]] = []
    offset = 0
    begin = 0
    blocks = len(vision)
    for index in range(blocks + 1):
        end = int(vision[index].offset) if index < blocks else len(tokens)
        if end < begin or end > len(tokens):
            raise ValueError(
                "context blocks must lie in order within the call's tokens"
            )
        if end > begin:
            # A scored prompt projects every text token; its last row also
            # samples.
            if scores:
                selection = TokenSelection.ALL_LOGITS
            elif index == blocks and token_output:
                selection = TokenSelection.LAST_LOGITS
            else:
                selection = TokenSelection.CACHE
            row = text_row(
                slot, tokens[begin:end], prefix, position, selection, spatial
            )
            position += end - begin
            prefix += end - begin
            offset += row.query
            rows.append(row)
        if index < blocks:
            last = index + 1 == blocks and end == len(tokens)
            # A block's last logits predict the prompt's next token.
            if scores or (last and token_output):
                selection = TokenSelection.LAST_LOGITS
            else:
                selection = TokenSelection.CACHE
            row, first = _vision_row(
                facts,
                vision[index],
                slot,
                prefix,
                position,
                False,
                selection,
                allocated,
                input_images,
            )
            position = next_position(row)
            prefix += row.query
            embeddings.append((offset + first, vision[index].features))
            offset += row.query
            rows.append(row)
        begin = end
    if not rows:
        raise ValueError("context prefill has no rows")
    following = next_position(rows[-1])
    return MediaCall(
        rows=rows,
        embeddings=embeddings,
        position=following,
        continued=following,
        sample=following,
        latent=None,
        last=rows[-1],
    )


def visual(
    facts: ImageFacts | None,
    denoiser: ImageDenoiser | None,
    *,
    slot: int,
    vision: VisionInput | None,
    latent: VisionInput | None,
    logical: int,
    visible: int,
    allocated: int,
    input_images: int,
    max_tokens: int,
    token_output: bool,
    completion_output: bool,
) -> MediaCall:
    """Return the row of a visual-state prefill.

    The row is one vision block closing an image, or a latent feature's
    conditioning, which the image denoiser writes. A completed feedback
    (``completion_output``) moves the language position past the image by
    the rope advance; otherwise the slot continues one position later.

    Raises:
        ValueError: The call has not exactly one feature, a latent call
            samples, or the row exceeds the call's token bound.
        NotImplementedError: The rank has no image runtime or denoiser.
    """
    if (vision is None) == (latent is None):
        raise ValueError("visual extend requires exactly one feature tensor")
    start = int(logical)
    advance = facts.rope_advance if facts is not None else 1
    position = start + (advance if completion_output else 0)
    continued = start + (advance if completion_output else 1)
    selection = (
        TokenSelection.LAST_LOGITS if token_output else TokenSelection.CACHE
    )

    if latent is not None:
        if token_output:
            raise ValueError("a latent state prefill samples no token")
        if facts is None:
            raise NotImplementedError("visual input requires the image runtime")
        if denoiser is None:
            raise NotImplementedError(
                "latent state prefill requires an image denoiser"
            )
        query = int(denoiser.sequence_length(latent.height, latent.width))
        if query > max_tokens:
            raise ValueError(
                "image state query span exceeds the call token bound"
            )
        positions = np.asarray(
            denoiser.conditioning_positions(latent.height, latent.width, start),
            dtype=np.int64,
        ).reshape(3, query)
        row = PlacedRow(
            slot=int(slot),
            tokens=np.zeros(query, dtype=np.int64),
            positions=positions,
            prefix=int(visible),
            selection=selection,
            causal=False,
            spatial=True,
        )
        return MediaCall(
            rows=[],
            embeddings=[],
            position=position,
            continued=continued,
            sample=start + advance,
            latent=ConditionRow(
                features=latent.features,
                height=latent.height,
                width=latent.width,
                slot=int(slot),
                prefix=int(visible),
                position=start,
            ),
            last=row,
        )

    assert vision is not None
    row, first = _vision_row(
        facts,
        vision,
        slot,
        visible,
        start,
        True,
        selection,
        allocated,
        input_images,
    )
    if row.query > max_tokens:
        raise ValueError("image state query span exceeds the call token bound")
    return MediaCall(
        rows=[row],
        embeddings=[(first, vision.features)],
        position=position,
        continued=continued,
        sample=start + advance,
        latent=None,
        last=row,
    )


def placed_rows(rows: Sequence[PlacedRow]) -> Rows:
    """Return ``rows`` as a forward's columns, every one placed."""
    count = len(rows)
    queries = np.array([row.query for row in rows], dtype=np.int64)
    starts = np.cumsum(queries) - queries
    return Rows(
        slot=np.array([row.slot for row in rows], dtype=np.int64),
        source=np.full(count, RowSource.PLACED, dtype=np.int64),
        prefix=np.array([row.prefix for row in rows], dtype=np.int64),
        query=queries,
        selection=np.array([row.selection for row in rows], dtype=np.int64),
        gated=np.zeros(count, dtype=bool),
        force=np.zeros(count, dtype=bool),
        causal=np.array([row.causal for row in rows], dtype=bool),
        position=np.array(
            [int(row.positions[0, 0]) if row.query else 0 for row in rows],
            dtype=np.int64,
        ),
        token_start=starts.astype(np.int64),
        token_count=queries.copy(),
        tokens=(
            np.concatenate([row.tokens for row in rows])
            if rows
            else np.zeros(0, dtype=np.int64)
        ),
        spatial=np.array([row.spatial for row in rows], dtype=bool),
        placed={index: row.positions for index, row in enumerate(rows)},
    )


def take_rows(rows: Rows, index: Sequence[int] | np.ndarray) -> Rows:
    """Return the rows ``index`` names, in that order, over the same tokens."""
    index = np.asarray(index, dtype=np.int64)
    placed = {
        new: rows.placed[int(old)]
        for new, old in enumerate(index)
        if int(old) in rows.placed
    }
    return Rows(
        slot=rows.slot[index],
        source=rows.source[index],
        prefix=rows.prefix[index],
        query=rows.query[index],
        selection=rows.selection[index],
        gated=rows.gated[index],
        force=rows.force[index],
        causal=rows.causal[index],
        position=rows.position[index],
        token_start=rows.token_start[index],
        token_count=rows.token_count[index],
        tokens=rows.tokens,
        spatial=rows.spatial[index],
        placed=placed,
    )


def concat_rows(parts: Sequence[Rows]) -> Rows:
    """Return the rows of ``parts`` in order, over their tokens concatenated."""
    if len(parts) == 1:
        return parts[0]
    token_offset = 0
    row_offset = 0
    starts = []
    placed: dict[int, np.ndarray] = {}
    for part in parts:
        starts.append(part.token_start + token_offset)
        for row, positions in part.placed.items():
            placed[row + row_offset] = positions
        token_offset += len(part.tokens)
        row_offset += len(part)

    def column(name: str) -> np.ndarray:
        return np.concatenate([getattr(part, name) for part in parts])

    return Rows(
        slot=column("slot"),
        source=column("source"),
        prefix=column("prefix"),
        query=column("query"),
        selection=column("selection"),
        gated=column("gated"),
        force=column("force"),
        causal=column("causal"),
        position=column("position"),
        token_start=np.concatenate(starts).astype(np.int64),
        token_count=column("token_count"),
        tokens=np.concatenate([part.tokens for part in parts]).astype(np.int64),
        spatial=column("spatial"),
        placed=placed,
    )


def host_rows(
    slots: Sequence[int],
    tokens: Sequence[np.ndarray],
    *,
    prefix: int | Sequence[int],
    selection: int,
    position: int | Sequence[int] | None = None,
) -> Rows:
    """Return causal rows of host ``tokens`` for ``slots``.

    Each row reads ``prefix`` visible tokens and its tokens follow each
    other from ``position`` (its prefix by default).
    """
    count = len(slots)
    queries = np.array([len(values) for values in tokens], dtype=np.int64)
    prefixes = np.broadcast_to(
        np.asarray(prefix, dtype=np.int64), (count,)
    ).copy()
    positions = (
        prefixes.copy()
        if position is None
        else np.broadcast_to(
            np.asarray(position, dtype=np.int64), (count,)
        ).copy()
    )
    return Rows(
        slot=np.asarray(slots, dtype=np.int64),
        source=np.full(count, RowSource.HOST, dtype=np.int64),
        prefix=prefixes,
        query=queries,
        selection=np.full(count, selection, dtype=np.int64),
        gated=np.zeros(count, dtype=bool),
        force=np.zeros(count, dtype=bool),
        causal=np.ones(count, dtype=bool),
        position=positions,
        token_start=(np.cumsum(queries) - queries).astype(np.int64),
        token_count=queries.copy(),
        tokens=(
            np.concatenate(tokens).astype(np.int64)
            if count
            else np.zeros(0, dtype=np.int64)
        ),
        spatial=np.zeros(count, dtype=bool),
    )
