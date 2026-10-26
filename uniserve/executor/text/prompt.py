"""Prompt logprob scoring of a prefill forward.

A prefill call of a request reporting prompt logprobs scores its prompt in
*chunks*: each chunk is a span of the forward's candidate logits rows whose
tokens it scores (:class:`PromptChunk`). A text prompt call is one chunk; a
prompt placing vision blocks is one chunk per causal text run, and a prompt
ending with a vision block retains that block's last logits for the next
call (:func:`media_chunks`).

A chunk's scored rows predict its tokens in order. Its first token is
predicted by what ``continued`` names: nothing for a request's first chunk
(0, which then scores from its second token), the slot's retained logits of
the previous call (1), or candidate row ``continued - 2`` (the last row of
the vision block before it). Each later token is predicted by the
candidate row before it. A chunk's last candidate row's logits are retained
in its tail row, which the commit copies into the slot's ``prompt_logits``.

:meth:`PromptChunks.plan` lays the chunks' reported words out after the
sampled tasks' compact words, every chunk scored at the forward's top
bucket; :func:`prompt_rows` expands them into the per-row columns
``score_prompt`` reads, and :func:`prompt_columns` pads those to whole
``score_prompt`` calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uniserve.errors import OutOfMemory

from .sampling import LogprobLayout, logprob_words, top_bucket

__all__ = [
    "PROMPT_FIELDS",
    "PromptChunk",
    "PromptChunks",
    "media_chunks",
    "prompt_chunk",
    "prompt_columns",
    "prompt_padding",
    "prompt_rows",
]

# Fields of one prompt row: hidden position, retained slot (or -1), target
# token index, chunk word offset, index in the chunk, the chunk's scored
# rows, their top count (-1 reports nothing), tail row.
PROMPT_FIELDS = 8


@dataclass(frozen=True, slots=True)
class PromptChunk:
    """One prompt chunk a prefill forward scores.

    Attributes:
        call: The chunk's call index.
        first: The chunk's first candidate logits row.
        count: The chunk's candidate rows; a chunk of none only retains the
            logits of row ``first``.
        token: The chunk's first token, counted in the forward.
        continued: What predicts the chunk's first token: nothing (0), the
            slot's retained logits (1), or candidate row ``continued - 2``.
        slot: The request slot.
        top: The top count the chunk reports per scored token.
    """

    call: int
    first: int
    count: int
    token: int
    continued: int
    slot: int
    top: int

    @property
    def scored(self) -> int:
        """Return the rows the chunk scores: a first chunk skips one."""
        return max(self.count - int(self.continued == 0), 0)


def _continues(logical: int, prompt_ready: bool) -> bool:
    # A chunk after the first continues from the slot's retained logits,
    # which an earlier call left only when it scored.
    continued = logical > 0
    if continued and not prompt_ready:
        raise ValueError("continued prompt scoring has no preceding logits")
    return continued


def prompt_chunk(
    call: int,
    slot: int,
    first: int,
    count: int,
    token: int,
    logical: int,
    prompt_ready: bool,
    top: int,
) -> PromptChunk:
    """Return the chunk of a text prefill call that samples.

    Only a sampled call records its chunk: a prefill without a token output
    scores nothing.

    Args:
        call: The call index.
        slot: The request slot.
        first: The call row's first candidate row.
        count: The call row's candidate rows.
        token: The call's first token in the forward.
        logical: The call's logical position before the call.
        prompt_ready: The slot retains the previous call's last logits.
        top: The request's reported top count per prompt token.

    Raises:
        ValueError: The call continues a prompt whose logits were not
            retained.
    """
    continued = _continues(logical, prompt_ready)
    return PromptChunk(call, first, count, token, int(continued), slot, top)


def media_chunks(
    call: int,
    slot: int,
    first: Sequence[int] | np.ndarray,
    count: Sequence[int] | np.ndarray,
    causal: Sequence[bool] | np.ndarray,
    query: Sequence[int] | np.ndarray,
    token: int,
    logical: int,
    prompt_ready: bool,
    top: int,
) -> list[PromptChunk]:
    """Return the chunks of a prompt call placing vision blocks.

    Each causal text row is a chunk; a vision block (a row attending to all
    of itself) scores nothing, and the text after it continues from the
    block's last candidate row. A call ending with a block retains that
    row's logits through a chunk of no candidate rows.

    Args:
        call: The call index.
        slot: The request slot.
        first: Each of the call's rows' first candidate row.
        count: Each row's candidate rows.
        causal: Whether each row attends causally.
        query: Each row's query tokens.
        token: The call's first token in the forward.
        logical: The call's logical position before the call.
        prompt_ready: The slot retains the previous call's last logits.
        top: The request's reported top count per prompt token.

    Raises:
        ValueError: The call continues a prompt whose logits were not
            retained.
    """
    previous = int(_continues(logical, prompt_ready))
    chunks = []
    rows = len(first)
    for row in range(rows):
        start, candidates = int(first[row]), int(count[row])
        last = start + candidates - 1
        if causal[row]:
            chunks.append(
                PromptChunk(call, start, candidates, token, previous, slot, top)
            )
        elif row + 1 == rows:
            chunks.append(PromptChunk(call, last, 0, 0, 0, slot, top))
        previous = 2 + last
        token += int(query[row])
    return chunks


@dataclass(slots=True)
class PromptChunks:
    """The prompt chunks a prefill forward scores, laid out in its words.

    Chunk ``i``'s columns describe :class:`PromptChunk` ``i``; its words
    report its own top count compactly, while ``score_prompt`` scores every
    chunk at the forward's top bucket.

    Attributes:
        calls: ``int64 [k]`` each chunk's call.
        rows: ``int64 [k]`` each chunk's first candidate row.
        row_counts: ``int64 [k]`` each chunk's candidate rows.
        tokens: ``int64 [k]`` each chunk's first token in the forward.
        continued: ``int64 [k]`` what predicts each chunk's first token.
        slots: ``int64 [k]`` each chunk's slot.
        offsets: ``int64 [k]`` where each chunk's words start in the
            binding's logprob column.
        sizes: ``int64 [k]`` each chunk's words.
        scored: ``int64 [k]`` each chunk's scored rows.
        tails: ``int64 [k]`` the tail row receiving each chunk's last
            logits: its index.
        tops: ``int64 [k]`` each chunk's reported top count.
        top: The top bucket every chunk is scored at: that of its widest
            chunk that scores a token.
        end: The logprob column's words after the chunks.
    """

    calls: np.ndarray
    rows: np.ndarray
    row_counts: np.ndarray
    tokens: np.ndarray
    continued: np.ndarray
    slots: np.ndarray
    offsets: np.ndarray
    sizes: np.ndarray
    scored: np.ndarray
    tails: np.ndarray
    tops: np.ndarray
    top: int
    end: int

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def plan(
        cls,
        chunks: Sequence[PromptChunk],
        cursor: int,
        vocab: int,
        capacity: int,
    ) -> PromptChunks:
        """Lay ``chunks``' words out from ``cursor`` in the logprob column.

        Args:
            chunks: The forward's chunks, in order.
            cursor: The words the sampled tasks' logprobs pack first.
            vocab: The global vocabulary.
            capacity: The logprob words a forward may report.

        Raises:
            OutOfMemory: The words exceed the binding's capacity.
        """

        def column(values: Sequence[int]) -> np.ndarray:
            return np.asarray(values, dtype=np.int64).reshape(-1)

        scored = column([chunk.scored for chunk in chunks])
        tops = column([chunk.top for chunk in chunks])
        widest = int(np.max(tops[scored > 0], initial=0))
        sizes = column(
            [logprob_words(int(s), int(t)) for s, t in zip(scored, tops)]
        )
        offsets = cursor + np.cumsum(sizes) - sizes
        end = cursor + int(sizes.sum())
        if end > capacity:
            raise OutOfMemory("logprob words exceed the binding's capacity")
        return cls(
            calls=column([chunk.call for chunk in chunks]),
            rows=column([chunk.first for chunk in chunks]),
            row_counts=column([chunk.count for chunk in chunks]),
            tokens=column([chunk.token for chunk in chunks]),
            continued=column([chunk.continued for chunk in chunks]),
            slots=column([chunk.slot for chunk in chunks]),
            offsets=offsets.astype(np.int64),
            sizes=sizes,
            scored=scored,
            tails=np.arange(len(chunks), dtype=np.int64),
            tops=tops,
            top=top_bucket(widest, vocab),
            end=end,
        )

    def layout(self, chunk: int) -> LogprobLayout | None:
        """Return chunk ``chunk``'s word layout; none when it scores none."""
        scored = int(self.scored[chunk])
        if scored == 0:
            return None
        return LogprobLayout(
            rows=np.arange(scored, dtype=np.int64),
            counts=np.full(scored, self.tops[chunk], dtype=np.int64),
        )


def prompt_rows(
    chunks: PromptChunks, logit_rows: np.ndarray, spare: int
) -> np.ndarray:
    """Return the prompt rows of ``chunks`` that ``score_prompt`` scores.

    Per row (:data:`PROMPT_FIELDS`): its hidden position (the token
    position of a candidate row of ``logit_rows``), its retained slot (or
    -1), its target token's index in the forward, its chunk's first word,
    its index in the chunk, the chunk's scored rows and their top count,
    and its tail row. A chunk's scored rows come first, then one row of its
    last candidate row that reports nothing (count -1) and retains its
    logits in its tail row; a scored row copies into the ``spare`` tail
    row.

    Args:
        chunks: The forward's chunks.
        logit_rows: Each candidate row's token position.
        spare: The tail row nothing reads.

    Returns:
        ``int64 [rows, PROMPT_FIELDS]``.

    Raises:
        AssertionError: A chunk names a row its forward does not project.
    """
    logit_rows = np.asarray(logit_rows, dtype=np.int64)

    def position(rows: np.ndarray) -> np.ndarray:
        if np.any((rows < 0) | (rows >= len(logit_rows))):
            raise AssertionError(
                "a prompt chunk names a row its forward does not project"
            )
        return logit_rows[rows]

    blocks = []
    for chunk in range(len(chunks)):
        first = int(chunks.rows[chunk])
        count = int(chunks.row_counts[chunk])
        continued = int(chunks.continued[chunk])
        # A first chunk scores its tokens after the first.
        fresh = int(continued == 0)
        scored = max(count - fresh, 0)
        index = np.arange(scored, dtype=np.int64)
        # The candidate row predicting the chunk's token `index + fresh`,
        # counted from the chunk's first; -1 is what precedes the chunk.
        before = index + fresh - 1
        rows = np.empty((scored + 1, PROMPT_FIELDS), dtype=np.int64)
        within = before >= 0
        rows[:scored, 0] = 0
        rows[:scored, 1] = -1
        rows[:scored][within, 0] = position(first + before[within])
        if np.any(~within):
            if continued == 1:
                rows[:scored][~within, 1] = chunks.slots[chunk]
            else:
                rows[:scored][~within, 0] = position(
                    np.asarray([continued - 2])
                )
        rows[:scored, 2] = chunks.tokens[chunk] + index + fresh
        rows[:scored, 3] = chunks.offsets[chunk]
        rows[:scored, 4] = index
        rows[:scored, 5] = scored
        rows[:scored, 6] = chunks.tops[chunk]
        rows[:scored, 7] = spare
        last = first + max(count - 1, 0)
        rows[scored] = (
            position(np.asarray([last]))[0],
            -1,
            0,
            0,
            0,
            0,
            -1,
            chunks.tails[chunk],
        )
        blocks.append(rows)
    if not blocks:
        return np.zeros((0, PROMPT_FIELDS), dtype=np.int64)
    return np.concatenate(blocks)


def prompt_padding(spare: int) -> np.ndarray:
    """Return a prompt row that reports nothing (top count -1).

    It reads hidden position 0, retains no slot and copies into the
    ``spare`` tail row, which nothing reads.

    Returns:
        ``int64 [PROMPT_FIELDS]``.
    """
    return np.array([0, -1, 0, 0, 0, 0, -1, spare], dtype=np.int64)


def prompt_columns(rows: np.ndarray, block: int, spare: int) -> np.ndarray:
    """Return prompt rows field-major, padded to whole ``score_prompt`` calls.

    Each call scores ``block`` rows; padding rows are
    :func:`prompt_padding` rows.

    Returns:
        ``int64 [PROMPT_FIELDS, calls * block]``.
    """
    calls = -(-len(rows) // block)
    padded = np.tile(prompt_padding(spare), (calls * block, 1))
    padded[: len(rows)] = rows
    return np.ascontiguousarray(padded.T)
