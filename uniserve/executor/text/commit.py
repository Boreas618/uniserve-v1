"""How a text batch's calls advance and what its commit installs.

Each active call of a batch advances its request's progress and contributes
one slot-state commit row (:func:`finish_rows`), in the layout of
:mod:`uniserve.sampling.state`: ``[slot, mode, task, cache, logical,
sampling, visible, penalties]``. The progress goes into the batch's
completion records; the rows wait as the lane's :class:`CommitWork` until
the worker has sealed the batch and calls the lane's commit, which pads them
to a commit bucket (:func:`commit_bucket`) and installs them.

- a prefill without a task: ``[slot, CACHE, -1, len, 0, 0, 0, 0]``; its
  cache becomes ``len`` and its logical position is kept;
- a prefill with a task: ``[slot, EXPLICIT, task, len, logical + query,
  rng + 1, 0, pen]``; it advances by its query and one draw, cache ``len``;
- a decode: ``[slot, DECODE, task, 0, 0, 0, 0, pen]``; it advances by one
  token and one draw, cache ``prefix + 1``;
- a verification: ``[slot, VERIFY, task, 0, logical, rng, prefix, pen]``;
  its position and draws are kept, its visible length is ``prefix`` and its
  computed length ``prefix + query``: the speculation base the worker
  resolves acceptance from;
- a media call without a task: ``[slot, CACHE, -1, len, 0, 0, 0, 0]``; its
  position becomes ``media_position``, cache ``len``;
- a media call with a task: ``[slot, EXPLICIT, task, len,
  media_continued, rng + 1, 0, pen]``; its position becomes
  ``media_position``, one draw, cache ``len``.

``len`` is the row's prefix plus its query
(:func:`~uniserve.executor.text.rows.written_length`), at most the slot's
allocated length; ``pen`` is whether the request's penalties count tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uniserve.sampling import state as slot_commit

from .protocol import CallKind
from .rows import written_length
from .shapes import bucket

__all__ = [
    "COMMIT_PADDING",
    "CommitWork",
    "Finished",
    "commit_bucket",
    "finish_rows",
]

# A row padding a commit bucket: it stores the padding slot's zero cache
# length again.
COMMIT_PADDING = np.array(
    [0, slot_commit.CACHE, -1, 0, 0, 0, 0, 0], dtype=np.int64
)
COMMIT_PADDING.flags.writeable = False


@dataclass(slots=True)
class Finished:
    """Each call's commit row and progress after the batch.

    Attributes:
        rows: ``int64 [n, COMMIT_FIELDS]`` commit rows.
        logical: ``int64 [n]`` logical positions.
        rng: ``int64 [n]`` sampling counters.
        visible: ``int64 [n]`` visible cache lengths.
        computed: ``int64 [n]`` computed cache lengths.
        prompt_ready: ``bool [n]`` the slot retains prompt logits.
        speculation: ``bool [n]`` the progress is a verification's
            speculation base.
    """

    rows: np.ndarray
    logical: np.ndarray
    rng: np.ndarray
    visible: np.ndarray
    computed: np.ndarray
    prompt_ready: np.ndarray
    speculation: np.ndarray


def finish_rows(
    *,
    kind: np.ndarray,
    slot: np.ndarray,
    prefix: np.ndarray,
    query: np.ndarray,
    logical: np.ndarray,
    rng: np.ndarray,
    allocated: np.ndarray,
    task: np.ndarray,
    penalties: np.ndarray,
    prompt_ready: np.ndarray,
    chunks: np.ndarray,
    media: np.ndarray | None = None,
    media_position: np.ndarray | None = None,
    media_continued: np.ndarray | None = None,
) -> Finished:
    """Advance active calls' progress and return their commit rows.

    Every argument is a ``[n]`` array over the calls; ``prefix`` and
    ``query`` are those of each call's last row, from which it samples.

    Args:
        kind: Each call's :class:`~uniserve.executor.text.protocol.CallKind`.
        slot: Each call's slot.
        prefix: Each row's visible prefix.
        query: Each row's query tokens.
        logical: Each call's logical position before the batch.
        rng: Each call's sampling counter before the batch.
        allocated: Each slot's allocated cache length.
        task: Each call's output position among its invocation's tasks,
            -1 for a call that does not sample.
        penalties: Whether each request's penalties count tokens.
        prompt_ready: Whether each slot retained prompt logits before.
        chunks: Whether each call scored prompt chunks, which retain its
            last logits.
        media: Whether each call places image features; none by default.
        media_position: A media call's logical position after the call.
        media_continued: The position a sampling media call's slot
            continues from.

    Raises:
        AssertionError: A decode or verification does not sample, or a call
            writes beyond its query or its slot's allocated length.
    """
    n = len(kind)
    task = np.asarray(task, dtype=np.int64)
    logical = np.asarray(logical, dtype=np.int64)
    rng = np.asarray(rng, dtype=np.int64)
    prefix = np.asarray(prefix, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    # Without media calls every call is a text call: no position overrides.
    media = (
        np.zeros(n, dtype=bool)
        if media is None
        else np.asarray(media, dtype=bool)
    )
    media_position = (
        np.zeros(n, dtype=np.int64)
        if media_position is None
        else np.asarray(media_position, dtype=np.int64)
    )
    media_continued = (
        np.zeros(n, dtype=np.int64)
        if media_continued is None
        else np.asarray(media_continued, dtype=np.int64)
    )
    sampled = task >= 0
    decode = ~media & (kind == CallKind.DECODE)
    verify = ~media & (kind == CallKind.VERIFY)
    prefill = media | (kind == CallKind.PREFILL)
    if np.any((decode | verify) & ~sampled):
        raise AssertionError("decode and verification calls always sample")

    # Every call but a verification writes its rows' tokens (a decode its
    # one token) and makes them visible.
    writes = ~verify
    cache = np.zeros(n, dtype=np.int64)
    cache[writes] = written_length(
        prefix[writes],
        query[writes],
        np.where(decode, 1, query)[writes],
        np.asarray(allocated, dtype=np.int64)[writes],
    )

    advanced = np.where(
        media,
        media_position,
        np.where(
            prefill & sampled,
            logical + query,
            np.where(decode, logical + 1, logical),
        ),
    )
    counted = rng + (sampled & ~verify)
    explicit = prefill & sampled
    mode = np.where(
        decode,
        slot_commit.DECODE,
        np.where(
            verify,
            slot_commit.VERIFY,
            np.where(explicit, slot_commit.EXPLICIT, slot_commit.CACHE),
        ),
    )

    rows = np.zeros((n, slot_commit.COMMIT_FIELDS), dtype=np.int64)
    rows[:, slot_commit.SLOT] = slot
    rows[:, slot_commit.MODE] = mode
    rows[:, slot_commit.TASK] = np.where(sampled, task, -1)
    rows[:, slot_commit.CACHE_LENGTH] = np.where(prefill, cache, 0)
    rows[:, slot_commit.LOGICAL] = np.where(
        explicit,
        np.where(media, media_continued, advanced),
        np.where(verify, logical, 0),
    )
    rows[:, slot_commit.SAMPLING] = np.where(
        explicit, counted, np.where(verify, rng, 0)
    )
    rows[:, slot_commit.VISIBLE] = np.where(verify, prefix, 0)
    rows[:, slot_commit.PENALTIES] = np.where(
        sampled, np.asarray(penalties, dtype=np.int64), 0
    )
    return Finished(
        rows=rows,
        logical=advanced,
        rng=counted,
        visible=np.where(verify, prefix, cache),
        computed=np.where(verify, prefix + query, cache),
        prompt_ready=np.asarray(prompt_ready, dtype=bool)
        | np.asarray(chunks, dtype=bool),
        speculation=verify,
    )


def commit_bucket(rows: int, decode: Sequence[int], capacity: int) -> int:
    """Return the commit bucket of ``rows`` commit rows.

    The smallest decode bucket holding them, else the binding's rows, at
    which a batch of more calls than a decode batch holds commits.
    """
    held = bucket(decode, rows)
    return capacity if held is None else held


@dataclass(slots=True)
class CommitWork:
    """What a lane's commit installs after the worker sealed its batch.

    Attributes:
        rows: ``int64 [n, COMMIT_FIELDS]`` commit rows.
        tails: ``int64 [k, 2]`` ``(tail row, slot)`` pairs: the retained
            prompt logits each scored chunk copies into its slot.
    """

    rows: np.ndarray
    tails: np.ndarray

    def columns(self, bucket: int) -> np.ndarray:
        """Return the rows field-major, padded to ``bucket`` rows.

        Returns:
            ``int64 [COMMIT_FIELDS, bucket]``.

        Raises:
            AssertionError: The rows exceed the bucket.
        """
        if len(self.rows) > bucket:
            raise AssertionError("commit rows exceed the binding")
        padded = np.tile(COMMIT_PADDING, (bucket, 1))
        padded[: len(self.rows)] = self.rows
        return np.ascontiguousarray(padded.T)
