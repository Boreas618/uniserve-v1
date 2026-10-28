"""Per-request states: the cache layers of types indexed by request slot.

A cache layer type with ``indexing == "states"`` (``StateConfig``) keeps one
block per request slot of the rank: fields shaped ``[slots + 1, ...]``,
which the worker allocates beside the unit pool and the block tables (slot
0 is the padding slot padding rows address). A request keeps its slot from
its first forward to its end, so its state carries from one forward to the
next; a forward whose row has no prefix (its first) starts the row's state
afresh, whatever an earlier request left in the slot, and a row whose
tables name a checkpoint at its prefix starts from that checkpoint
(:mod:`uniserve.prefix_cache.checkpoints`). :func:`rows` gives the slot of
every row (sequence) of the causal LM forward running.

Two kinds of forward leave a row's slot as it was:

- a forward whose rows keep no state (:func:`kept`): rows evaluating a
  request's earlier tokens again, from a checkpoint, while the request's
  slot holds its state at its latest token. They write their checkpoints;
- a verification (:class:`Verification`): its rows hold a request's next
  token and draft tokens, of which the request accepts a prefix only once
  the forward's logits are selected. Each layer defers its slots' advance
  to :meth:`Verification.accept`, which advances them over the accepted
  tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

import torch

from .state import State, StateConfig

# The request slot of every row of the forward running, while one runs.
_rows: torch.Tensor | None = None
# The rows of that forward that keep their final state in their slots;
# ``None`` when every row does.
_kept: torch.Tensor | None = None
# The verification that forward is, if it is one.
_verification: Verification | None = None


class Verification:
    """The deferred slot advances of one verification forward.

    Each cache layer kept per request slot defers, while the forward runs,
    the advance of its rows' slots over a number of their tokens
    (:meth:`defer`). Once the forward's selections are known, the caller
    accepts each row's token count (:meth:`accept`) before the runner's
    next forward, whose metadata the advances read.
    """

    def __init__(self) -> None:
        self._advances: list[Callable[[torch.Tensor], None]] = []

    def defer(self, advance: Callable[[torch.Tensor], None]) -> None:
        """Keep one layer's advance of the forward's rows' slots."""
        self._advances.append(advance)

    def accept(self, counts: torch.Tensor) -> None:
        """Advance every row's slots over its first ``counts`` tokens.

        ``counts`` is int64 ``[rows]``, at most each row's tokens; zero
        leaves a row's slots as they were. Each deferred advance runs once.
        """
        advances, self._advances = self._advances, []
        for advance in advances:
            advance(counts)


def rows() -> torch.Tensor | None:
    """The request slot of every row of the causal LM forward running.

    An int64 ``[rows]`` tensor at a fixed address, in the order of the
    forward's sequences (``AttentionBatch`` rows); ``None`` outside such a
    forward.
    """
    return _rows


def kept() -> torch.Tensor | None:
    """Which rows of the forward running keep their final state in their
    slots: bool ``[rows]``, or ``None`` when every row does.
    """  # noqa: D205
    return _kept


def verification() -> Verification | None:
    """The verification the forward running is, ``None`` for another."""
    return _verification


@contextmanager
def selected(
    slots: torch.Tensor,
    *,
    kept: torch.Tensor | None = None,
    verification: Verification | None = None,
) -> Iterator[None]:
    """Run a forward whose rows are the request slots ``slots``.

    ``kept`` (bool, one entry per row) names the rows that keep their final
    state in their slots, every row when ``None``; a ``verification``
    collects the advances of a verification forward's slots instead.
    """
    global _rows, _kept, _verification
    previous = (_rows, _kept, _verification)
    _rows, _kept, _verification = slots, kept, verification
    try:
        yield
    finally:
        _rows, _kept, _verification = previous


def fields(
    config: StateConfig,
) -> Mapping[str, tuple[tuple[int, ...], torch.dtype]]:
    """One slot's extent and element type of each field of a ``"states"``
    layer type.
    """  # noqa: D205
    declared = config.buffers(
        num_blocks=1, block_size=1, dtype=None, quantizer=None
    )
    return {
        name: (tuple(requirement.shape[1:]), requirement.dtype)
        for name, requirement in declared.items()
    }


def bind(
    configs: Mapping[str, StateConfig], banks: Mapping[str, torch.Tensor]
) -> dict[str, State]:
    """Each state layer's state over its banks: field ``f`` of layer ``l``
    is bank ``"l/f"``, ``[slots + 1, ...]``.

    Raises:
        ValueError: A bank is missing or disagrees with its field.
    """  # noqa: D205
    states = {}
    for name, config in configs.items():
        tensors = {}
        for field, (shape, dtype) in fields(config).items():
            bank = banks.get(f"{name}/{field}")
            if (
                bank is None
                or tuple(bank.shape[1:]) != shape
                or bank.dtype != dtype
            ):
                raise ValueError(
                    f"the request state bank {name}/{field} disagrees with "
                    "its field"
                )
            tensors[field] = bank
        states[name] = config.bind(
            tensors, block_size=1, dtype=None, quantizer=None
        )
    return states
