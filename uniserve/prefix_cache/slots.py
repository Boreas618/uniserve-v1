"""Per-request states: the cache layers of types indexed by request slot.

A cache layer type with ``indexing == "states"`` (``StateConfig``) keeps one
block per request slot of the rank: fields shaped ``[slots + 1, ...]``,
which the worker allocates beside the unit pool and the block tables (slot
0 is the padding slot padding rows address). A request keeps its slot from
its first forward to its end, so its state carries from one forward to the
next; a forward whose row has no prefix (its first) starts the row's state
afresh, whatever an earlier request left in the slot. :func:`rows` gives
the slot of every row (sequence) of the causal LM forward running.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch

from .state import State, StateConfig

# The request slot of every row of the forward running, while one runs.
_rows: torch.Tensor | None = None


def rows() -> torch.Tensor | None:
    """The request slot of every row of the causal LM forward running.

    An int64 ``[rows]`` tensor at a fixed address, in the order of the
    forward's sequences (``AttentionBatch`` rows); ``None`` outside such a
    forward.
    """
    return _rows


@contextmanager
def selected(slots: torch.Tensor) -> Iterator[None]:
    """Run a forward whose rows are the request slots ``slots``."""
    global _rows
    previous, _rows = _rows, slots
    try:
        yield
    finally:
        _rows = previous


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
