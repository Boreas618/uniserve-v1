"""Numerical prefix state layouts and borrowed block storage."""

from typing import NamedTuple

from torch import nn

from .state import Config, State, StateConfig, block_spans


class CacheBinding(NamedTuple):
    """A cache layer's state in the active execution context, and the block
    table its pages are addressed through (``None`` for a ``"states"``
    layer, addressed by request slot).
    """  # noqa: D205

    state: State
    table: int | None


def bound(module: nn.Module) -> CacheBinding | None:
    """The cache layer ``module`` owns, as the active execution context
    binds it; ``None`` outside one or for a module owning none.
    """  # noqa: D205
    from uniserve.nn import _binding

    return _binding.cache.get().get(id(module))


__all__ = [
    "CacheBinding",
    "Config",
    "State",
    "StateConfig",
    "block_spans",
    "bound",
]
