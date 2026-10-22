"""LoRA adapters: the adapter each token of a forward computes with.

The extension point of LoRA serving. A plugin keeps an adapter pool at
fixed device addresses, allocated before the CUDA graphs are captured, and
loads and unloads adapters into its slots through the worker hooks
``load_lora_adapter`` and ``unload_lora_adapter`` (``uniserve.hooks``),
whose payload's ``adapter`` names the slot the daemon assigned. A request
naming an adapter as its ``model`` computes with that slot, and the
plugin's kernels read, for every token of the causal LM forward running,
the slot of its request's adapter: :func:`tokens`, a ``[T]`` int64 tensor
at a fixed address, which the runner fills inside the forward, captured or
not; 0 selects no adapter.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from uniserve.nn.attention.gather import ROW_SECTIONS, row_columns_stride

# The adapter slot of every token of the forward running, while one runs.
_tokens: torch.Tensor | None = None


def tokens() -> torch.Tensor | None:
    """The adapter slot of every token of the causal LM forward running.

    ``None`` outside such a forward.
    """
    return _tokens


@contextmanager
def selected(slots: torch.Tensor) -> Iterator[None]:
    """Run a forward whose tokens compute with the adapters ``slots``."""
    global _tokens
    previous, _tokens = _tokens, slots
    try:
        yield
    finally:
        _tokens = previous


def gather_rows(
    slot_adapters: torch.Tensor,
    slots: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write the adapter of each of ``slots`` (one token each) into ``out``.

    ``slot_adapters`` is the ``[S]`` int64 adapter of every request slot.
    Returns ``out``, written in place, so a captured graph reads it there.
    """
    return torch.index_select(slot_adapters, 0, slots, out=out)


def gather_tokens(
    slot_adapters: torch.Tensor,
    row_columns: torch.Tensor,
    max_rows: int,
    tables: int,
    token_slots: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write the adapter of each packed token into ``out``.

    ``row_columns`` is a prefill's packed int64 row columns
    (``uniserve.nn.attention.gather.gather_request_rows``) for at most
    ``max_rows`` rows over ``tables`` tables: each row's slot, and each
    token's row. ``token_slots`` receives each token's slot and ``out`` its
    adapter, for ``len(out)`` tokens; both are written in place.
    """
    stride = row_columns_stride(max_rows)
    rows = row_columns[(ROW_SECTIONS + tables) * stride :][: out.shape[0]]
    torch.index_select(row_columns[:stride], 0, rows, out=token_slots)
    return torch.index_select(slot_adapters, 0, token_slots, out=out)
