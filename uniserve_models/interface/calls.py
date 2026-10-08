"""Shared scaffolding of capability step functions.

A binding's step functions run on the binding's stream with its bound
operators (:func:`scope`), read the worker's buffers through torch views
imported once at bind time (:func:`tensors`), and publish results into
product regions (:func:`write`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

__all__ = ["module", "scope", "tensors", "write"]


def module(prepared, path: str):
    """The module at ``path`` (relative to the binding root; ``""`` is it)."""
    path = str(path)
    return prepared.module.get_submodule(path) if path else prepared.module


def tensors(buffers: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Torch views of the worker's buffers, imported once."""
    return {
        str(name): torch.from_dlpack(value) for name, value in buffers.items()
    }


@contextmanager
def scope(context) -> Iterator[None]:
    """Run one step on the binding's stream with its bound operators."""
    with torch.inference_mode(), context.activate():
        yield


def write(target: torch.Tensor, value: torch.Tensor) -> list[int]:
    """Copy ``value`` into the leading elements of a product region.

    Converts to the product's dtype on the current stream and returns the
    value's shape, which the worker publishes as the product's shape.

    Raises:
        ValueError: The value exceeds the region.
    """
    shape = list(value.shape)
    if value.numel() > target.numel():
        raise ValueError("a result exceeds its product bound")
    target.reshape(-1)[: value.numel()].view(value.shape).copy_(
        value.detach(), non_blocking=True
    )
    return shape
