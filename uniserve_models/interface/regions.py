"""Typed windows of byte arenas the worker binds once.

The worker binds each device's product arena (the storage of every
cross-call tensor product on that device) to a binding once, as a ``uint8``
tensor. A call then names a product by its *region*: a TVM-FFI array
``[offset, dtype, shape]`` with the byte ``offset`` of the product's
storage in the arena, its torch dtype name and its shape. :func:`view`
turns a region into a tensor view without importing another DLPack tensor.
The worker aligns every product's offset to its element size.
"""

from __future__ import annotations

from math import prod

import torch

__all__ = ["view"]


def view(arena: torch.Tensor, region) -> torch.Tensor:
    """Return the typed, shaped view ``region`` names in ``arena``.

    Args:
        arena: The bound one-dimensional ``uint8`` arena.
        region: ``[offset, dtype, shape]``: a byte offset, a torch dtype
            name such as ``"bfloat16"`` and a shape.

    Raises:
        ValueError: The region is malformed or exceeds the arena.
    """
    offset, dtype, shape = region
    offset, shape = int(offset), tuple(int(extent) for extent in shape)
    dtype = getattr(torch, str(dtype), None)
    if not isinstance(dtype, torch.dtype) or offset < 0:
        raise ValueError(f"malformed arena region {list(region)!r}")
    size = prod(shape) * dtype.itemsize
    if offset % dtype.itemsize or offset + size > arena.numel():
        raise ValueError("arena region exceeds or misaligns its arena")
    return arena[offset : offset + size].view(dtype).view(shape)
