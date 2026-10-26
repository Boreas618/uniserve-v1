"""Call predicates a prepared batch applies on the device.

A call the engine queued behind an unresolved call carries a ``uint8``
predicate: its predecessor's completion flag. The worker leases it on the
computation's device behind its producer and never reads it on the host
(``crates/worker/src/executor/batch.rs``). The call runs whatever its
predicate holds; when it is false, the executor leaves every state the call
would change as it was (:func:`kept`), and the worker reports the call
predicated: no product of it is published, so what the call wrote into its
products is never read.

The worker appends a prepared batch's predicates after every other tensor,
in call order; each predicated call's entry names its predicate's position
in the batch's tensors (``entry["predicate"]``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

__all__ = ["kept", "split"]


def split(
    calls: Sequence[Mapping[str, Any]], tensors: Sequence[torch.Tensor]
) -> tuple[list[torch.Tensor], dict[int, torch.Tensor]]:
    """A prepared batch's tensors without its predicates, and the predicates.

    Returns the tensors before the predicates, and each predicated call's
    ``uint8 [1]`` device predicate by the call's index.

    Raises:
        ValueError: The predicates are not the batch's last tensors, one
            per predicated call in call order.
    """
    positions = {
        int(entry["index"]): int(entry["predicate"])
        for entry in calls
        if entry.get("predicate") is not None
    }
    if not positions:
        return list(tensors), {}
    first = len(tensors) - len(positions)
    if sorted(positions.values()) != list(range(first, len(tensors))):
        raise ValueError("a batch's predicates follow its other tensors")
    predicates = {
        index: tensors[position] for index, position in positions.items()
    }
    return list(tensors[:first]), predicates


def kept(
    predicate: torch.Tensor | None,
    value: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    """``value`` where ``predicate`` holds and ``previous`` where it does not.

    The value a write of a call leaves: what it computed, or, for a call its
    false predicate skipped, what the state held before. ``predicate`` is the
    call's ``uint8 [1]`` device predicate; without one the call always runs.
    The selection is a device operation in stream order: nothing waits for
    the predicate on the host.
    """
    if predicate is None:
        return value
    return torch.where(predicate.bool(), value, previous)
