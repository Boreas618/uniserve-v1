"""Execution-context workspace served from worker allocations on demand.

Most bindings declare their workspace before binding (``component.prepare``
records it with a :class:`~uniserve_models.bindings.context.DeclaredWorkspace`
and the worker allocates it). Some call sites only learn their workspace at
their first numerical call: a sparse attention layer prepares its operators,
projection buffers and exchange storage for the block pattern of the first
input it sees. A :class:`RuntimeWorkspace` serves such requests whenever
they arrive by calling back into the worker, which allocates every field
from its own storage and returns it as a DLPack tensor that retains the
allocation.

Shared requests carry their role's field name (``"{role}/{field}"``): the
worker keeps one backing per field on the binding's stream, so contexts that
run serialized on that stream share it, and a larger request replaces the
backing. A replaced backing stays allocated until the provider closes,
because graphs captured earlier address it. A request made while a graph is
being captured is served only from an existing backing, because a capture
must only address storage that existed before it began.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

import torch

from uniserve_models.bindings import capturing
from uniserve_models.bindings.context import role_name
from uniserve_models.tensors import BufferConfig


class RuntimeWorkspace:
    """Workspace provider backed by the worker's allocation callback.

    ``allocate(name, role, dtype, shape, host)`` returns a tensor of exactly
    ``shape`` and ``dtype``: a fresh allocation for an empty ``role``, or a
    view of the shared backing of field ``name`` otherwise. Allocations live
    as long as the tensors referencing them.
    """

    def __init__(self, allocate: Callable[..., object], name: str) -> None:
        self._allocate = allocate
        self._name = name
        self._sequence = 0
        # The flat backing each shared field was last served from. Operators
        # request their scratch on every call, so a request that fits the
        # backing is served from it without calling the worker, which a
        # graph capture requires.
        self._shared: dict[str, torch.Tensor] = {}
        self._replaced: list[torch.Tensor] = []

    def _serve(
        self,
        prefix: str,
        role: str,
        requirements: Mapping[str, BufferConfig],
        device: torch.device | str,
    ) -> Mapping[str, torch.Tensor]:
        device = torch.device(device)
        views = {}
        for field, config in requirements.items():
            name = f"{prefix}/{field}"
            elements = _elements(config.shape)
            backing = self._shared.get(name) if role else None
            if (
                backing is None
                or backing.dtype != config.dtype
                or backing.numel() < elements
            ):
                if capturing(device):
                    raise RuntimeError(
                        f"prepare {prefix!r} workspace for this size before "
                        "capture"
                    )
                shape = tuple(config.capacity_shape or config.shape)
                backing = torch.from_dlpack(
                    self._allocate(
                        name,
                        role,
                        str(config.dtype).removeprefix("torch."),
                        list(shape),
                        bool(config.host),
                    )
                ).reshape(-1)
                if role:
                    previous = self._shared.get(name)
                    if previous is not None:
                        self._replaced.append(previous)
                    self._shared[name] = backing
            views[field] = backing[:elements].view(config.shape)
        return MappingProxyType(views)

    def scratch(self, role, requirements, device):
        name = role_name(role)
        return self._serve(name, name, requirements, device)

    def allocate(self, requirements, device):
        prefix = f"{self._name}/{self._sequence:04d}"
        self._sequence += 1
        return self._serve(prefix, "", requirements, device)

    def storage(self, requirement):
        """Back one attention storage requirement with local fields.

        Raises:
            RuntimeError: A field must be peer-mapped across its group, which
                the parallel runtime's symmetric storage serves.
        """
        if requirement.symmetric:
            raise RuntimeError(
                "peer-mapped attention storage is served by the parallel "
                "runtime"
            )
        tensors = dict(
            self.allocate(requirement.fields, requirement.group.device)
        )
        return tensors, {}

    def release(self) -> None:
        # Persistent allocations are released with the views holding them.
        self._sequence = 0

    def close(self) -> None:
        self.release()
        self._shared.clear()
        self._replaced.clear()


def _elements(shape) -> int:
    count = 1
    for extent in shape:
        count *= int(extent)
    return count
