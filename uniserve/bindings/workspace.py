"""Workspace backing of execution contexts.

A :class:`Workspace` backs the bindings of one execution context: it
allocates persistent fields with ``torch.empty`` and borrows the shared
scratch of its stream from a :class:`Scratch`.
:class:`TensorBuffers` names tensor backing, allocated or borrowed, and
lends compact views into it; a :class:`GatherPool` lends reusable gather
buffers on a stream. :func:`close_resources` attempts every release of an
ordered close.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType

import torch

from uniserve.bindings import capturing
from uniserve.tensors import BufferConfig


def close_resources(*actions: Callable[[], object]) -> None:
    """Attempt every ordered release.

    Raises the first failure, with later failures added as notes.
    """
    failure: BaseException | None = None
    for action in actions:
        try:
            action()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"Resource cleanup also failed: {error!r}")
    if failure is not None:
        raise failure


class TensorBuffers:
    """Named tensor backing and compact views into it.

    Backing is either allocated here (:meth:`allocate`) or borrowed from the
    caller (:meth:`from_tensors`, for example Rust-owned DLPack tensors).
    Views borrow compact leading elements of a field's backing, not a strided
    crop, and are cached per requested layout so repeated preparation keeps
    both addresses and values.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self._tensors = dict(tensors)
        self._views: dict[tuple, Mapping[str, torch.Tensor]] = {}
        self._closed = False

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor]) -> TensorBuffers:
        """Borrow contiguous external tensors without changing their values.

        The caller keeps ownership: closing these buffers drops references
        only.
        """
        for name, tensor in tensors.items():
            if (
                not isinstance(name, str)
                or not name
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "tensor backing requires nonempty names and contiguous "
                    "storage"
                )
        return cls(tensors)

    @classmethod
    def allocate(
        cls,
        configs: Mapping[str, BufferConfig],
        *,
        device: torch.device | str,
        pin_storage: bool = False,
    ) -> TensorBuffers:
        """Allocate each field's capacity; values are uninitialized."""
        device = torch.device(device)
        tensors = {}
        for name, config in configs.items():
            shape = config.capacity_shape or config.shape
            tensors[name] = torch.empty(
                shape,
                dtype=config.dtype,
                device="cpu" if config.host else device,
                pin_memory=config.host and pin_storage,
            )
        return cls(tensors)

    def backing(self, name: str) -> torch.Tensor:
        """Borrow a field's entire capacity without changing its layout."""
        self._require_open()
        try:
            return self._tensors[name]
        except KeyError:
            raise ValueError(f"tensor {name!r} has no backing") from None

    def view(
        self, configs: Mapping[str, BufferConfig]
    ) -> Mapping[str, torch.Tensor]:
        """Borrow each field's leading elements in its requested shape.

        Raises:
            ValueError: A field has no backing of the requested dtype, needs
                host representation it does not have, or exceeds its
                capacity in some dimension.
        """
        self._require_open()
        key = tuple(configs.items())
        cached = self._views.get(key)
        if cached is not None:
            return cached
        views = {}
        for name, config in configs.items():
            tensor = self._tensors.get(name)
            if tensor is None or tensor.dtype != config.dtype:
                raise ValueError(
                    f"tensor {name} has no backing with the declared dtype"
                )
            if config.host and tensor.device.type != "cpu":
                raise ValueError(f"tensor {name} requires host representation")
            capacity = tuple(tensor.shape)
            if len(config.shape) != len(capacity) or any(
                size > limit
                for size, limit in zip(config.shape, capacity, strict=True)
            ):
                raise ValueError(f"tensor {name} exceeds resident capacity")
            elements = 1
            for size in config.shape:
                elements *= size
            views[name] = tensor.reshape(-1)[:elements].view(config.shape)
        result = MappingProxyType(views)
        self._views[key] = result
        return result

    def close(self) -> None:
        self._views.clear()
        self._tensors.clear()
        self._closed = True

    def __enter__(self) -> TensorBuffers:
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("tensor buffers are closed")


def _schema(configs: Mapping[str, BufferConfig]) -> tuple:
    """Fields, dtypes, placement and rank: what one shared backing serves."""
    return tuple(
        (name, config.dtype, config.host, len(config.shape))
        for name, config in configs.items()
    )


class Scratch:
    """Transient work areas shared by contexts serialized on one stream.

    One backing list per (role, device, schema): a request reuses the newest
    backing that fits; a larger request allocates a new backing in front and
    keeps the older ones, which already captured graphs still address. In a
    worker, backings on ``device`` allocate from ``pool``, the allocator pool
    of the device memory the worker owns for this scratch. A context whose
    module computes on another device (a generation tower) takes backings
    there from torch's own allocator: a pool's use count lives on the
    device that created it, so using it on another device would release it
    there at the end of each allocation.
    """

    def __init__(
        self,
        pool: torch.cuda.MemPool | None = None,
        device: int | None = None,
    ) -> None:
        if pool is not None and device is None:
            raise ValueError("a scratch pool needs the device it belongs to")
        self._backings: dict[tuple, list[TensorBuffers]] = {}
        self._pool = pool
        self._device = (
            None if device is None else torch.device("cuda", int(device))
        )

    def view(
        self,
        role: object,
        requirements: Mapping[str, BufferConfig],
        device: torch.device | str,
    ) -> Mapping[str, torch.Tensor]:
        device = torch.device(device)
        backings = self._backings.setdefault(
            (role, device, _schema(requirements)), []
        )
        for backing in backings:
            try:
                return backing.view(requirements)
            except ValueError:
                continue
        if capturing(device):
            raise RuntimeError(
                f"prepare {role!r} scratch for this size before capture"
            )
        pool = (
            torch.cuda.use_mem_pool(self._pool, device)
            if self._pool is not None and device == self._device
            else contextlib.nullcontext()
        )
        with pool:
            backing = TensorBuffers.allocate(requirements, device=device)
        backings.insert(0, backing)
        return backing.view(requirements)

    def close(self) -> None:
        for backings in self._backings.values():
            for backing in backings:
                backing.close()
        self._backings.clear()


class Workspace:
    """The workspace of an execution context, in torch memory.

    Shared scratch is grown and kept by a :class:`Scratch`; persistent
    fields are tensors of their own.

    Args:
        scratch: The shared :class:`Scratch` contexts serialized on one
            stream borrow; ``None`` creates one this workspace owns.
    """

    def __init__(self, scratch: Scratch | None = None) -> None:
        self._owns_scratch = scratch is None
        self._scratch = Scratch() if scratch is None else scratch
        self._allocations: list[TensorBuffers] = []

    def scratch(self, role, requirements, device):
        return self._scratch.view(role, requirements, device)

    def allocate(self, requirements, device):
        allocation = TensorBuffers.allocate(requirements, device=device)
        self._allocations.append(allocation)
        return allocation.view(requirements)

    def storage(self, requirement):
        """Back one attention storage requirement on its group's device.

        Returns the tensor of every declared field.
        """
        return dict(self.allocate(requirement.fields, requirement.group.device))

    def release(self) -> None:
        """Release persistent allocations; shared scratch survives sizes."""
        actions = [
            allocation.close for allocation in reversed(self._allocations)
        ]
        self._allocations.clear()
        close_resources(*actions)

    def close(self) -> None:
        self.release()
        if self._owns_scratch:
            self._scratch.close()


class GatherPool:
    """Reusable gather buffers for nested projection consumers on a stream.

    A borrow lends distinct storage until its consumer has enqueued its
    reads. Capacity is never released while graphs may address it.
    """

    def __init__(self, group) -> None:
        self.group = group
        self._buffers: list[torch.Tensor] = []
        self._borrowed: set[int] = set()

    @contextlib.contextmanager
    def borrow(
        self, size: int, device: torch.device, *, capacity: int | None = None
    ) -> Iterator[torch.Tensor]:
        if torch.device(device) != self.group.device:
            raise ValueError(
                "projection exchange must use its communicator's device"
            )
        if capacity is not None and size > capacity:
            raise ValueError("projection exchange exceeds the bound workspace")
        amount = size if capacity is None else capacity
        slot = next(
            (
                index
                for index, buffer in enumerate(self._buffers)
                if index not in self._borrowed and buffer.numel() >= amount
            ),
            None,
        )
        if slot is None:
            if capturing(torch.device(device)):
                raise RuntimeError(
                    "prepare projection exchange backing before capture"
                )
            self._buffers.append(
                torch.empty(amount, dtype=torch.uint8, device=device)
            )
            slot = len(self._buffers) - 1
        self._borrowed.add(slot)
        try:
            yield self._buffers[slot]
        finally:
            self._borrowed.discard(slot)
