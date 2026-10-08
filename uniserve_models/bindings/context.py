"""Execution contexts: numerical bindings and their workspace backing.

An :class:`ExecutionContext` prepares one module tree for a maximum size: it
walks the tree, creates a binding (prepared operator, plan and workspace) for
every numerical call site, and installs those bindings into the
``uniserve_models.nn._binding`` context variables while a call runs.

Workspace backing comes from a provider:

* :class:`StandaloneWorkspace` (the default) allocates with ``torch.empty``,
  for direct library use.
* :class:`DeclaredWorkspace` records every request without allocating. The
  worker uses it to learn a binding's requirements, allocates them itself and
  hands them back as tensors.
* :class:`BoundWorkspace` serves the same requests, in the same order and by
  the same names, from tensors the caller supplies.

A preparation walk is deterministic for one module and size, so a declared
walk and a bound walk issue identical requests. Shared scratch requests are
named ``"{role}/{field}"``; persistent allocations ``"{sequence:04d}/{field}"``
in request order.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import GenericAlias, MappingProxyType
from typing import Any

import torch
from torch import nn

from uniserve_models.bindings import capturing
from uniserve_models.tensors import BufferConfig

# Owners whose asynchronous accesses could not be proven finished. An aborted
# context retains them until the process exits instead of releasing storage
# that device work may still address.
_retained: list[object] = []


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


def streams_idle(streams: Iterable[torch.cuda.Stream]) -> bool:
    """Report, without waiting, whether every stream finished its work.

    A device fault reported by the query leaves completion unproven, so it
    counts as unfinished work.
    """
    try:
        return all(stream.query() for stream in streams)
    except RuntimeError:
        return False


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
    keeps the older ones, which already captured graphs still address.
    """

    def __init__(self) -> None:
        self._backings: dict[tuple, list[TensorBuffers]] = {}

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
        backing = TensorBuffers.allocate(requirements, device=device)
        backings.insert(0, backing)
        return backing.view(requirements)

    def close(self) -> None:
        for backings in self._backings.values():
            for backing in backings:
                backing.close()
        self._backings.clear()


def role_name(role: object) -> str:
    """The stable name of a scratch role (``("vsa", 0)`` -> ``"vsa.0"``)."""
    if isinstance(role, tuple):
        return ".".join(str(part) for part in role)
    return str(role)


@dataclass(frozen=True)
class Requirement:
    """One workspace field a binding needs.

    ``role`` names a shared scratch area (one backing per stream, sized for
    every binding prepared on it); ``None`` marks persistent storage owned
    by this binding alone. ``symmetric`` lists the group members (process
    ranks, in member order) across which a persistent field must be
    peer-mapped; ``None`` for local storage.
    """

    name: str
    role: str | None
    dtype: torch.dtype
    shape: tuple[int, ...]
    host: bool
    device: torch.device
    symmetric: tuple[int, ...] | None = None


# Peer views of a symmetric field ``{prefix}/{field}`` are served as
# ``{prefix}/{field}.peer{index}``, one per group member in member order.
def _peer_name(name: str, index: int) -> str:
    return f"{name}.peer{index}"


def _split(requirement) -> tuple[dict, dict]:
    """Local and symmetric fields of an attention storage requirement."""
    local = {
        name: config
        for name, config in requirement.fields.items()
        if name not in requirement.symmetric
    }
    symmetric = {
        name: requirement.fields[name] for name in requirement.symmetric
    }
    return local, symmetric


class StandaloneWorkspace:
    """Allocate workspace with the caching allocator, for library calls."""

    def __init__(self, scratch: Scratch | None = None) -> None:
        self._owns_scratch = scratch is None
        self._scratch = Scratch() if scratch is None else scratch
        self._allocations: list[TensorBuffers] = []
        # Peer-mapped allocations, retained until release.
        self._symmetric: list[Any] = []

    def scratch(self, role, requirements, device):
        return self._scratch.view(role, requirements, device)

    def allocate(self, requirements, device):
        allocation = TensorBuffers.allocate(requirements, device=device)
        self._allocations.append(allocation)
        return allocation.view(requirements)

    def storage(self, requirement):
        """Back one attention storage requirement on its group's device.

        Returns the local tensor of every field and, for each symmetric
        field, every member's peer-mapped view in member order.
        """
        local, symmetric = _split(requirement)
        tensors = dict(self.allocate(local, requirement.group.device))
        peers = {}
        if symmetric:
            # Peer-mapped storage comes from the parallel runtime, which maps
            # every member's allocation; import it only where it is needed.
            from uniserve_models.runtime._peer_storage import (
                allocate_symmetric_storage,
            )

            for name, config in symmetric.items():
                shared = allocate_symmetric_storage(
                    requirement.group, config.shape, config.dtype
                )
                self._symmetric.append(shared)
                tensors[name] = shared.local
                peers[name] = tuple(shared.peers)
        return tensors, peers

    def release(self) -> None:
        """Release persistent allocations; shared scratch survives sizes."""
        actions = [
            allocation.close for allocation in reversed(self._allocations)
        ]
        self._allocations.clear()
        self._symmetric.clear()
        close_resources(*actions)

    def close(self) -> None:
        self.release()
        if self._owns_scratch:
            self._scratch.close()


class DeclaredWorkspace:
    """Record workspace requests and serve meta-device placeholders.

    Nothing is allocated. Repeated requests for one shared field record its
    elementwise largest extent; persistent requests are numbered in order.
    """

    def __init__(self) -> None:
        self._requirements: dict[str, Requirement] = {}
        self._sequence = 0

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        return tuple(self._requirements.values())

    def scratch(self, role, requirements, device):
        prefix = role_name(role)
        return self._record(prefix, prefix, requirements, device)

    def allocate(self, requirements, device):
        prefix = f"{self._sequence:04d}"
        self._sequence += 1
        return self._record(prefix, None, requirements, device)

    def storage(self, requirement):
        """Record an attention storage requirement as persistent fields.

        Symmetric fields carry the group's members, so the allocator maps
        each member's allocation into every other member.
        """
        prefix = f"{self._sequence:04d}"
        self._sequence += 1
        members = tuple(requirement.group.ranks)
        tensors = dict(
            self._record(
                prefix, None, requirement.fields, requirement.group.device
            )
        )
        peers = {}
        for name in requirement.symmetric:
            key = f"{prefix}/{name}"
            self._requirements[key] = dataclasses.replace(
                self._requirements[key], symmetric=members
            )
            peers[name] = tuple(tensors[name] for _ in members)
        return tensors, peers

    def _record(self, prefix, role, requirements, device):
        device = torch.device(device)
        views = {}
        for field, config in requirements.items():
            name = f"{prefix}/{field}"
            shape = tuple(config.capacity_shape or config.shape)
            previous = self._requirements.get(name)
            if previous is not None:
                if (
                    previous.dtype != config.dtype
                    or len(previous.shape) != len(shape)
                    or previous.host != config.host
                ):
                    raise ValueError(
                        f"workspace field {name} changes its representation"
                    )
                shape = tuple(
                    max(old, new)
                    for old, new in zip(previous.shape, shape, strict=True)
                )
            self._requirements[name] = Requirement(
                name, role, config.dtype, shape, config.host, device
            )
            views[field] = torch.empty(
                config.shape, dtype=config.dtype, device="meta"
            )
        return MappingProxyType(views)

    def release(self) -> None:
        self._requirements.clear()
        self._sequence = 0

    def close(self) -> None:
        self.release()


class BoundWorkspace:
    """Serve declared requests from caller-owned tensors.

    The tensors are borrowed (for example Rust allocations exported through
    DLPack) and never released here. A shared field's tensor may exceed one
    request; requests take compact leading-element views.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self._buffers = TensorBuffers.from_tensors(tensors)
        self._sequence = 0

    def scratch(self, role, requirements, device):
        return self._serve(role_name(role), requirements)

    def allocate(self, requirements, device):
        prefix = f"{self._sequence:04d}"
        self._sequence += 1
        return self._serve(prefix, requirements)

    def storage(self, requirement):
        """Serve an attention storage requirement from its declared names.

        A symmetric field's peer views are served as
        ``{prefix}/{field}.peer{index}`` in member order.
        """
        prefix = f"{self._sequence:04d}"
        self._sequence += 1
        tensors = dict(self._serve(prefix, requirement.fields))
        peers = {}
        for name in requirement.symmetric:
            config = requirement.fields[name]
            named = {
                _peer_name(f"{prefix}/{name}", index): config
                for index in range(requirement.group.size)
            }
            try:
                views = self._buffers.view(named)
            except ValueError as error:
                raise ValueError(
                    f"bound workspace has no peer views of {prefix}/{name}"
                ) from error
            peers[name] = tuple(views[key] for key in named)
        return tensors, peers

    def _serve(self, prefix, requirements):
        named = {
            f"{prefix}/{field}": config
            for field, config in requirements.items()
        }
        try:
            views = self._buffers.view(named)
        except ValueError as error:
            raise ValueError(
                f"bound workspace does not serve {sorted(named)}: {error}"
            ) from error
        return MappingProxyType(
            {name.split("/", 1)[1]: tensor for name, tensor in views.items()}
        )

    def release(self) -> None:
        self._sequence = 0

    def close(self) -> None:
        self._buffers.close()


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


def _representation(module, cache, stream):
    """The root device and compute dtype of a module tree."""
    reference = next(
        (value for value in module.parameters() if not value.is_meta), None
    )
    if reference is None:
        reference = next(
            (value for value in module.buffers() if not value.is_meta), None
        )
    if reference is not None:
        device = reference.device
    elif cache is not None:
        device = cache.device
    elif stream is not None:
        device = stream.device
    else:
        device = torch.device("cpu")

    # Source-only expert ranks retain typed meta parameters. Their dtype still
    # governs empty participation; the execution stream supplies the device.
    representation = reference
    if representation is None:
        representation = next(module.parameters(), None)
    dtype = (
        representation.dtype if representation is not None else torch.float32
    )
    return device, dtype


def _layer_representation(module, inherited):
    """A layer's own floating-point placement, else its parent's."""
    for value in (
        *module.parameters(recurse=False),
        *module.buffers(recurse=False),
    ):
        if value.is_floating_point() and not value.is_meta:
            return value.device, value.dtype
    return inherited


def _devices(module: nn.Module, kind: str) -> set[torch.device]:
    return {
        value.device for value in getattr(module, kind)() if not value.is_meta
    }


def _order(producer: torch.cuda.Stream, consumer: torch.cuda.Stream) -> None:
    """Make ``consumer`` wait for work already enqueued on ``producer``."""
    if producer == consumer:
        return
    event = torch.cuda.Event()
    with torch.cuda.device(producer.device):
        event.record(producer)
    with torch.cuda.device(consumer.device):
        consumer.wait_event(event)


class _DeviceTransfers:
    """Delivery for ordinary modules placed on another device than their parent.

    Placement and streams belong to the execution context; forward hooks
    borrow them only during an activation, so shared modules keep
    independent invocations.
    """

    def __init__(self, module: nn.Module, device: torch.device) -> None:
        inherited = {"": device}
        self.modules: dict[int, tuple[nn.Module, torch.device]] = {}
        # (origin device, origin stream, destination device) -> stream.
        self.streams: dict[tuple[int, int, int], torch.cuda.Stream] = {}
        for path, child in module.named_modules(remove_duplicate=False):
            parent = inherited[path.rpartition(".")[0]] if path else device
            placements = _devices(child, "parameters") or _devices(
                child, "buffers"
            )
            target = next(iter(placements)) if len(placements) == 1 else parent
            if path and target != parent:
                self.modules[id(child)] = (child, target)
            inherited[path] = target

    def scope(self) -> _TransferScope | None:
        return _TransferScope(self) if self.modules else None

    def stream(
        self, origin: torch.cuda.Stream, device: torch.device
    ) -> torch.cuda.Stream:
        route = (origin.device.index, origin.cuda_stream, device.index)
        if route in self.streams:
            return self.streams[route]
        if origin.device.index == device.index:
            self.streams[route] = origin
            return origin
        # Capture can use another origin stream than eager warmup. Its target
        # keeps the same serialized delivery stream and allocation domain.
        stream = next(
            (
                stream
                for (_, _, destination), stream in self.streams.items()
                if destination == device.index
            ),
            None,
        )
        if stream is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm cross-device numerical calls before capture"
                )
            stream = torch.cuda.Stream(device)
        self.streams[route] = stream
        self.streams[
            (device.index, stream.cuda_stream, origin.device.index)
        ] = origin
        return stream

    def reset(self) -> None:
        self.streams.clear()

    def close(self) -> None:
        self.reset()
        self.modules.clear()


@dataclass
class _Call:
    module: int
    target: torch.device | None
    out: Any
    origin: torch.cuda.Stream | None = None
    stream: torch.cuda.Stream | None = None
    scopes: list = dataclasses.field(default_factory=list)

    def close(self) -> None:
        actions = []
        if self.origin is not None and self.stream is not None:
            origin, stream = self.origin, self.stream
            actions.append(lambda: _order(stream, origin))
        for scope in reversed(self.scopes):
            actions.append(functools.partial(scope.__exit__, None, None, None))
        close_resources(*actions)


# The activation whose transfer hooks own module calls. Hooks registered by
# other (nested or concurrent) activations of shared modules stay inert.
_ACTIVE_TRANSFERS: contextvars.ContextVar[_TransferScope | None] = (
    contextvars.ContextVar("uniserve_transfers", default=None)
)


class _TransferScope:
    """Forward hooks that move one activation's tensors across devices."""

    def __init__(self, owner: _DeviceTransfers) -> None:
        self.owner = owner
        self.token = None
        self.hooks: list = []
        self.calls: list[_Call] = []

    def __enter__(self) -> _TransferScope:
        self.token = _ACTIVE_TRANSFERS.set(self)
        try:
            for module, _ in self.owner.modules.values():
                self.hooks.append(
                    module.register_forward_pre_hook(
                        self.prepare, prepend=True, with_kwargs=True
                    )
                )
                self.hooks.append(
                    module.register_forward_hook(
                        self.finish, with_kwargs=True, always_call=True
                    )
                )
        except BaseException as error:
            try:
                self.__exit__(None, None, None)
            except BaseException as cleanup:
                error.add_note(f"transfer hook cleanup failed: {cleanup}")
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        calls, self.calls = self.calls, []
        hooks, self.hooks = self.hooks, []
        token, self.token = self.token, None
        actions = [call.close for call in reversed(calls)]
        actions += [hook.remove for hook in reversed(hooks)]
        if token is not None:
            actions.append(lambda: _ACTIVE_TRANSFERS.reset(token))
        close_resources(*actions)

    def _active(self) -> bool:
        return _ACTIVE_TRANSFERS.get() is self

    def prepare(self, module, args, kwargs):
        if not self._active():
            return None
        from uniserve_models.distributed.transfers import copy_inputs

        entry = self.owner.modules.get(id(module))
        if entry is None:
            raise RuntimeError("module device binding is closed")
        device = entry[1]
        call = _Call(id(module), None, kwargs.get("out"))
        self.calls.append(call)
        if device.type == "cuda":
            origin = torch.cuda.current_stream()
            stream = self.owner.stream(origin, device)
            call.origin, call.stream = origin, stream
            _order(origin, stream)
            for scope in (torch.cuda.device(device), torch.cuda.stream(stream)):
                scope.__enter__()
                call.scopes.append(scope)
        # Traversal returns the origin while moving the inputs, so discovery
        # does not reconstruct every dataclass and container twice.
        inputs, call.target = copy_inputs(args, kwargs, device)
        return inputs

    def finish(self, module, args, kwargs, result):
        if not self._active() or not self.calls:
            return None
        if self.calls[-1].module != id(module):
            return None
        from uniserve_models.distributed.transfers import copy_outputs

        call = self.calls.pop()
        try:
            copied = (
                result
                if call.target is None or result is None
                else copy_outputs(result, call.target, call.out)
            )
        except BaseException as error:
            try:
                call.close()
            except BaseException as cleanup:
                error.add_note(f"device transfer cleanup failed: {cleanup}")
            raise
        call.close()
        return copied


def _empty() -> Mapping:
    return MappingProxyType({})


class ExecutionContext:
    """Bind ordinary numerical modules to prepared operators and backing.

    Callers retire captured graphs and readers before preparing another size
    or closing. The stream, cache, experts, weights, an explicitly supplied
    scratch owner and a supplied workspace provider are borrowed; the context
    releases only its own bindings and the provider's persistent storage.

    Args:
        module: The root of the module tree to bind.
        stream: The execution stream: ``None`` (the caller's current stream),
            a ``torch.cuda.Stream``, or an integer ``CUstream`` handle owned
            by the caller (wrapped in ``torch.cuda.ExternalStream``).
        cache: The KV cache view object layers read by ``cache_name``.
        attention: Attention backend name or object.
        vsa: Sparse attention backend name.
        matmul: Matmul backend name.
        moe: Routed-expert backend name.
        groups: Communicators the bound modules use; defaults to every
            communicator in the module tree.
        scratch: A shared :class:`Scratch` (standalone workspace only).
        derive_host_lengths: Plans may read missing host lengths from the
            device.
        experts: The expert exchange routed experts communicate through.
        weights: Expert weight prefetch.
        workspace: Workspace provider; defaults to a standalone provider.
        collectives: Stream-bound collective bindings selected during an
            activation (the parallel runtime supplies them for its streams);
            ``None`` uses the process groups directly.
    """

    __class_getitem__ = classmethod(GenericAlias)

    def __init__(
        self,
        module: nn.Module,
        *,
        stream: torch.cuda.Stream | int | None = None,
        cache=None,
        attention="auto",
        vsa="auto",
        matmul="auto",
        moe="auto",
        groups=None,
        scratch: Scratch | None = None,
        derive_host_lengths: bool = True,
        experts=None,
        weights=None,
        workspace=None,
        collectives: Mapping | None = None,
    ) -> None:
        from uniserve_models.distributed import communicators

        self.module = module
        self.cache = cache
        self.experts = experts
        self.weights = weights
        self.groups = (
            tuple(groups) if groups is not None else communicators(module)
        )
        handle_stream = isinstance(stream, int)
        device, dtype = _representation(
            module, cache, None if handle_stream else stream
        )
        if handle_stream:
            stream = (
                torch.cuda.ExternalStream(stream, device=device)
                if stream
                else None
            )
        if stream is not None and stream.device != device:
            raise ValueError(
                "the execution stream must belong to the root module device"
            )
        self.stream = stream
        self._device, self._dtype = device, dtype
        self._backends = {
            "attention": attention,
            "vsa": vsa,
            "matmul": matmul,
            "moe": moe,
        }
        self.derive_host_lengths = derive_host_lengths
        self.collectives = collectives
        self.workspace = (
            StandaloneWorkspace(scratch) if workspace is None else workspace
        )
        self.constants: Mapping[str, torch.Tensor] = _empty()
        self.workspace_views: Mapping[str, torch.Tensor] = _empty()
        self.graph_streams: set[torch.cuda.Stream] = set()
        self._transfers = _DeviceTransfers(module, device)
        self._entered: contextlib.ExitStack | None = None
        self._closed = False
        self._max_tokens: int | None = None
        self._clear_bindings()

    # Binding tables an activation installs into context variables. They are
    # cleared in place, so an activation entered before ``prepare`` sees the
    # bindings that preparation creates.
    _BINDINGS = (
        "_operators",
        "_merged",
        "_attention",
        "_vsa",
        "_vsa_operators",
        "_moe",
        "_vsa_output",
        "_vsa_context",
        "_context_backing",
        "_vsa_transport",
        "_exchange",
        "_chunks",
        "_gather_pools",
    )

    def _clear_bindings(self) -> None:
        for name in self._BINDINGS:
            table = self.__dict__.get(name)
            if table is None:
                setattr(self, name, {})
            else:
                table.clear()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("execution context is closed")

    # Workspace requests made by bindings during preparation.

    def scratch(self, role, requirements, device):
        """Borrow one role's transient work area on this serialized stream."""
        self._require_open()
        return self.workspace.scratch(role, requirements, device)

    def _allocate(self, requirements, device):
        return self.workspace.allocate(requirements, device)

    def _matmul_workspace(self, requirements, device):
        return self.scratch("matmul", requirements, device)

    def _moe_workspace(self, requirements, device):
        return self.scratch("moe", requirements, device)

    def _vsa_buffers(self, slot, requirements, device):
        return self.scratch(("vsa", slot), requirements, device)

    def _attention_workspace(self, requirements, device):
        persistent = {
            name: config
            for name, config in requirements.items()
            if name != "scratch"
        }
        views = dict(self._allocate(persistent, device)) if persistent else {}
        if "scratch" in requirements:
            shared = self.scratch(
                "attention", {"scratch": requirements["scratch"]}, device
            )
            views["scratch"] = shared["scratch"]
        return views

    def _prepare_exchange(self, layer, tokens, device, dtype) -> None:
        size = layer.exchange.group.size
        if size == 1:
            return
        rows = -(-tokens // size) * size
        widths = (
            ("heads_send", layer.local_heads + 2 * layer.local_kv_heads),
            ("heads_receive", layer.local_heads + 2 * layer.local_kv_heads),
            ("output_receive", layer.local_heads),
        )
        need = {
            name: rows * heads * layer.head_dim * dtype.itemsize
            for name, heads in widths
        }
        previous = self._exchange.get(id(layer))
        if previous is not None and all(
            previous.tensors[name].numel() >= amount
            for name, amount in need.items()
        ):
            return
        # Flat byte fields carry Q/K/V heads out and local query heads back.
        from uniserve_models.bindings.attention import ExchangeBuffers

        views = self._allocate(
            {
                name: BufferConfig((amount,), torch.uint8)
                for name, amount in need.items()
            },
            device,
        )
        self._exchange[id(layer)] = ExchangeBuffers(views)

    def _bind_output(self, parallel, **options):
        """Back and bind one layer's Ulysses output storage."""
        from uniserve_models.nn.attention.storage import (
            bind_output_storage,
            output_requirements,
        )

        requirements = output_requirements((parallel,), **options)
        tensors, peers = {}, {}
        for key, requirement in requirements.items():
            tensors[key], fields = self.workspace.storage(requirement)
            peers[key] = fields.get("output", ())
        return bind_output_storage(requirements, tensors, peers)[parallel]

    def _bind_context(self, parallel, **options):
        """Back and bind one layer's context K/V storage, if it needs any."""
        from uniserve_models.nn.attention.storage import (
            bind_context_storage,
            context_requirements,
        )

        requirements = context_requirements((parallel,), **options)
        if not requirements:
            return None
        tensors = {
            key: self.workspace.storage(requirement)[0]
            for key, requirement in requirements.items()
        }
        return bind_context_storage(requirements, tensors)[parallel]

    def _context_transport(self, layer, rows, dtype):
        parallel = layer.context_parallel
        group = parallel.context_group
        context_size = group.size
        exchange_size = layer.exchange.group.size
        if self._max_tokens is not None:
            rows = max(
                rows,
                -(-self._max_tokens // (context_size * exchange_size))
                * exchange_size,
            )
        self._prepare_exchange(layer, rows * context_size, group.device, dtype)
        if context_size == 1 or rows == 0:
            return None
        heads, head_dim = layer.local_kv_heads, layer.head_dim
        key = (parallel.key_group, rows, heads, head_dim, dtype)
        buffers = self._context_backing.get(key)
        if buffers is None:
            buffers = self._bind_context(
                parallel,
                rows=rows,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
                block_size=1,
            )
            self._context_backing[key] = buffers
        self._vsa_context[parallel] = buffers
        return buffers

    def _vsa_exchange(self, layer, rows, heads, head_dim, dtype):
        parallel = getattr(layer, "parallel", None)
        if parallel is None:
            return None
        ulysses, context_group = parallel.ulysses_group, parallel.context_group
        groups = (ulysses, context_group, parallel.key_group)
        dimensions = (heads, head_dim, dtype)
        key = next(
            (
                key
                for key in self._vsa_transport
                if key[:3] == groups
                and key[3] >= rows
                and key[4:] == dimensions
            ),
            (*groups, rows, *dimensions),
        )
        if key not in self._vsa_transport:
            if capturing(self._device):
                raise RuntimeError(
                    "prepare VSA communication storage before capture"
                )
            options = {
                "rows": rows,
                "heads": heads,
                "head_dim": head_dim,
                "dtype": dtype,
            }
            # Registered symmetric windows belong to stream-bound
            # communicators, which the parallel runtime supplies; independent
            # buffers use the ordinary collective path.
            output = self._bind_output(parallel, **options)
            context = self._bind_context(parallel, **options, block_size=64)
            self._vsa_transport[key] = (output, context)
        output, context = self._vsa_transport[key]
        self._vsa_output[parallel] = output
        if context is not None:
            self._vsa_context[parallel] = context
        return parallel

    # Preparation.

    def prepare(
        self,
        size,
        *,
        constants: TensorBuffers | None = None,
        workspace: TensorBuffers | None = None,
    ) -> None:
        """Bind every call site for ``size``, replacing previous capacity.

        Supplied ``constants`` and ``workspace`` buffers remain borrowed and
        are never closed by this owner.
        """
        self._require_open()
        self._release()
        try:
            self._prepare(size, constants, workspace)
        except BaseException as error:
            try:
                self._release()
            except BaseException as cleanup:
                error.add_note(
                    f"execution preparation cleanup failed: {cleanup}"
                )
            raise

    def _prepare(self, size, constants, workspace) -> None:
        from uniserve_models.bindings.attention import AttentionBinding
        from uniserve_models.bindings.matmul import MatmulBinding
        from uniserve_models.bindings.moe import MoEBinding
        from uniserve_models.bindings.vsa import VsaBinding
        from uniserve_models.model.inputs import TextSize
        from uniserve_models.nn.attention import Attention
        from uniserve_models.nn.attention.vsa import BlockAttention
        from uniserve_models.nn.linear import (
            ColumnParallelLinear,
            Linear,
            MergedColumnParallelLinear,
        )
        from uniserve_models.nn.moe import FusedMoE

        text = isinstance(size, TextSize)
        max_rows = size.num_tokens if text else None
        self._max_tokens = max_rows
        module = self.module
        with self.activate():
            for name, query, supplied in (
                ("constants", "constant_buffers", constants),
                ("workspace", "workspace_buffers", workspace),
            ):
                requirements = (
                    getattr(module, query)(size)
                    if hasattr(module, query)
                    else {}
                )
                if supplied is not None:
                    views = supplied.view(requirements)
                elif requirements:
                    views = self._allocate(requirements, self._device)
                else:
                    views = _empty()
                if name == "constants":
                    self.constants = views
                else:
                    self.workspace_views = views
            if hasattr(module, "prepare_constants"):
                module.prepare_constants(size, out=self.constants)

            representations = {"": (self._device, self._dtype)}
            vsa_slot = 0
            for path, child in module.named_modules():
                parent = path.rpartition(".")[0]
                device, dtype = _layer_representation(
                    child, representations[parent]
                )
                representations[path] = (device, dtype)

                merged = isinstance(child, MergedColumnParallelLinear)
                if merged or isinstance(child, Linear):
                    self._bind_matmul(
                        MatmulBinding, child, dtype, max_rows, merged
                    )
                if isinstance(child, ColumnParallelLinear):
                    self._bind_gather(child, device, max_rows)
                if isinstance(child, FusedMoE):
                    binding = MoEBinding(
                        child,
                        self._backends["moe"],
                        size if text else None,
                        device,
                        self._moe_workspace,
                        self.experts,
                        self.weights,
                    )
                    self._moe[id(child)] = binding
                    if text:
                        binding.prepare(size)
                if isinstance(child, BlockAttention):
                    # Alternate projection buffers so output consumption can
                    # overlap the next layer's input preparation.
                    self._vsa[id(child)] = VsaBinding(
                        self._backends["vsa"],
                        self.scratch,
                        self._attention_workspace,
                        functools.partial(self._vsa_buffers, vsa_slot % 2),
                        self._vsa_exchange,
                        self._vsa_operators,
                    )
                    vsa_slot += 1
                if isinstance(child, Attention):
                    self._bind_attention(
                        AttentionBinding,
                        child,
                        device,
                        dtype,
                        size if text else None,
                    )

    def _bind_matmul(self, factory, layer, dtype, rows, merged) -> None:
        bindings = self._merged if merged else self._operators
        binding = factory(
            layer, self._backends["matmul"], rows, self._matmul_workspace
        )
        bindings[id(layer)] = binding
        if merged:
            weights = []
            quantizers = []
            for name, branch in layer.projections.items():
                weights.append((name, id(branch.weight)))
                quantizers.append(branch.input_quantizer)
            bindings[(tuple(weights), layer.branch_width)] = binding
        else:
            bindings[id(layer.weight)] = binding
            quantizers = [layer.input_quantizer]
        if rows is not None and len(set(map(id, quantizers))) == 1:
            binding._prepare(dtype, dtype, quantizers[0], rows)

    def _bind_gather(self, layer, device, rows) -> None:
        distribution = layer.input_distribution
        axes = getattr(layer, "gather_axes", None)
        if axes is None:
            axes = distribution.shard_axes(0)
        group = distribution.mesh.get_group(axes)
        if group.size == 1:
            return
        # Two complete gather slots cover all chunks. FP32 capacity also fits
        # encoded values and their scales, independent of the kernel.
        width = layer.weight.shape[1]
        capacity = (
            None
            if rows is None
            else 2 * (-(-rows // group.size)) * group.size * width * 4
        )
        # Stream-bound pools arrive with the parallel runtime's stream
        # communicators; this context keeps one standalone pool per group.
        pool = self._gather_pools.get(group)
        if pool is None:
            pool = GatherPool(group)
            self._gather_pools[group] = pool
        self._chunks[id(layer)] = functools.partial(
            pool.borrow, capacity=capacity
        )
        if capacity and not isinstance(self.workspace, DeclaredWorkspace):
            with pool.borrow(capacity, device):
                pass

    def _bind_attention(self, factory, layer, device, dtype, size) -> None:
        name = layer.cache_name
        state = table = None
        if name is not None and self.cache is not None:
            state = self.cache.state(name)
            table = self.cache.table(name)
            device, dtype = state.key.device, state.key.dtype
        binding = factory(
            layer,
            self._backends["attention"],
            state,
            size,
            device,
            dtype,
            self._attention_workspace,
            self._context_transport,
            table,
            derive_host_lengths=self.derive_host_lengths,
        )
        self._attention[id(layer)] = binding
        if size is not None:
            binding.prepare(dtype, size)
            self._prepare_exchange(layer, size.num_tokens, device, dtype)

    # Calls.

    def bind_attention(self, batch, *, replay: bool = False) -> None:
        """Bind only layers reading this batch's cache tables.

        Missing host lengths are derived once when enabled. Replay binds one
        reader per table when launches consume device metadata.
        """
        from uniserve_models.backends.attention._sequences import (
            batch_host_lengths,
        )

        self._require_open()
        with self.activate():
            entries = batch.entries
            readers = [
                binding
                for binding in self._attention.values()
                if (
                    len(entries) == 1
                    if binding.table is None
                    else binding.table in entries
                )
            ]
            if replay and not any(
                reader.builds_launch_plan for reader in readers
            ):
                # Device-driven launches share one numerical input per table;
                # host-planned launches bind every reading layer.
                first = {}
                for reader in readers:
                    first.setdefault(reader.table, reader)
                readers = list(first.values())
            lengths = any(
                reader.reads_host_lengths(batch.entry(reader.table))
                for reader in readers
            )
            mirrored = (
                batch_host_lengths(batch, derive=self.derive_host_lengths)
                if lengths
                else batch
            )
            for reader in readers:
                reader.bind(mirrored, source=batch)

    def join_expert_layers(self) -> None:
        """Complete the open expert step at layers this forward skipped."""
        exchange = self.experts
        if exchange is None or not exchange.capacity:
            return
        modules = [
            module
            for module, binding in self._moe.items()
            if binding.exchange is exchange
        ]
        for module in exchange.pending_layers(modules):
            binding = self._moe[module]
            binding.join(binding.module.hidden_size, self._dtype)

    def kernels(self) -> list[dict]:
        """Selected kernels by module path; empty once closed."""
        from uniserve_models.nn.linear import MergedColumnParallelLinear

        records = []
        if self._closed:
            return records
        for path, child in self.module.named_modules():
            matmul = (
                self._merged
                if isinstance(child, MergedColumnParallelLinear)
                else self._operators
            )
            for bindings in (matmul, self._moe, self._vsa, self._attention):
                binding = bindings.get(id(child))
                if binding is not None:
                    for record in binding.kernels():
                        records.append({"path": path, **record})
        return records

    @contextlib.contextmanager
    def activate(self) -> Iterator[ExecutionContext]:
        """Enter the bindings, transfers and stream for one call."""
        from uniserve_models.nn import _binding
        from uniserve_models.nn.attention._parallel import (
            context_scope,
            output_scope,
        )
        from uniserve_models.runtime.communication import (
            stream_collective_scope,
        )

        self._require_open()
        with contextlib.ExitStack() as stack:
            if self.stream is not None:
                stack.enter_context(torch.cuda.device(self.stream.device))
                stack.enter_context(torch.cuda.stream(self.stream))
            stack.enter_context(stream_collective_scope(self.collectives))
            transfers = self._transfers.scope()
            if transfers is not None:
                stack.enter_context(transfers)
            stack.enter_context(output_scope(self._vsa_output))
            stack.enter_context(context_scope(self._vsa_context))
            for variable, values in (
                (_binding.matmul, self._operators),
                (_binding.merged_matmul, self._merged),
                (_binding.attention, self._attention),
                (_binding.vsa, self._vsa),
                (_binding.moe, self._moe),
                (_binding.attention_storage, self._exchange),
                (_binding.linear_chunks, self._chunks),
            ):
                stack.callback(variable.reset, variable.set(values))
            if self.weights is not None:
                stack.enter_context(self.weights.activate())
            yield self

    def idle(self) -> bool:
        """Report whether every stream this context used has drained."""
        streams = list(self._transfers.streams.values())
        streams.extend(self.graph_streams)
        if self.stream is not None:
            streams.append(self.stream)
        elif self._device.type == "cuda":
            streams.append(torch.cuda.current_stream(self._device))
        return streams_idle(streams)

    # Release.

    def _release(self) -> None:
        """Close bindings and persistent workspace, keeping shared scratch."""
        actions = [binding.close for binding in self._attention.values()]
        actions += [binding.close for binding in self._vsa.values()]
        actions += [pair[1].close for pair in self._vsa_operators.values()]
        actions += [binding.close for binding in self._moe.values()]
        actions.append(self.workspace.release)
        try:
            close_resources(*actions)
        finally:
            self._transfers.reset()
            self._clear_bindings()
            self.constants = _empty()
            self.workspace_views = _empty()
            self._max_tokens = None

    def close(self, *, aborted: bool = False) -> None:
        """Release owned resources after their readers finish.

        An aborted close retains every resource until the process exits
        without waiting for a possibly failed peer or device.
        """
        if self._closed:
            return
        self._closed = True
        if aborted:
            _retained.append(self.__dict__.copy())
            return
        try:
            self._release()
        finally:
            self._transfers.close()
            self.workspace.close()

    def __enter__(self) -> ExecutionContext:
        if self._entered is not None:
            raise RuntimeError(
                "execution context ownership cannot be entered twice"
            )
        stack = contextlib.ExitStack()
        stack.enter_context(self.activate())
        self._entered = stack
        return self

    def __exit__(self, kind, error, traceback) -> None:
        stack, self._entered = self._entered, None
        if stack is None:
            raise RuntimeError("execution context ownership was not entered")
        stack.__exit__(kind, error, traceback)
        aborted = error is not None and not self.idle()
        try:
            self.close(aborted=aborted)
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note(f"execution context cleanup failed: {cleanup}")
