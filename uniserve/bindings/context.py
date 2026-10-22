"""Execution contexts: the numerical bindings of a module tree.

An :class:`ExecutionContext` prepares one module tree for a maximum size: it
walks the tree, creates a binding (prepared operator, plan and workspace) for
every numerical call site, and installs those bindings into the
``uniserve.nn._binding`` context variables while a call runs. Workspace
backing comes from a :class:`uniserve.bindings.workspace.Workspace`.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import GenericAlias, MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from uniserve.bindings import capturing
from uniserve.bindings.workspace import (
    GatherPool,
    Scratch,
    TensorBuffers,
    Workspace,
    close_resources,
)
from uniserve.prefix_cache import CacheBinding
from uniserve.tensors import BufferConfig

if TYPE_CHECKING:
    # The binding and layer modules import this one; their names are needed
    # only to annotate the tables an ExecutionContext keeps.
    from uniserve.bindings.attention import (
        AttentionBinding,
        ExchangeBuffers,
    )
    from uniserve.bindings.matmul import MatmulBinding
    from uniserve.bindings.moe import MoEBinding
    from uniserve.bindings.vsa import VsaBinding
    from uniserve.nn._binding import ChunkStorage, MergedKey
    from uniserve.nn.attention._parallel import (
        AttentionBuffers,
        ParallelAttention,
    )

# Owners whose asynchronous accesses could not be proven finished. An aborted
# context retains them until the process exits instead of releasing storage
# that device work may still address.
_retained: list[object] = []


def streams_idle(streams: Iterable[torch.cuda.Stream]) -> bool:
    """Report, without waiting, whether every stream finished its work.

    A device fault reported by the query leaves completion unproven, so it
    counts as unfinished work.
    """
    try:
        return all(stream.query() for stream in streams)
    except RuntimeError:
        return False


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
        self.token: contextvars.Token[_TransferScope | None] | None = None
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
        from uniserve.distributed.transfers import copy_inputs

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
        from uniserve.distributed.transfers import copy_outputs

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


class _Variables:
    """Context variables set to given values until exit restores them."""

    __slots__ = ("_tokens", "_values")

    def __init__(
        self, values: tuple[tuple[contextvars.ContextVar, Any], ...]
    ) -> None:
        self._values = values
        self._tokens: list[contextvars.Token] = []

    def __enter__(self) -> None:
        tokens = self._tokens
        for variable, value in self._values:
            tokens.append(variable.set(value))

    def __exit__(self, *exc_info: object) -> None:
        tokens, self._tokens = self._tokens, []
        for token in reversed(tokens):
            token.var.reset(token)


class _Ordering:
    """On exit, order ``consumer`` after the work enqueued on ``producer``."""

    __slots__ = ("_consumer", "_producer")

    def __init__(
        self, producer: torch.cuda.Stream, consumer: torch.cuda.Stream
    ) -> None:
        self._producer, self._consumer = producer, consumer

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        _order(self._producer, self._consumer)


class _Activation:
    """One call's entry into an :class:`ExecutionContext`.

    Entering selects the context's device and stream (and, for a module
    tree placed on another device, its delivery stream there), sets the
    context variables to the context's binding tables and opens its device
    transfers and weight prefetch; exiting closes each in reverse order,
    also when the call raised, and raises the first failure among them.
    Which variables and tables these are is resolved when the context is
    constructed, so an activation does only this per-call work.
    """

    __slots__ = ("_context", "_entered")

    def __init__(self, context: ExecutionContext) -> None:
        self._context = context
        # Context managers entered so far, exited in reverse.
        self._entered: list[contextlib.AbstractContextManager] = []

    def _enter(self, manager: contextlib.AbstractContextManager) -> None:
        manager.__enter__()
        self._entered.append(manager)

    def __enter__(self) -> ExecutionContext:
        context = self._context
        context._require_open()
        try:
            stream = context.stream
            if stream is not None:
                self._enter(torch.cuda.device(stream.device))
                self._enter(torch.cuda.stream(stream))
                if context.device != stream.device:
                    # A module placed on another device computes on its
                    # delivery stream there, between the context's stream's
                    # prior work and its next.
                    delivery = context._transfers.stream(stream, context.device)
                    _order(stream, delivery)
                    self._enter(_Ordering(delivery, stream))
                    self._enter(torch.cuda.device(context.device))
                    self._enter(torch.cuda.stream(delivery))
            self._enter(_Variables(context._variables))
            transfers = context._transfers.scope()
            if transfers is not None:
                self._enter(transfers)
            if context.weights is not None:
                self._enter(context.weights.activate())
        except BaseException as error:
            self._exit(type(error), error, error.__traceback__)
            raise
        return context

    def __exit__(self, kind, error, traceback) -> None:
        self._exit(kind, error, traceback)

    def _exit(self, kind, error, traceback) -> None:
        entered, self._entered = self._entered, []
        failure: BaseException | None = None
        for manager in reversed(entered):
            try:
                manager.__exit__(kind, error, traceback)
            except BaseException as caught:
                if failure is None:
                    failure = caught
                else:
                    failure.add_note(
                        f"Activation cleanup also failed: {caught!r}"
                    )
        if failure is not None:
            raise failure


class ExecutionContext:
    """Bind ordinary numerical modules to prepared operators and backing.

    Callers retire captured graphs and readers before preparing another size
    or closing. The stream, cache, experts, weights, an explicitly supplied
    scratch owner and a supplied workspace provider are borrowed; the context
    releases only its own bindings and the provider's persistent storage.

    Args:
        module: The root of the module tree to bind.
        device: The device of the context's stream and of the tensors its
            callers exchange; ``None`` takes the root module's. A module tree
            placed wholly on one other device (a generation tower's codec)
            computes there: an activation runs it on a delivery stream of
            that device ordered after the context's stream, and the stream
            waits for it on exit. Within the tree, submodules placed on
            other devices than their parent run there through transfers
            that move their inputs there and their outputs back.
        stream: The execution stream: ``None`` (the caller's current stream),
            a ``torch.cuda.Stream``, or an integer ``CUstream`` handle owned
            by the caller (wrapped in ``torch.cuda.ExternalStream``).
        cache: The prefix cache view object layers read by ``cache_name``.
        attention: Attention backend name or object.
        vsa: Sparse attention backend name.
        matmul: Matmul backend name.
        moe: Routed-expert backend name.
        groups: Communicators the bound modules use; defaults to every
            communicator in the module tree.
        scratch: A shared :class:`Scratch` (default workspace only).
        derive_host_lengths: Plans may read missing host lengths from the
            device.
        experts: The expert exchange routed experts communicate through.
        weights: Expert weight prefetch.
        workspace: Workspace provider; defaults to a :class:`Workspace`.
        collectives: Stream-bound collective bindings selected during an
            activation (the parallel runtime supplies them for its streams);
            ``None`` uses the process groups directly.
        tiles: ``TiledEncoder`` modules mapped to borrowed operators, each
            returning exactly that encoder's encoding of any one tile, such
            as the replay of a graph captured at the tile shape; the
            encoders' tiled calls within an activation encode their tiles
            through them. Their owner keeps them valid for the context's
            lifetime.
    """

    __class_getitem__: classmethod[Any, [Any], GenericAlias] = classmethod(
        GenericAlias
    )

    def __init__(
        self,
        module: nn.Module,
        *,
        device: torch.device | str | None = None,
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
        tiles: Mapping[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
    ) -> None:
        from uniserve.distributed import communicators

        self.module = module
        self.cache = cache
        self.experts = experts
        self.weights = weights
        self.groups = (
            tuple(groups) if groups is not None else communicators(module)
        )
        # An integer handle becomes a stream once the device is known.
        bound = None if isinstance(stream, int) else stream
        root, dtype = _representation(module, cache, bound)
        device = root if device is None else torch.device(device)
        if isinstance(stream, int) and stream:
            bound = torch.cuda.ExternalStream(stream, device=device)
        if bound is not None and bound.device != device:
            raise ValueError(
                "the execution stream must belong to the context's device"
            )
        # The module computes on the one device holding all its parameters
        # (or, without any, buffers), else on the context's device.
        placements = _devices(module, "parameters") or _devices(
            module, "buffers"
        )
        placement = next(iter(placements)) if len(placements) == 1 else device
        self.stream: torch.cuda.Stream | None = bound
        self._device, self._dtype = placement, dtype
        self._backends = {
            "attention": attention,
            "vsa": vsa,
            "matmul": matmul,
            "moe": moe,
        }
        self.derive_host_lengths = derive_host_lengths
        self.collectives = collectives
        self.workspace = Workspace(scratch) if workspace is None else workspace
        self.constants: Mapping[str, torch.Tensor] = _empty()
        self.workspace_views: Mapping[str, torch.Tensor] = _empty()
        self.graph_streams: set[torch.cuda.Stream] = set()
        self._transfers = _DeviceTransfers(module, placement)
        self._entered: contextlib.ExitStack | None = None
        self._closed = False
        self._max_tokens: int | None = None
        self._clear_bindings()
        # Borrowed tile operators by encoder id, the key the encoders read.
        self._tiles = {
            id(encoder): operator for encoder, operator in (tiles or {}).items()
        }

        # The context variables an activation sets and the values it sets
        # them to. The tables are cleared in place, never replaced, so the
        # pairs hold for the context's lifetime.
        from uniserve.distributed.collectives import stream_bindings
        from uniserve.nn import _binding
        from uniserve.nn.attention._parallel import context_bindings

        self._variables: tuple[tuple[contextvars.ContextVar, Any], ...] = (
            (stream_bindings, self.collectives),
            (context_bindings, self._vsa_context),
            (_binding.matmul, self._operators),
            (_binding.merged_matmul, self._merged),
            (_binding.attention, self._attention),
            (_binding.vsa, self._vsa),
            (_binding.moe, self._moe),
            (_binding.attention_storage, self._exchange),
            (_binding.linear_chunks, self._chunks),
            (_binding.tile_operators, self._tiles),
            (_binding.cache, self._cache),
        )

    # Binding tables an activation installs into context variables. They are
    # cleared in place, so an activation entered before ``prepare`` sees the
    # bindings that preparation creates.
    _BINDINGS = (
        "_operators",
        "_merged",
        "_attention",
        "_attention_plans",
        "_vsa",
        "_vsa_operators",
        "_moe",
        "_vsa_context",
        "_context_backing",
        "_vsa_storage",
        "_exchange",
        "_chunks",
        "_gather_pools",
        "_cache",
    )

    # weight or module id -> prepared matmul operator
    _operators: dict[int, MatmulBinding]
    # module id, or branch key -> prepared merged operator
    _merged: dict[int | MergedKey, MatmulBinding]
    _attention: dict[int, AttentionBinding]
    # Launch plans every attention call site shares, keyed by the provider
    # (layer shape and batch signature); the layers of one forward bind the
    # same batch, so a host-planning provider plans it once.
    _attention_plans: dict[tuple, Any]
    _vsa: dict[int, VsaBinding]
    # (device, dtype, heads, head_dim, pattern) -> (provider name, operator)
    _vsa_operators: dict[tuple, tuple[str, Any]]
    _moe: dict[int, MoEBinding]
    _vsa_context: dict[ParallelAttention, AttentionBuffers]
    # (context group, rows, heads, head_dim, dtype) -> context K/V buffers
    _context_backing: dict[tuple, AttentionBuffers]
    # (context group, rows, heads, head_dim, dtype) -> the gathered K/V of a
    # context-partitioned sparse attention layer
    _vsa_storage: dict[tuple, AttentionBuffers]
    # module id -> attention exchange storage
    _exchange: dict[int, ExchangeBuffers]
    # module id -> gather storage for chunked token exchange
    _chunks: dict[int, ChunkStorage]
    # communication group -> its standalone gather pool
    _gather_pools: dict[Any, GatherPool]
    # module id -> the cache layer a module other than attention owns
    _cache: dict[int, CacheBinding]

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
        """Borrow one role's transient work area on this serialized stream.

        Modules placed on another device than the context's (reached
        through transfers) take that device's area of the role, a field of
        its own name.
        """
        self._require_open()
        if torch.device(device) != self._device:
            parts = role if isinstance(role, tuple) else (role,)
            role = (*parts, str(torch.device(device)))
        return self.workspace.scratch(role, requirements, device)

    def _allocate(self, requirements, device):
        return self.workspace.allocate(requirements, device)

    def _matmul_workspace(self, requirements, device):
        return self.scratch("matmul", requirements, device)

    def _moe_workspace(self, requirements, device):
        return self.scratch("moe", requirements, device)

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
        from uniserve.bindings.attention import ExchangeBuffers

        views = self._allocate(
            {
                name: BufferConfig((amount,), torch.uint8)
                for name, amount in need.items()
            },
            device,
        )
        self._exchange[id(layer)] = ExchangeBuffers(views)

    def _bind_context(self, parallel, **options):
        """Back and bind one layer's context K/V storage, if it needs any."""
        from uniserve.nn.attention.storage import (
            bind_context_storage,
            context_requirements,
        )

        requirements = context_requirements((parallel,), **options)
        if not requirements:
            return None
        tensors = {
            key: self.workspace.storage(requirement)
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
        key = (group, rows, heads, head_dim, dtype)
        buffers = self._context_backing.get(key)
        if buffers is None:
            buffers = self._bind_context(
                parallel,
                rows=rows,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
            )
            self._context_backing[key] = buffers
        self._vsa_context[parallel] = buffers
        return buffers

    def _vsa_exchange(self, layer, rows, heads, head_dim, dtype):
        """Bind the gathered K/V storage of a context-partitioned VSA layer.

        ``rows`` are the layer's rows of one context interval. Context
        storage serves every row count up to its own (``ParallelAttention``
        borrows compact leading views), so storage prepared for more rows
        serves this call too.

        Raises:
            RuntimeError: The storage would first be created during graph
                capture.
        """
        parallel = layer.context_parallel
        group = parallel.context_group
        dimensions = (heads, head_dim, dtype)
        key = next(
            (
                key
                for key in self._vsa_storage
                if key[0] == group and key[1] >= rows and key[2:] == dimensions
            ),
            (group, rows, *dimensions),
        )
        if key not in self._vsa_storage:
            if capturing(self._device):
                raise RuntimeError(
                    "prepare VSA communication storage before capture"
                )
            buffers = self._bind_context(
                parallel,
                rows=key[1],
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
            )
            assert buffers is not None
            self._vsa_storage[key] = buffers
        self._vsa_context[parallel] = self._vsa_storage[key]

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
        from uniserve.bindings.attention import AttentionBinding
        from uniserve.bindings.matmul import MatmulBinding
        from uniserve.bindings.moe import MoEBinding
        from uniserve.bindings.vsa import VsaBinding
        from uniserve.models.inputs import TextSize
        from uniserve.nn.attention import Attention
        from uniserve.nn.attention.vsa import BlockAttention
        from uniserve.nn.linear import (
            ColumnParallelLinear,
            Linear,
            MergedColumnParallelLinear,
        )
        from uniserve.nn.moe import FusedMoE

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
                # An optional hook: ``nn.Module`` declares no such method.
                getattr(module, "prepare_constants")(size, out=self.constants)

            representations = {"": (self._device, self._dtype)}
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
                if merged and child.exchange is not None:
                    self._bind_exchange(child, device, dtype, max_rows)
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
                        ordinal=sum(
                            other.exchange is not None
                            for other in self._moe.values()
                        ),
                    )
                    self._moe[id(child)] = binding
                    if text:
                        binding.prepare(size)
                if isinstance(child, BlockAttention):
                    self._vsa[id(child)] = VsaBinding(
                        self._backends["vsa"],
                        self.scratch,
                        self._attention_workspace,
                        self._vsa_exchange,
                        self._vsa_operators,
                    )
                if isinstance(child, Attention):
                    self._bind_attention(
                        AttentionBinding,
                        child,
                        device,
                        dtype,
                        size if text else None,
                    )
                elif (
                    getattr(child, "cache_name", None) is not None
                    and hasattr(child, "cache_state")
                    and self.cache is not None
                ):
                    # Another cache layer type's owner reads its state and
                    # block table through ``prefix_cache.bound``.
                    name = child.cache_name
                    paged = self.cache.paged(name)
                    self._cache[id(child)] = CacheBinding(
                        self.cache.state(name),
                        self.cache.table(name) if paged else None,
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
            self._merged[(tuple(weights), layer.branch_width)] = binding
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
        if capacity:
            with pool.borrow(capacity, device):
                pass

    def _bind_exchange(self, layer, device, dtype, rows) -> None:
        """Reserve an exchanged projection's step backing for ``rows`` tokens.

        An exchange projects each step's rows for every member's heads into
        backing that stays readable until the step's transfer is joined: one
        step slot when the whole shard is one step, two when a transfer
        overlaps the next step, and for encoded inputs a whole-weight
        projection region per slot before its member split. The backing
        comes from the exchange group's gather pool, so captured graphs
        allocate nothing per step.
        """
        exchange = layer.exchange
        group = exchange.group
        amount = None
        if rows is not None:
            capacity = -(-rows // group.size)
            step = (
                capacity
                if exchange.rows is None
                else min(exchange.rows, capacity)
            )
            branches = tuple(layer.projections.values())
            width = sum(int(branch.weight.shape[0]) for branch in branches)
            encoded = any(
                branch.input_quantizer is not None for branch in branches
            )
            slots = 1 if exchange.rows is None else 2
            amount = slots * (1 + encoded) * step * width * dtype.itemsize
        pool = self._gather_pools.get(group)
        if pool is None:
            pool = GatherPool(group)
            self._gather_pools[group] = pool
        self._chunks[id(layer)] = functools.partial(
            pool.borrow, capacity=amount
        )
        if amount:
            with pool.borrow(amount, device):
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
            plans=self._attention_plans,
        )
        self._attention[id(layer)] = binding
        if size is not None:
            binding.prepare(dtype, size)
            self._prepare_exchange(layer, size.num_tokens, device, dtype)

    # Calls.

    def plans_attention(self) -> bool:
        """Whether an attention layer plans each launch on the host.

        A captured forward of such a context reads plans that
        :meth:`bind_attention` builds before every launch; without one, the
        launch reads only device metadata.
        """
        return any(
            binding.builds_launch_plan for binding in self._attention.values()
        )

    def bind_attention(self, batch, *, replay: bool = False) -> None:
        """Bind only layers reading this batch's cache tables.

        Missing host lengths are derived once when enabled. Replay binds one
        reader per table when launches consume device metadata.
        """
        from uniserve.backends.attention._sequences import (
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
                first: dict[Any, AttentionBinding] = {}
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

    @contextlib.contextmanager
    def expert_step(
        self, capacity: int, *, join: bool = True, reached: int = 0
    ) -> Iterator[None]:
        """Run the enclosed forward as one expert step of ``capacity``.

        ``capacity`` is the step's per-rank token capacity the expert group
        agreed on; zero runs the forward without an expert step (a context
        without expert-parallel layers, or a call outside one). When the
        forward returns, the expert layers it did not reach are joined with
        no tokens, so every rank exchanges at every layer of the step.

        A step may span two calls of this context: the first passes
        ``join=False``, leaving its tail to the second, which declares the
        first ``reached`` expert layers (in collective order) done.
        """
        from uniserve.bindings.moe import ExpertStep

        if not capacity:
            yield
            return
        if self.experts is None:
            raise ValueError("an expert step requires an expert exchange")
        layers = sorted(
            (
                binding
                for binding in self._moe.values()
                if binding.exchange is not None
            ),
            key=lambda binding: binding.ordinal,
        )
        with ExpertStep(
            self.experts, layers, int(capacity), self._dtype, reached=reached
        ) as step:
            yield
            if join:
                step.join_tail()

    def expert_layers(self, module: nn.Module) -> int:
        """Expert-parallel call sites of this context inside ``module``."""
        inside = {id(child) for child in module.modules()}
        return sum(
            binding.exchange is not None and key in inside
            for key, binding in self._moe.items()
        )

    def kernels(self) -> list[dict]:
        """Selected kernels by module path; empty once closed."""
        from uniserve.nn.linear import MergedColumnParallelLinear

        records: list[dict] = []
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

    def activate(self) -> _Activation:
        """Enter the bindings, transfers and stream for one call.

        Use as ``with context.activate():``; entering returns the context.
        """
        return _Activation(self)

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
        # The tile operators are borrowed; their owner releases them.
        self._tiles.clear()
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
