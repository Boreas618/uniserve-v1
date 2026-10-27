"""Distributed routed experts: expert exchanges and distributed weights.

The executor's expert union (``uniserve.executor.experts.ExpertUnion``) owns
the expert step protocol: agreement on each step's per-rank token capacity
over the worker's host group, microbatch threads and their rotation, join
graphs and close ordering. It uses these functions to construct and drive
the numerical side:

* ``layers(model, disaggregated)`` describes the expert layers that
  exchange tokens, in collective (traversal) order.
* ``create_exchange(...)`` constructs one microbatch's expert exchange
  (``uniserve.backends.experts``), collectively over the expert
  union, bootstrapped with the caller's host functions.
* ``bind_microbatches(exchanges)`` binds the microbatches' shared plan,
  when the exchange has one.
* ``symmetric_bytes(...)`` sizes the peer-mapped storage of split MegaMoE,
  which the caller allocates and maps.
* ``join(owner, capacity)`` runs one expert step of ``capacity`` in which
  this rank joins every expert layer of the owner's execution context with
  no tokens of its own; :class:`ExpertLayers` is the owner on an expert-only
  rank.
* ``warm_thread(device)`` prepares a microbatch thread's thread-local
  numerical state (cuBLAS handle) before graph capture.

An exchange's ``close``, ``suspend``, ``restore``, ``release`` and
``rebuild`` (``uniserve.backends.experts.ExpertTransport``) are called on it
directly; ``close`` is collective for exchanges with device communicators.

Distributed weight data parallelism (DWDP) exchanges weights instead of
tokens: ``weight_fields(model)`` places each expert layer's weights for its
kernels and returns this rank's shards, the caller publishes them and maps
every field's full extent, ``bind_weights`` installs the full views as the
layer's parameters, and ``bind_prefetch`` takes the caller's submission of
each layer's peer reads and returns the :class:`WeightPrefetch` schedule a
prepared component's expert calls prefetch peers' weights through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn

from uniserve.backends import experts as transports
from uniserve.cuda_graph import break_graph
from uniserve.distributed import Communicator
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import QuantizedTensor
from uniserve.weight_prefetch import SlotCopies


def _module(model) -> nn.Module:
    """The loaded module tree of a ``model.load`` handle."""
    return getattr(model, "module", model)


def _layers(model, disaggregated: bool) -> list[FusedMoE]:
    """Routed-expert layers that exchange tokens, in traversal order.

    With dedicated expert ranks every routed-expert layer exchanges;
    colocated, the layers whose experts are partitioned across ranks that
    hold different tokens, not across a tensor-parallel group.
    """
    return [
        module
        for module in _module(model).modules()
        if isinstance(module, FusedMoE)
        and (
            disaggregated
            or (
                module.expert_group.size > 1
                and module.expert_group is not module.group
            )
        )
    ]


def _encoding(layer: FusedMoE) -> str:
    quantizer = layer.up_gate.input_quantizer
    return "bf16" if quantizer is None else str(quantizer.format)


def layers(model, disaggregated: int) -> list[int]:
    """The sizes of the expert layers that exchange tokens.

    Returns:
        A flat list of ``num_experts, top_k, hidden_size`` per layer, the
        layers back to back in collective (traversal) order.
    """
    return [
        int(value)
        for layer in _layers(model, bool(disaggregated))
        for value in (layer.num_experts, layer.top_k, layer.hidden_size)
    ]


def symmetric_bytes(
    model, transport: str, size: int, attention_ranks: int, max_tokens: int
) -> int:
    """Bytes of one microbatch's peer-mapped storage on every member.

    Only split MegaMoE (``megamoe`` with dedicated expert ranks) exchanges
    through storage the worker maps; every other transport returns zero.
    ``size`` and ``attention_ranks`` are the union's member count and its
    attention ranks, as in :func:`create_exchange`.
    """
    attention_ranks = int(attention_ranks)
    if transport != "megamoe" or not attention_ranks:
        return 0
    from uniserve.backends.experts.split import (
        symmetric_bytes as storage_bytes,
    )

    first = _layers(model, True)[0]
    return storage_bytes(
        size=int(size),
        attention_ranks=attention_ranks,
        max_tokens=int(max_tokens),
        num_experts=int(first.num_experts),
        top_k=int(first.top_k),
        hidden=int(first.hidden_size),
        intermediate=int(first.intermediate_size),
        format=_encoding(first),
    )


def create_exchange(
    model,
    transport: str,
    rank: int,
    size: int,
    attention_ranks: int,
    host: transports.HostBootstrap,
    max_tokens: int,
    device: int,
    stream: int,
    yield_turn: Any,
    storage: torch.Tensor | None,
    pointers: Sequence[int],
):
    """Construct one microbatch's token transport, collectively.

    Args:
        model: The loaded model.
        transport: ``alltoall``, ``deepep`` or ``megamoe``.
        rank: This member's index in the expert union.
        size: The union's member count.
        attention_ranks: The union's attention ranks when routed experts
            run on dedicated ranks; zero when every rank holds its experts.
        host: The worker's host collectives over the union.
        max_tokens: Tokens one rank sends per step at most.
        device: The CUDA ordinal.
        stream: The microbatch's CUDA stream handle; construction enqueues
            its device work there.
        yield_turn: The microbatch rotation's function handing the host turn
            to the next microbatch after a dispatch; it releases the
            interpreter lock while it waits. ``None`` with one microbatch.
        storage: Split MegaMoE's mapped storage as an int8 tensor, else
            ``None``.
        pointers: Every member's address of that storage; empty without it.

    Returns:
        The transport, an opaque handle passed back unchanged.
    """
    attention_ranks = int(attention_ranks)
    candidates = _layers(model, bool(attention_ranks))
    if not candidates:
        raise ValueError("expert parallelism requires routed expert layers")
    first = candidates[0]
    target = torch.device("cuda", int(device))
    computation = torch.cuda.ExternalStream(int(stream), device=target)
    symmetric = None
    if storage is not None:
        symmetric = {
            "storage": storage,
            "pointers": [int(value) for value in pointers],
        }
    with torch.cuda.device(target), torch.cuda.stream(computation):
        return transports.create(
            str(transport),
            # The union communicator: ranks in member order, unbound.
            Communicator(tuple(range(int(size))), int(rank), "experts", target),
            host,
            max_tokens=int(max_tokens),
            top_k=int(first.top_k),
            num_experts=int(first.num_experts),
            hidden_size=int(first.hidden_size),
            intermediate_size=int(first.intermediate_size),
            activation=first.activation,
            attention_ranks=attention_ranks,
            device=target,
            yield_turn=yield_turn,
            encoding=_encoding(first),
            symmetric=symmetric,
        )


def bind_microbatches(exchanges) -> None:
    """Bind the microbatches' shared native plan, when the transport has one.

    Split MegaMoE launches the experts of every layer and microbatch once;
    call after every microbatch's binding is prepared.
    """
    exchanges = list(exchanges)
    if len({id(exchange) for exchange in exchanges}) != len(exchanges):
        raise ValueError("each microbatch requires its own expert exchange")
    exchanges[0].bind_microbatches(exchanges)


class ExpertLayers:
    """The routed expert layers of an expert-only rank on one stream.

    The rank holds no forward of its own: its layers run only the tokens
    other ranks route to them, through :func:`join`.

    Args:
        loaded: The loaded model, the rank's routed expert layers.
        stream: The CUDA stream handle of one microbatch.
        scratch: The stream's shared work areas.
        exchange: The microbatch's expert exchange.
        max_tokens: The most tokens a rank sends per expert step.
    """

    def __init__(
        self, loaded, stream: int, scratch, exchange, max_tokens: int
    ) -> None:
        from uniserve.models import TextSize
        from uniserve.runner.context import execution_context

        self.context = execution_context(
            loaded.module,
            int(stream),
            loaded.attention,
            scratch,
            experts=exchange,
        )
        self.context.prepare(TextSize(int(max_tokens), 1))

    def close(self) -> None:
        """Release the layers' operators after every join graph retired."""
        self.context.close()


def join(owner, capacity: int) -> None:
    """Join one expert step of ``capacity`` tokens per rank with no tokens.

    ``owner`` holds the execution context of a module with expert-parallel
    layers as ``context``: a runner, a prepared component or the
    :class:`ExpertLayers` of an expert-only rank. On the context's stream,
    every expert layer exchanges, runs the tokens other ranks route to this
    rank's experts and returns their results; this rank sends none.
    """
    context = owner.context
    if context is None:
        raise ValueError("expert joins require a prepared execution context")
    with (
        torch.inference_mode(),
        context.activate(),
        context.expert_step(int(capacity)),
    ):
        pass


def warm_thread(device: int) -> None:
    """Create this thread's numerical handles on ``device`` before capture.

    cuBLAS handles are thread-local; a microbatch thread must own one before
    any of its calls is captured.
    """
    target = torch.device("cuda", int(device))
    torch.cuda.set_device(target)
    # A first allocation makes the device's primary context current on this
    # thread, which the handle creation requires.
    torch.empty(0, device=target)
    torch.cuda.current_blas_handle()


def _prefetched_layers(model) -> list[FusedMoE]:
    """Expert layers whose weights are distributed: partitioned and loaded."""
    return [
        module
        for module in _module(model).modules()
        if isinstance(module, FusedMoE)
        and module.expert_group.size > 1
        and not module.up_gate.weight.is_meta
    ]


@torch.inference_mode()
def weight_fields(model) -> list[Any]:
    """Place every distributed expert layer's weights and list its shards.

    Each layer's weights are placed in the CuTeDSL kernels' representation
    before the worker publishes them; the full-extent views later bound
    keep that representation. Every field except a scalar or per-expert
    ``tensor_scale`` (which stays resident in full) is sharded over the
    expert group by its leading (expert) axis.

    Returns:
        ``[counts, names, shards]``: the number of fields of each layer in
        traversal order, then every field's name and its shard (a
        contiguous tensor of this rank's experts), the layers back to back.

    Raises:
        ValueError: No layer is distributed, the layers span several groups
            or devices, or a layer's kernels are not CuTeDSL.
    """
    from uniserve.backends import moe
    from uniserve.models.inputs import TextSize

    layers = _prefetched_layers(model)
    if not layers:
        raise ValueError("DWDP requires loaded expert shards")
    group = layers[0].expert_group
    device = layers[0].up_gate.weight.device
    if any(layer.expert_group != group for layer in layers):
        raise ValueError("DWDP weights must share one expert group")
    if device.type != "cuda":
        raise ValueError("DWDP requires CUDA peer memory")
    counts: list[int] = []
    names: list[str] = []
    shards: list[torch.Tensor] = []
    for layer in layers:
        provider = moe.resolve("auto", module=layer, device=device)
        if provider.name != "cutedsl":
            raise ValueError("DWDP requires CuTeDSL expert kernels")
        # Place the checkpoint encoding before publishing immutable pages;
        # prepared operators only borrow views of that representation.
        provider.prepare(
            module=layer, size=TextSize(1, 1), workspace={}
        ).close()
        fields = 0
        for name in ("up_gate", "down"):
            weight = getattr(layer, name).weight
            parts = (
                weight.buffers().items()
                if isinstance(weight, QuantizedTensor)
                else [("weight", weight)]
            )
            for field, value in parts:
                if field == "tensor_scale":
                    continue
                names.append(f"{name}.{field}")
                shards.append(value.contiguous())
                fields += 1
        counts.append(fields)
    return [counts, names, shards]


@torch.inference_mode()
def bind_weights(
    model, layer: int, names: Sequence[str], *fields: torch.Tensor
) -> None:
    """Install one layer's full weights over the caller's mapped views.

    ``names`` are the field names :func:`weight_fields` listed for the
    layer and ``fields`` the full field of each, as a contiguous byte
    tensor (every rank's shard in expert order). The layer then holds every
    expert; its group stays, so an unbound call still requires the
    prefetch.
    """
    module = _prefetched_layers(model)[int(layer)]
    views = {
        str(name): value for name, value in zip(names, fields, strict=True)
    }
    group = module.expert_group
    for name in ("up_gate", "down"):
        linear = getattr(module, name)
        weight = linear.weight

        def full(field, value, prefix=name):
            data = views[f"{prefix}.{field}"].view(value.dtype)
            return data.view(value.shape[0] * group.size, *value.shape[1:])

        if isinstance(weight, QuantizedTensor):
            buffers = {}
            for field, value in weight.buffers().items():
                if field == "tensor_scale":
                    # Kernels derive their scales at preparation, so these
                    # few immutable values stay fully resident.
                    buffers[field] = (
                        value if value.ndim == 0 else group.all_gather(value)
                    )
                else:
                    buffers[field] = full(field, value)
            placed = weight.quantizer.from_tensors(
                buffers,
                shape=(module.num_experts, *weight.shape[1:]),
                dtype=weight.dtype,
                scale_layout=weight.scale_layout,
                row_order=weight.row_order,
            )
        else:
            placed = full("weight", weight)
        linear.weight = nn.Parameter(placed, requires_grad=False)
        linear.num_experts = module.num_experts
    module.expert_slice = slice(0, module.num_experts)


@dataclass
class _Invocation:
    """The prefetch state of one invocation (an activation)."""

    # Whether an expert layer was reached: the first one orders every slot's
    # next copy after the stream's earlier work.
    started: bool = False
    # The layer the previous layer prefetched, whose copy is submitted.
    queued: int | None = None
    # The layer whose copy was submitted last, which the invocation's end
    # joins even if its expert call never ran.
    last: int | None = None


class WeightPrefetch:
    """The prefetch schedule of a rank's distributed expert weights.

    Every distributed expert layer reads its full weights through views of
    this rank's resident shard and of two reusable slots, which the worker
    maps. Layer ``i`` reads the other ranks' shards from slot ``i % 2``
    (:class:`uniserve.weight_prefetch.SlotCopies`): the worker's ``copy``
    submits one layer's peer reads into its slot on a stream it names.

    Expert calls bracket their computation with :meth:`before` and
    :meth:`after`, on their computing stream; an activation brackets one
    invocation (:meth:`activate`). ``before`` makes the layer readable and
    prefetches the next layer into the other slot, ``after`` releases the
    layer's slot after its last reader, and the invocation's end joins the
    last copy it submitted. Inference waits on local copy and compute events
    alone. Under graph capture a copy cannot be recorded: it becomes the
    graph's break work (:func:`uniserve.cuda_graph.break_graph`), which
    every replay submits between the two segments, and the wait for a
    layer's slot is captured after the break that submits its copy.

    Invocations serialize on their computing stream.

    Args:
        layers: The distributed expert layers in traversal order.
        device: Their CUDA device.
        copy: ``copy(layer, stream)`` submits every peer read of ``layer``
            on the CUDA stream handle ``stream``.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        device: torch.device,
        copy: Callable[[int, int], Any],
    ) -> None:
        self._ordinals = {
            id(layer): index for index, layer in enumerate(layers)
        }
        self.device = torch.device(device)
        self._slots = SlotCopies(
            self.device,
            2,
            len(self._ordinals),
            lambda layer, stream: copy(layer, stream.cuda_stream),
        )
        # Open invocations, innermost last.
        self._invocations: list[_Invocation] = []

    def contains(self, module) -> bool:
        """Whether ``module`` is one of the distributed expert layers."""
        return id(module) in self._ordinals

    def _ordinal(self, module) -> int:
        ordinal = self._ordinals.get(id(module))
        if ordinal is None:
            raise ValueError("module has no prefetched expert weights")
        return ordinal

    def _invocation(self) -> _Invocation:
        if not self._invocations:
            raise RuntimeError(
                "a distributed expert call requires an active invocation"
            )
        return self._invocations[-1]

    def _compute(self) -> torch.cuda.Stream:
        return torch.cuda.current_stream(self.device)

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Bracket one eager or captured invocation on the current stream.

        Raises:
            RuntimeError: The weights are closed.
        """
        if self._slots.closed:
            raise RuntimeError("distributed expert weights are closed")
        self._invocations.append(_Invocation())
        try:
            yield
        finally:
            last = self._invocations.pop().last
            if last is not None:
                self._slots.wait(last, self._compute())

    def before(self, module) -> None:
        """Wait for this layer's weights and prefetch the next layer's."""
        index = self._ordinal(module)
        invocation = self._invocation()
        compute = self._compute()
        capturing = torch.cuda.is_current_stream_capturing()

        if not invocation.started:
            invocation.started = True
            # The stream's earlier work may still read either slot.
            self._slots.release_all(compute)
        if invocation.queued != index:
            self._prefetch(invocation, index, capturing)
        self._slots.wait(index, compute)

        following = index + 1
        if following < self._slots.layers:
            invocation.queued = following
            self._prefetch(invocation, following, capturing)
        else:
            invocation.queued = None

    def _prefetch(
        self, invocation: _Invocation, index: int, capturing: bool
    ) -> None:
        """Copy ``index`` into its slot now, or at every replay."""
        invocation.last = index
        if capturing:
            break_graph(partial(self._slots.copy, index))
        else:
            self._slots.copy(index)

    def after(self, module) -> None:
        """Release the layer's slot after its last reader."""
        self._slots.release(self._ordinal(module), self._compute())

    def close(self) -> None:
        """Drain the copies once every invocation and graph retired.

        The worker releases the weights' memory after this returns.

        Raises:
            RuntimeError: An invocation is active.
        """
        if self._invocations:
            raise RuntimeError("cannot close an active weight prefetch")
        self._slots.close()


def bind_prefetch(model, copy: Callable[[int, int], Any]) -> WeightPrefetch:
    """The prefetch object execution contexts pass to their expert calls.

    Args:
        model: The loaded model, whose distributed expert layers
            :func:`bind_weights` installed.
        copy: The worker's ``copy(layer, stream)``, which submits every
            peer read of a layer on a CUDA stream handle.
    """
    layers = _prefetched_layers(model)
    return WeightPrefetch(layers, layers[0].up_gate.weight.device, copy)
