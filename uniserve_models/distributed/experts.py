"""Distributed routed experts: expert exchanges and distributed weights.

The rank runtime owns the expert step protocol: agreement on each step's
per-rank token capacity over its host group, microbatch threads and their
rotation, join graphs and close ordering. It reaches these functions by
import path to construct the numerical side:

* ``layers(model, disaggregated)`` describes the expert layers that
  exchange tokens, in collective (traversal) order.
* ``create_exchange(...)`` constructs one microbatch's expert exchange
  (``uniserve_models.backends.experts``), collectively over the expert
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
* ``close_exchange(exchange)`` releases an exchange; collective for
  exchanges with device communicators.

Distributed weight data parallelism (DWDP) exchanges weights instead of
tokens: ``weight_fields(model)`` places each expert layer's weights for its
kernels and returns this rank's shards, the caller publishes them and maps
every field's full extent, ``bind_weights`` installs the full views as the
layer's parameters, and ``bind_prefetch`` returns the object a prepared
component's expert calls prefetch peers' weights through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from torch import nn

from uniserve_models.backends import experts as transports
from uniserve_models.distributed import Communicator
from uniserve_models.nn.moe import FusedMoE
from uniserve_models.quantization import QuantizedTensor


def _module(model) -> nn.Module:
    """The loaded module tree of a ``model.load`` handle."""
    return getattr(model, "module", model)


def _layers(model, disaggregated: bool) -> list[FusedMoE]:
    """Routed-expert layers that exchange tokens, in traversal order.

    With dedicated expert ranks every routed-expert layer exchanges;
    colocated, the layers whose experts are partitioned across ranks.
    """
    return [
        module
        for module in _module(model).modules()
        if isinstance(module, FusedMoE)
        and (disaggregated or module.expert_group.size > 1)
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


def _group(group: Mapping[str, Any], device: torch.device) -> Communicator:
    """The union communicator: ranks in member order, unbound."""
    size = int(group["size"])
    return Communicator(
        tuple(range(size)), int(group["rank"]), "experts", device
    )


def symmetric_bytes(
    model, transport: str, group: Mapping[str, Any], max_tokens: int
) -> int:
    """Bytes of one microbatch's peer-mapped storage on every member.

    Only split MegaMoE (``megamoe`` with dedicated expert ranks) exchanges
    through storage the worker maps; every other transport returns zero.
    """
    attention_ranks = int(group["attention_ranks"])
    if transport != "megamoe" or not attention_ranks:
        return 0
    from uniserve_models.backends.experts.split import symmetric_bytes as size

    first = _layers(model, True)[0]
    return size(
        size=int(group["size"]),
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
    group: Mapping[str, Any],
    host: Mapping[str, Any],
    max_tokens: int,
    device: int,
    stream: int,
    hooks: Mapping[str, Any],
    symmetric: Mapping[str, Any] | None,
):
    """Construct one microbatch's token transport, collectively.

    Args:
        model: The ``model.load`` handle.
        transport: ``alltoall``, ``deepep`` or ``megamoe``.
        group: ``{"rank", "size", "attention_ranks"}`` of the union.
        host: The worker's host functions over the union
            (:class:`uniserve_models.backends.experts.HostBootstrap`).
        max_tokens: Tokens one rank sends per step at most.
        device: The CUDA ordinal.
        stream: The microbatch's CUDA stream handle; construction enqueues
            its device work there.
        hooks: ``{"yield": function or None}``: hands the host turn to the
            next microbatch after a dispatch.
        symmetric: Split MegaMoE's mapped storage, ``{"storage": int8
            tensor, "pointers": every member's address}``, else ``None``.

    Returns:
        The transport, an opaque handle passed back unchanged.
    """
    attention_ranks = int(group["attention_ranks"])
    candidates = _layers(model, bool(attention_ranks))
    if not candidates:
        raise ValueError("expert parallelism requires routed expert layers")
    first = candidates[0]
    target = torch.device("cuda", int(device))
    computation = torch.cuda.ExternalStream(int(stream), device=target)
    storage = None
    if symmetric is not None:
        storage = {
            "storage": torch.from_dlpack(symmetric["storage"]),
            "pointers": [int(value) for value in symmetric["pointers"]],
        }
    yield_turn = hooks.get("yield")
    if yield_turn is not None:
        # The worker's function waits for this microbatch's next turn. Other
        # microbatch threads run numerical code meanwhile, so the wait must
        # never hold the interpreter lock, whatever the process default for
        # native calls is.
        yield_turn.release_gil = True
    with torch.cuda.device(target), torch.cuda.stream(computation):
        return transports.create(
            str(transport),
            _group(group, target),
            transports.HostBootstrap(host),
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
            symmetric=storage,
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
        from uniserve_models.model import TextSize
        from uniserve_models.runner.context import execution_context

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


def close_exchange(exchange) -> None:
    """Release a transport after every binding and graph using it retired."""
    exchange.close()


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
    from uniserve_models.backends import moe
    from uniserve_models.model.inputs import TextSize

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


class WeightPrefetch:
    """A binding's view of the worker's distributed expert weights.

    Expert calls bracket their computation with :meth:`before` and
    :meth:`after`; an activation brackets one invocation. The worker's
    hooks submit peer reads and order them by events against the calling
    stream. Under graph capture, reads cannot be recorded: :meth:`before`
    hands them to the capture's recorder (:meth:`capture`), which ends the
    current graph segment, replays the read eagerly and begins the next.
    """

    def __init__(self, layers: list[FusedMoE], hooks: Mapping[str, Any]):
        self._ordinals = {
            id(layer): index for index, layer in enumerate(layers)
        }
        self.device = layers[0].up_gate.weight.device
        self._hooks = hooks
        self._submit: Callable[[Callable[[], object]], None] | None = None

    def _stream(self) -> int:
        return torch.cuda.current_stream(self.device).cuda_stream

    def contains(self, module) -> bool:
        return id(module) in self._ordinals

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Bracket one eager or captured invocation."""
        self._hooks["begin"]()
        try:
            yield
        finally:
            self._hooks["end"](self._stream())

    @contextmanager
    def capture(
        self, submit: Callable[[Callable[[], object]], None]
    ) -> Iterator[None]:
        """Route reads to a graph capture's segment recorder."""
        previous, self._submit = self._submit, submit
        try:
            yield
        finally:
            self._submit = previous

    def before(self, module) -> None:
        """Wait for this layer's weights and prefetch the next layer's."""
        # Under capture the worker hands each read to the recorder while it
        # orders the layer's waits, so every wait lands in the segment after
        # the read it depends on.
        record = None if self._submit is None else self._record
        self._hooks["before"](
            self._ordinals[id(module)], self._stream(), record
        )

    def _record(self, index: int) -> None:
        if self._submit is None:
            raise RuntimeError("capture expert weight reads with a recorder")
        self._submit(partial(self._hooks["copy"], int(index)))

    def after(self, module) -> None:
        """Release the layer's slots after its last reader."""
        self._hooks["after"](self._ordinals[id(module)], self._stream())


def bind_prefetch(model, hooks: Mapping[str, Any]) -> WeightPrefetch:
    """The prefetch object bindings pass to their expert calls.

    ``hooks`` are the worker's ``begin()``, ``end(stream)``, ``before(layer,
    stream, record)``, ``after(layer, stream)`` and ``copy(index)``;
    ``record`` is ``None`` eagerly and, under capture, receives each copy
    index to submit between graph segments.
    """
    return WeightPrefetch(_prefetched_layers(model), hooks)
