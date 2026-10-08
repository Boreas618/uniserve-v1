"""Distributed routed experts: the ``experts.*`` interface entries.

The worker owns the expert step protocol: agreement on each step's per-rank
token capacity over its host group, microbatch threads and their rotation,
join graphs and close ordering. These functions construct and bind the
numerical side:

* ``experts.layers(model, disaggregated)`` describes the expert layers that
  exchange tokens, in collective (traversal) order.
* ``experts.create_exchange(...)`` constructs one microbatch's token
  transport (``uniserve_models.backends.experts``), collectively over the
  expert union, bootstrapped with the worker's host functions.
* ``experts.bind_microbatches(exchanges)`` binds the microbatches' shared
  plan, when the transport has one.
* ``experts.symmetric_bytes(...)`` sizes the peer-mapped storage of split
  MegaMoE, which the worker allocates and maps.
* ``experts.bind(prepared)`` returns ``{"join": join(capacity)}`` for a
  bound component: one expert step of ``capacity`` in which this rank joins
  every expert layer with no tokens of its own.
* ``experts.warm_thread(device)`` prepares a microbatch thread's
  thread-local numerical state (cuBLAS handle) before graph capture.
* ``experts.close_exchange(exchange)`` releases a transport; collective for
  transports with device communicators.

Distributed weight data parallelism (DWDP) exchanges weights instead of
tokens: ``experts.weight_fields(model)`` places each expert layer's weights
for its kernels and returns this rank's shards, the worker publishes them
and maps every field's full extent, ``experts.bind_weights`` installs the
full views as the layer's parameters, and ``experts.bind_prefetch`` returns
the object a binding's expert calls prefetch peers' weights through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from torch import nn

from uniserve_models.backends import experts as transports
from uniserve_models.distributed import Communicator
from uniserve_models.nn.moe import FusedMoE
from uniserve_models.quantization import QuantizedTensor


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "experts.layers": layers,
        "experts.symmetric_bytes": symmetric_bytes,
        "experts.create_exchange": create_exchange,
        "experts.bind_microbatches": bind_microbatches,
        "experts.bind": bind,
        "experts.warm_thread": warm_thread,
        "experts.close_exchange": close_exchange,
        "experts.weight_fields": weight_fields,
        "experts.bind_weights": bind_weights,
        "experts.bind_prefetch": bind_prefetch,
    }


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


def layers(model, disaggregated: int) -> list[dict[str, Any]]:
    """Describe the expert layers that exchange tokens.

    Returns:
        One record per layer: ``path``, ``num_experts``, ``top_k``,
        ``hidden_size``, ``intermediate_size``, ``activation``,
        ``partitioned`` (the expert group size), ``input_encoding``
        (``bf16``, ``nvfp4`` or ``mxfp8``) and ``meta`` (whether this rank
        holds no expert weights).
    """
    names = {
        id(module): path for path, module in _module(model).named_modules()
    }
    return [
        {
            "path": names[id(layer)],
            "num_experts": int(layer.num_experts),
            "top_k": int(layer.top_k),
            "hidden_size": int(layer.hidden_size),
            "intermediate_size": int(layer.intermediate_size),
            "activation": str(layer.activation),
            "partitioned": int(layer.expert_group.size),
            "input_encoding": _encoding(layer),
            "meta": int(layer.up_gate.weight.is_meta),
        }
        for layer in _layers(model, bool(disaggregated))
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
            activation=str(first.activation),
            attention_ranks=attention_ranks,
            device=target,
            yield_turn=hooks.get("yield"),
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


def bind(prepared) -> dict[str, Callable[..., Any]]:
    """Join functions of a bound component with expert-parallel layers.

    ``join(capacity)`` runs one expert step of ``capacity`` tokens per
    rank on the binding's stream in which this rank sends no tokens: every
    expert layer exchanges, runs the tokens other ranks route to this
    rank's experts and returns their results.
    """
    context = prepared.context
    if context is None:
        raise ValueError("expert joins require a bound component")

    def join(capacity: int) -> None:
        with (
            torch.inference_mode(),
            context.activate(),
            context.expert_step(int(capacity)),
        ):
            pass

    return {"join": join}


def warm_thread(device: int) -> None:
    """Create this thread's numerical handles on ``device`` before capture.

    cuBLAS handles are thread-local; a microbatch thread must own one before
    any of its calls is captured.
    """
    with torch.cuda.device(int(device)):
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
def weight_fields(model) -> list[dict[str, Any]]:
    """Place every distributed expert layer's weights and list its shards.

    Each layer's weights are placed in the CuTeDSL kernels' representation
    before the worker publishes them; the full-extent views later bound
    keep that representation. Every field except a scalar or per-expert
    ``tensor_scale`` (which stays resident in full) is sharded over the
    expert group by its leading (expert) axis.

    Returns:
        One record per layer, in traversal order: ``{"layer": ordinal,
        "fields": [{"name", "shard"}]}``, ``shard`` a contiguous tensor of
        this rank's experts.

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
    result = []
    for ordinal, layer in enumerate(layers):
        provider = moe.resolve("auto", module=layer, device=device)
        if provider.name != "cutedsl":
            raise ValueError("DWDP requires CuTeDSL expert kernels")
        # Place the checkpoint encoding before publishing immutable pages;
        # prepared operators only borrow views of that representation.
        provider.prepare(
            module=layer, size=TextSize(1, 1), workspace={}
        ).close()
        fields = []
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
                fields.append(
                    {"name": f"{name}.{field}", "shard": value.contiguous()}
                )
        result.append({"layer": ordinal, "fields": fields})
    return result


@torch.inference_mode()
def bind_weights(model, layer: int, fields: Mapping[str, Any]) -> None:
    """Install one layer's full weights over the worker's mapped views.

    ``fields`` maps each name :func:`weight_fields` listed to the full
    field as a contiguous byte tensor (every rank's shard in expert order).
    The layer then holds every expert; its group stays, so an unbound call
    still requires the prefetch.
    """
    module = _prefetched_layers(model)[int(layer)]
    views = {
        str(name): torch.from_dlpack(value) for name, value in fields.items()
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
        recorded = self._hooks["before"](
            self._ordinals[id(module)], self._stream()
        )
        if not recorded:
            return
        if self._submit is None:
            raise RuntimeError("capture expert weight reads with a recorder")
        for index in recorded:
            self._submit(partial(self._hooks["copy"], int(index)))

    def after(self, module) -> None:
        """Release the layer's slots after its last reader."""
        self._hooks["after"](self._ordinals[id(module)], self._stream())


def bind_prefetch(model, hooks: Mapping[str, Any]) -> WeightPrefetch:
    """The prefetch object bindings pass to their expert calls.

    ``hooks`` are the worker's ``begin()``, ``end(stream)``, ``before(layer,
    stream) -> [copy]``, ``after(layer, stream)`` and ``copy(index)``.
    """
    return WeightPrefetch(_prefetched_layers(model), hooks)
