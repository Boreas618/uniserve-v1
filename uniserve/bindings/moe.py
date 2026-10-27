"""Routed-expert operators and workspace for one ``FusedMoE`` call site.

An expert-parallel call site exchanges tokens with the other ranks of its
expert group inside an :class:`ExpertStep`: every rank calls the same expert
layers in the same order in every step, ranks without tokens included, at
one per-rank token capacity the ranks agreed on before the step.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence

import torch

from uniserve.models.inputs import TextSize
from uniserve.nn.functional import Routes
from uniserve.quantization import QuantizedTensor, RowOrder

from ..backends import moe as moe_backend
from ..backends import record_kernel_choice
from . import capturing

# The expert step of the running numerical invocation, if any.
_STEP: contextvars.ContextVar[ExpertStep | None] = contextvars.ContextVar(
    "uniserve_expert_step", default=None
)


class ExpertStep:
    """One expert step of a numerical invocation.

    ``capacity`` is the step's per-rank token capacity, identical on every
    rank of the group; ``layers`` are the invocation's expert-parallel call
    sites in collective (traversal) order. While the step is open, each call
    site records that the forward reached it; :meth:`join_tail` then joins
    the layers the forward did not reach, with no tokens, so every rank
    exchanges at every layer of the step. A forward may stop early (a
    pipeline stage, a readout), but it may not skip a layer before one it
    reaches. One step may span two invocations of the same rank: the first
    leaves its tail unjoined, the second starts with the first's layers
    ``reached`` and joins what neither reached.

    The first call of a transport that needs it waits on the group's host
    barrier, so a peer's preceding compilation cannot consume a device
    collective timeout; that call must not be under CUDA graph capture.
    """

    def __init__(
        self,
        exchange,
        layers: Sequence[MoEBinding],
        capacity: int,
        dtype: torch.dtype,
        *,
        reached: int = 0,
    ) -> None:
        if not 0 < capacity <= exchange.max_tokens:
            raise ValueError(
                f"expert step capacity {capacity} is outside the exchange's "
                f"{exchange.max_tokens} tokens per rank"
            )
        self.exchange = exchange
        self.layers = tuple(layers)
        self.capacity = capacity
        self.dtype = dtype
        # A step one invocation continues from a preceding one (a readout's
        # exchanging tail) starts with that invocation's layers reached.
        self.reached = bytearray(len(self.layers))
        self.reached[:reached] = b"\x01" * min(reached, len(self.layers))
        self._token: contextvars.Token[ExpertStep | None] | None = None

    def __enter__(self) -> ExpertStep:
        if _STEP.get() is not None:
            raise RuntimeError("an expert step is already open")
        self.exchange.capacity = self.capacity
        self._token = _STEP.set(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.exchange.capacity = 0
        # ``__enter__`` stored the token of the step this exit closes.
        assert self._token is not None
        _STEP.reset(self._token)
        self._token = None

    def enter(self, binding: MoEBinding) -> None:
        """Record that the forward reached ``binding``'s layer."""
        exchange = self.exchange
        if not exchange.ready:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm the expert exchange before capture")
            exchange.barrier()
            exchange.ready = True
        self.reached[binding.ordinal] = 1

    def join_tail(self) -> None:
        """Join every layer after the last one the forward reached.

        Raises:
            RuntimeError: The forward skipped a layer before one it reached.
        """
        tail = self.reached.find(0)
        if tail < 0:
            return
        if any(self.reached[tail:]):
            raise RuntimeError(
                "a forward skipped an expert layer before one it reached"
            )
        for binding in self.layers[tail:]:
            binding.join(binding.module.hidden_size, self.dtype)


def current_step() -> ExpertStep:
    """The open expert step of an expert-parallel call site.

    Raises:
        RuntimeError: No expert step is open.
    """
    step = _STEP.get()
    if step is None:
        raise RuntimeError("expert-parallel layers run inside an expert step")
    return step


class MoEBinding:
    """Specialize one ``FusedMoE`` call site for a token capacity.

    The operator is prepared for the largest token count seen, never during
    CUDA capture; a smaller call reuses it.

    An expert-parallel call site (``module.expert_group`` spans several
    ranks, or experts run on dedicated ranks) borrows an expert transport
    (``backends.experts``): each call sends its tokens to the ranks holding
    their experts, runs this rank's experts over the tokens it receives,
    ``P * C`` rows for a group of ``P`` ranks and the open step's capacity
    ``C``, and combines the partial sums back. Its local operator is
    prepared once for ``P`` times the transport's largest capacity. Hidden
    states of NVFP4 experts travel in the experts' input encoding, dense
    hidden states otherwise. ``ordinal`` is the call site's position among
    the invocation's expert-parallel layers, in collective order.
    """

    def __init__(
        self,
        module,
        backend,
        size,
        device,
        allocate,
        exchange=None,
        weights=None,
        ordinal: int = 0,
    ):
        self.module, self.backend, self.device = module, backend, device
        self.allocate = allocate
        self.ordinal = ordinal
        self.size = size
        self.operator = None
        self.provider: str | None = None
        self.weights = (
            weights
            if weights is not None and weights.contains(module)
            else None
        )
        self.exchange = exchange if self.weights is None else None
        # Experts a tensor-parallel group divides (its group is the expert
        # group) see every token on every rank: no exchange moves them.
        if (
            module.expert_group.size > 1
            and module.expert_group is not module.group
            and exchange is None
            and self.weights is None
        ):
            raise ValueError(
                "distributed experts require token exchange or weight prefetch"
            )
        # Unit route weights of an uncombined expert-parallel call, filled
        # once for the exchange's largest local token count.
        self._unit = (
            None
            if self.exchange is None
            else torch.ones(
                (self.exchange.max_tokens, 1),
                dtype=torch.float32,
                device=device,
            )
        )

    def prepare(self, size: TextSize):
        if (
            self.exchange is not None
            and self.exchange.source_only
            and self.exchange.transport != "megamoe"
        ):
            # This rank owns routing/attention, so its expert parameters can
            # remain meta. Only expert ranks prepare numerical expert kernels.
            self.provider = "remote"
            self._invalid_expert = self.module.num_experts
            return None
        previous = self.operator
        if self.exchange is not None:
            # Received rows, not local tokens, reach the local experts.
            size = TextSize(
                self.exchange.group.size * self.exchange.max_tokens, 1
            )
        if previous is not None and previous.size.num_tokens >= size.num_tokens:
            return previous
        if capturing(self.device):
            raise RuntimeError(
                "expert capacity must be prepared before capture"
            )
        if self.size is not None:
            size = TextSize(
                max(size.num_tokens, self.size.num_tokens),
                max(size.batch_size, self.size.batch_size),
            )
        if (
            self.exchange is not None
            and self.exchange.transport == "megamoe"
            and self.exchange.attention_ranks
        ):
            operator = self.exchange.fused.prepare(self.module, size)
            provider: moe_backend.Backend | None = None
        elif self.exchange is not None and self.exchange.transport == "megamoe":
            # The fused kernel exchanges, computes and combines in one launch
            # over the exchange's symmetric buffers.
            from ..backends.moe import megamoe

            provider = megamoe.Backend()
            reason = provider.unsupported(self.module)
            if reason is not None:
                raise ValueError(f"MegaMoE cannot serve this layer: {reason}")
            operator = provider.prepare_fused(
                module=self.module, size=size, buffer=self.exchange.fused
            )
        else:
            provider = moe_backend.resolve(
                self.backend, module=self.module, device=self.device
            )
            requirements = provider.workspace_buffers(
                module=self.module, size=size
            )
            operator = provider.prepare(
                module=self.module,
                size=size,
                workspace=self.allocate(requirements, self.device),
            )
        if previous is not None:
            previous.close()
        self.operator = operator
        self.provider = "megamoe" if provider is None else provider.name
        self._invalid_expert = (
            self.module.num_experts
            if provider is None
            else provider.invalid_expert(self.module)
        )
        record_kernel_choice()
        return operator

    def kernels(self):
        """Describe the grouped-expert kernel serving this call site.

        The record names the prepared provider, the expert count, and each
        projection's weight representation and resident row order (the
        physical order the provider placed; ``linear`` is logical order).
        An expert-parallel call site also names its expert group size, its
        exchange transport (``ExpertExchange.transport``, the name the launch
        selects it by) and the representation its hidden states travel in.
        Empty until the call site is prepared.
        """
        if self.provider is None:
            return []
        projections = {}
        for name in ("up_gate", "down"):
            weight = getattr(self.module, name).weight
            quantized = isinstance(weight, QuantizedTensor)
            projections[name] = {
                "weight": weight.quantizer.format
                if quantized
                else str(weight.dtype).removeprefix("torch."),
                "row_order": weight.row_order.value
                if quantized
                else RowOrder.LINEAR.value,
            }
        parallel = (
            {}
            if self.exchange is None
            else {
                "expert_parallel": self.exchange.group.size,
                "exchange": self.exchange.transport,
                # The representation hidden states travel in.
                "payload": "nvfp4"
                if isinstance(self.operator, moe_backend.NVFP4Operator)
                else "dense",
            }
        )
        return [
            {
                "op": "moe",
                "provider": self.provider,
                "experts": self.module.num_experts,
                "activation": self.module.activation,
                **projections,
                **parallel,
            }
        ]

    def __call__(self, hidden, topk_ids, topk_weights, *, combine=True):
        exchange = self.exchange
        if exchange is None:
            operator = self.prepare(TextSize(hidden.shape[0], 1))
            if self.weights is not None:
                self.weights.before(self.module)
            output = operator(hidden, topk_ids, topk_weights, combine=combine)
            if self.weights is not None:
                self.weights.after(self.module)
            return output

        operator = self.prepare(TextSize(hidden.shape[0], 1))
        current_step().enter(self)
        if exchange.transport == "megamoe":
            # The fused kernel exchanges, runs the experts and combines in
            # one launch over the exchange's symmetric buffers.
            output = operator(hidden, topk_ids, topk_weights)
        else:
            quantizer = self.module.up_gate.input_quantizer
            if (
                exchange.transport == "deepep"
                and quantizer is not None
                and quantizer.format == "nvfp4"
                and not isinstance(hidden, QuantizedTensor)
            ):
                # Both roles declare the same wire representation, including
                # empty participants whose expert parameters may remain meta.
                hidden = quantizer.quantize(hidden)
            if (
                isinstance(operator, moe_backend.NVFP4Operator)
                and exchange.transport == "alltoall"
            ):
                # NVFP4 experts read their input encoding, so the hidden
                # states travel in it: rows already stored in it as they
                # are, BF16 rows (a join's empty rows included) encoded here
                # to the bytes the operator would encode after the exchange.
                # The representation is thus a property of the layer, the
                # same on every rank of the step, including ranks that only
                # join it.
                hidden = operator.encode(hidden)
            received = exchange.dispatch(
                hidden,
                topk_ids,
                topk_weights,
                invalid_expert=self._invalid_expert,
            )
            # Other microbatches submit their dispatch while this one's
            # tokens travel.
            exchange.yield_turn()
            # The local experts combine their routes into this rank's
            # partial sums, and the exchange sums the ranks' partials (FP32,
            # one BF16 rounding).
            partial = (
                torch.zeros(
                    received[0].shape,
                    dtype=received[0].dtype,
                    device=received[0].device,
                )
                if exchange.source_only
                else operator(*received)
            )
            output = exchange.combine(partial, hidden.shape[0])
        # Uncombined, the combined rows are the one-route Routes of unit
        # weight.
        if combine:
            return output
        assert self._unit is not None
        return Routes(output.unsqueeze(1), self._unit[: output.shape[0]])

    def join(self, hidden_size: int, dtype: torch.dtype) -> None:
        """Exchange at this layer with no tokens of this rank's own.

        The other ranks' tokens routed to this rank's experts still arrive,
        run through them, and return; this rank contributes and receives
        nothing for itself. A rank joins the expert layers a step's forward
        did not reach, so every rank exchanges at every layer of every step.
        """
        top_k = self.module.top_k
        self(
            torch.empty((0, hidden_size), dtype=dtype, device=self.device),
            torch.empty((0, top_k), dtype=torch.int32, device=self.device),
            torch.empty((0, top_k), dtype=torch.float32, device=self.device),
        )

    def close(self):
        if self.operator is not None:
            self.operator.close()
        self.operator = self.provider = None
        self._unit = None
