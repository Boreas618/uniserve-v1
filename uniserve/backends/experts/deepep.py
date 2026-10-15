"""Elastic expert dispatch/combine over a shared source/expert world.

Attention ranks precede expert ranks in a disaggregated world. DeepEP's
uniform expert placement is expressed by reserving unrouted expert slots
on attention ranks and translating the real expert ids at dispatch. The
caller still sees global model expert ids and unpadded hidden states.
"""

from __future__ import annotations

import torch

from uniserve.distributed import Communicator
from uniserve.quantization import QuantizedTensor, ScaleLayout

from . import ExpertTransport


class DeepEP(ExpertTransport):
    """Own one elastic communication buffer and its native communicator.

    Construction and ``close`` are collective over ``group``; ``bootstrap``
    distributes the NCCL unique id and runs the host barriers. Calls must
    alternate ``dispatch`` and ``combine`` in the same order on all ranks,
    including ranks with no tokens. Different microbatches use different
    instances, each with its own communication stream and buffer. A caller
    drains device work before closing; on distributed failure it exits the
    process without collective cleanup. The first call waits on a host
    barrier, so a peer's preceding compilation cannot consume a device
    collective timeout.

    ``attention_ranks=0`` serves ordinary expert parallelism. Otherwise the
    first ``attention_ranks`` members source tokens and the remaining ranks
    hold equal contiguous expert partitions. Attention ranks hold no experts
    and expert ranks source no tokens. Route weights are applied by expert
    computation exactly once, before ``combine``.
    """

    transport = "deepep"

    def __init__(
        self,
        group: Communicator,
        bootstrap,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        max_tokens: int,
        attention_ranks: int = 0,
        yield_turn=None,
    ):
        expert_ranks = group.size - attention_ranks
        if (
            group.size < 2
            or not 0 <= attention_ranks < group.size
            or num_experts < 1
            or num_experts % expert_ranks
            or not 0 < top_k <= min(num_experts, 32)
            or min(hidden_size, max_tokens) < 1
        ):
            raise ValueError("invalid elastic expert placement or capacity")
        if group.device.type != "cuda":
            raise ValueError("DeepEP requires a CUDA communicator")

        from uniserve_kernels.deepep import load

        super().__init__(
            group,
            bootstrap,
            max_tokens=max_tokens,
            attention_ranks=attention_ranks,
            barrier_first=True,
            yield_turn=yield_turn,
        )
        self.hidden_size = hidden_size
        # DeepEP's combine distributes int4 vectors over 32-lane warps.
        self.wire_hidden = (hidden_size + 255) // 256 * 256
        self.top_k = top_k
        local_experts = num_experts // expert_ranks
        self.expert_offset = attention_ranks * local_experts
        self.local_start = (group.rank - attention_ranks) * local_experts
        self.wire_experts = group.size * local_experts
        self._native = load()
        identity = bootstrap.bcast(
            self._native.get_local_nccl_unique_id()
            if group.rank == 0
            else None,
            root=0,
        )
        self._comm = self._native.create_nccl_comm(
            identity, group.size, group.rank
        )
        # The elastic metadata can use one guard slot beyond a token bucket.
        # The caller-visible capacity remains max_tokens.
        nbytes = self._native.calculate_elastic_buffer_size(
            self._comm,
            max_tokens + 1,
            self.wire_hidden,
            top_k,
            False,
            True,
            True,
        )
        self._buffer = self._native.ElasticBuffer(
            group.rank,
            group.size,
            self._comm,
            nbytes,
            False,
            True,
            True,
            True,
            3,
            129,
            120,
            120,
            True,
        )
        torch.cuda.synchronize(group.device)
        bootstrap.barrier()
        self._buffer.barrier(True, False)
        torch.cuda.synchronize(group.device)
        self._sms = min(
            24,
            torch.cuda.get_device_properties(
                group.device
            ).multi_processor_count,
        )
        self._qps = min(self._sms * 16 + 1, 129)
        # Routing state from dispatch to its matching combine: source
        # metadata, copied expert ids, cumulative receive counts, optional
        # token metadata and channel links, the capacity and received shape.
        self._handle: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                int,
                torch.Size,
            ]
            | None
        ) = None

    def dispatch(self, hidden, topk_ids, topk_weights, *, invalid_expert):
        """Route BF16 or encoded NVFP4 rows without changing their values.

        Returned tensors have a static worst-case row extent, suitable for
        CUDA capture. Unused rows carry zero states and weights and the
        ``invalid_expert`` id; no host read of the received token count is
        required. The returned views stay live through the matching
        ``combine``.
        """
        states, ids, weights = self._dispatch(
            hidden, topk_ids, topk_weights, capacity=self.capacity
        )
        # Each numerical provider declares the sentinel its kernel skips.
        return states, torch.where(ids < 0, invalid_expert, ids), weights

    def _dispatch(self, hidden, ids, weights, *, capacity):
        if self._buffer is None or self._handle is not None:
            raise RuntimeError("dispatch requires an open, idle exchange")
        tokens = hidden.shape[0]
        if (
            hidden.ndim != 2
            or hidden.shape[1] != self.hidden_size
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.group.device
            or ids.shape != (tokens, self.top_k)
            or weights.shape != ids.shape
            or ids.dtype != torch.int32
            or weights.dtype != torch.float32
            or not tokens <= capacity <= self.max_tokens
            or self.attention_ranks
            and not self.source_only
            and tokens
        ):
            raise ValueError("expert dispatch inputs do not fit the exchange")
        encoded = isinstance(hidden, QuantizedTensor)
        scales = fields = None
        if encoded:
            if (
                hidden.quantizer.format != "nvfp4"
                or hidden.quantizer.calibrated_scale is None
                or hidden.scale_layout is not ScaleLayout.LINEAR
            ):
                raise ValueError(
                    "DeepEP encoded inputs require calibrated linear NVFP4"
                )
            fields = hidden.buffers()
            # DeepEP infers the mathematical hidden width from the data
            # column count, which combine also uses. Keep that width with
            # byte-valued columns: the leading H/2 bytes are packed E2M1.
            # Scales travel through its opaque 32-bit scale-pack channel.
            # No value is dequantized or requantized at this boundary.
            wire = torch.nn.functional.pad(
                fields["values"],
                (0, self.wire_hidden - self.hidden_size // 2),
            )
            scales = torch.nn.functional.pad(
                fields["block_scale"].view(torch.uint8),
                (0, (self.wire_hidden - self.hidden_size) // 16),
            ).view(torch.int32)
        else:
            wire = torch.nn.functional.pad(
                hidden, (0, self.wire_hidden - self.hidden_size)
            )
        indices = torch.where(ids >= 0, ids + self.expert_offset, ids).to(
            self._native.topk_idx_t
        )
        values = self._buffer.dispatch(
            wire.contiguous(),
            scales,
            indices.contiguous(),
            weights.contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            capacity,
            self.wire_experts,
            1,
            self._sms,
            self._qps,
            None,
            None,
            False,
            False,
            True,
            False,
            False,
            True,
        )
        (
            received,
            received_scales,
            local_ids,
            route_weights,
            copied_ids,
            _,
            counts,
            _,
            metadata,
            _,
            token_metadata,
            links,
            _,
        ) = values
        self._handle = (
            metadata,
            copied_ids,
            counts,
            token_metadata,
            links,
            capacity,
            received.shape,
        )
        valid = (
            torch.arange(received.shape[0], device=hidden.device) < counts[-1]
        )
        valid_routes = valid[:, None] & (local_ids >= 0)
        global_ids = torch.where(
            valid_routes, local_ids + self.local_start, -1
        ).to(torch.int32)
        route_weights = torch.where(valid_routes, route_weights, 0)
        # Encoded inputs, and only those, retained their fields above.
        if fields is not None:
            # Static unused receive rows have no routes. Initialize their
            # encoded fields as well so subsequent numerical kernels never
            # observe uninitialized scale bytes while masking padding.
            packed = torch.where(
                valid[:, None], received[:, : self.hidden_size // 2], 0
            ).contiguous()
            scale_bytes = received_scales.contiguous().view(torch.uint8)
            scale_bytes = torch.where(
                valid[:, None], scale_bytes[:, : self.hidden_size // 16], 0
            ).contiguous()
            states = hidden.quantizer.from_tensors(
                {
                    "values": packed,
                    "block_scale": scale_bytes,
                    "tensor_scale": fields["tensor_scale"],
                },
                shape=(received.shape[0], self.hidden_size),
                dtype=hidden.dtype,
            )
        else:
            states = torch.where(
                valid[:, None], received[:, : self.hidden_size], 0
            )
        return states, global_ids, route_weights

    def combine(self, partial, tokens):
        """Return each source's weighted expert sums in its original row order.

        ``partial`` has the same logical shape as dispatch's returned hidden
        states. Expert computation has already applied route weights;
        passing them to DeepEP again would multiply them twice.
        """
        output = self._combine(partial)
        if output.shape[0] != tokens:
            raise RuntimeError("the combine returned another token count")
        return output

    def _combine(self, partial):
        if self._buffer is None or self._handle is None:
            raise RuntimeError("combine requires a preceding dispatch")
        metadata, ids, counts, token_metadata, links, capacity, shape = (
            self._handle
        )
        if partial.shape != (shape[0], self.hidden_size):
            raise ValueError("expert output shape differs from dispatched rows")
        wire = torch.nn.functional.pad(
            partial, (0, self.wire_hidden - self.hidden_size)
        ).contiguous()
        output, _, _ = self._buffer.combine(
            wire,
            None,
            None,
            None,
            metadata,
            ids,
            counts,
            token_metadata,
            links,
            self.wire_experts,
            capacity,
            self._sms,
            self._qps,
            None,
            None,
            False,
            False,
            False,
        )
        self._handle = None
        # Communication alignment is private to the transport. Numerical
        # consumers, including fused residual norms, read packed token rows.
        return output[:, : self.hidden_size].contiguous()

    def close(self):
        """Release the collective resources after all invocations finish.

        Every member reaches the host barrier after draining its own work,
        because a local drain does not retire a peer's remote accesses.
        """
        if self._handle is not None:
            raise RuntimeError("complete the expert exchange before closing")
        if self._buffer is None:
            return
        self.barrier()
        self._buffer.destroy()
        self._native.destroy_nccl_comm(self._comm)
        self._buffer = None
