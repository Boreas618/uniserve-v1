"""Split MegaMoE execution over a source/expert union.

Attention (source) ranks precede expert ranks in the union. Each source
writes its encoded tokens, routes and weights into peer-mapped symmetric
storage and launches the attention-side kernel in two phases around the
expert ranks' persistent grouped-GEMM launch. The serving runtime allocates
the symmetric storage and maps every member's allocation
(:func:`symmetric_bytes` sizes it); models keep ordinary routed-expert
calls, and their input quantizers fix the numerical representation
independently of rank placement.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from uniserve.quantization import QuantizedTensor, RowOrder, ScaleLayout

from ..moe import Operator, WeightLayout, verify


def _padded(width: int) -> int:
    """Physical tile padding of a hidden or intermediate width."""
    return (width + 511) // 512 * 512


# The physical row orders the expert kernels read: eight gate channels
# followed by the matching eight up channels, and ``down`` in logical order.
_ORDERS = {"up_gate": RowOrder.INTERLEAVED_8, "down": RowOrder.LINEAR}


def weight_layout(module) -> WeightLayout:
    """The layout split MegaMoE's expert kernels read ``module``'s weights in.

    The kernels read encoded values in ``_ORDERS`` and block scales as
    MN-major 128x4 tiles. A layer whose hidden or intermediate width the
    kernels pad to whole 512-column tiles is read through padded copies its
    operator derives from the weights as loaded, so its layout is empty.
    """
    # [E, 2I, H] and [E, H, I]: the resident widths of this rank's experts.
    hidden, intermediate = module.down.weight.shape[1:]
    if _padded(hidden) != hidden or _padded(intermediate) != intermediate:
        return {}
    return {
        name: (ScaleLayout.SWIZZLED_128X4_MN, order)
        for name, order in _ORDERS.items()
    }


def symmetric_bytes(
    *,
    size: int,
    attention_ranks: int,
    max_tokens: int,
    num_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    format: str,
) -> int:
    """Bytes of one member's symmetric storage for one microbatch."""
    from uniserve_kernels.deepgemm import load

    native = load()
    alignment = native.get_token_alignment_for_mega_moe()
    capacity = (max_tokens + alignment - 1) // alignment * alignment
    byte_count, _, _ = native.get_symm_buffer_size_for_mega_moe_m2n(
        attention_ranks,
        size - attention_ranks,
        num_experts,
        capacity,
        top_k,
        _padded(hidden),
        _padded(intermediate),
        format == "nvfp4",
    )
    return int(byte_count)


class Buffer:
    """Collective buffers for one microbatch's ordered expert calls.

    Construction is collective over ``group``: every member seeds its
    symmetric storage, then waits on the host barrier before any peer may
    address it. Calls alternate dispatch and combine before reusing that
    layout. The caller drains streams and closes bound operators before
    retiring the buffer.
    """

    def __init__(
        self,
        group,
        bootstrap,
        *,
        max_tokens,
        num_experts,
        top_k,
        hidden,
        intermediate,
        activation,
        attention_ranks,
        device,
        format: str,
        symmetric: Mapping[str, Any],
        yield_turn=None,
    ):
        if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
            raise ValueError("split MegaMoE requires a Blackwell CUDA device")
        if not 0 < attention_ranks < group.size:
            raise ValueError(
                "split MegaMoE requires both attention and experts"
            )
        if num_experts % (group.size - attention_ranks):
            raise ValueError("experts must divide the expert ranks")
        if activation != "silu":
            raise ValueError("split MegaMoE requires SiLU experts")
        if format not in {"mxfp8", "nvfp4"}:
            raise ValueError(
                "split MegaMoE requires MXFP8 or NVFP4 activations"
            )
        from uniserve_kernels.deepgemm import load

        self.group, self.device = group, device
        self.bootstrap = bootstrap
        self.attention_ranks = attention_ranks
        self.expert_ranks = group.size - attention_ranks
        self.source = group.rank < attention_ranks
        self.max_tokens, self.num_experts = max_tokens, num_experts
        self.top_k, self.hidden_size = top_k, hidden
        # Physical tile padding belongs to the provider, not the model's
        # mathematical dimensions. Padded channels have exactly zero weight.
        self.hidden = _padded(hidden)
        self.intermediate = _padded(intermediate)
        self.format = format
        self.operators: dict[int, _SplitOperator] = {}
        self.step = None
        self.yield_turn = yield_turn or (lambda: None)

        self.native = load()
        alignment = self.native.get_token_alignment_for_mega_moe()
        self.capacity = (
            (self.max_tokens + alignment - 1) // alignment * alignment
        )
        _, _, views = self.native.get_symm_buffer_size_for_mega_moe_m2n(
            self.attention_ranks,
            self.expert_ranks,
            self.num_experts,
            self.capacity,
            self.top_k,
            self.hidden,
            self.intermediate,
            format == "nvfp4",
        )
        # Rust owns the storage and every peer mapping; members appear in
        # union order.
        self.storage = symmetric["storage"]
        self.pointers = [int(pointer) for pointer in symmetric["pointers"]]
        if len(self.pointers) != group.size:
            raise ValueError("symmetric storage maps every union member")
        self.storage.zero_()
        # The next launch waits for its predecessor's cleanup phase. Seed
        # the initial completed phase before publishing this buffer to peers.
        self.storage[20:24].view(torch.int32).fill_(self.group.size)
        torch.cuda.current_stream(self.device).synchronize()
        bootstrap.barrier()
        self.x, self.scales, self.ids, self.weights, *_ = views(self.storage)

    def prepare(self, module, size):
        """Bind a numerical layer and compile its launches without execution."""
        quantizer = module.up_gate.input_quantizer
        if quantizer is None or quantizer.format != self.format:
            raise ValueError("one exchange must share one activation encoding")
        if module.down.input_quantizer is None or (
            module.down.input_quantizer.format != quantizer.format
        ):
            raise ValueError(
                "both expert projections must use the same encoding"
            )
        if quantizer.format == "nvfp4" and (
            quantizer.calibrated_scale is None
            or module.down.input_quantizer.calibrated_scale is None
        ):
            raise ValueError("NVFP4 experts require calibrated input scales")
        operator = _SplitOperator(
            module=module, size=size, workspace={}, buffer=self
        )
        self.operators[id(module)] = operator
        return operator

    def bind_microbatches(self, buffers):
        """Prepare one expert launch for all layers and microbatches.

        A persistent grid occupies the expert device. Launching independent
        grids on different streams can make peers serve different
        microbatches and wait forever. One launch visits layer-major, then
        microbatch order on every expert rank, matching the attention calls'
        cooperative order.
        """
        # `mega_moe_m2n_eg` takes a fixed four microbatch buffers, so the
        # kernel serves at most four microbatches.
        if not 1 <= len(buffers) <= 4:
            raise ValueError("split MegaMoE serves one to four microbatches")
        if any(
            tuple(buffer.operators) != tuple(self.operators)
            or (buffer.group, buffer.format, buffer.max_tokens)
            != (self.group, self.format, self.max_tokens)
            for buffer in buffers
        ):
            raise ValueError("microbatches must share an expert layer sequence")
        if not self.source:
            step = _Step(buffers)
            for buffer in buffers:
                buffer.step = step

    def attention(
        self, output, hidden, ids, weights, *, phase, compile_only=False
    ):
        self.native.mega_moe_m2n_ag(
            output,
            hidden,
            ids,
            weights,
            None,
            True,
            self.storage,
            self.pointers,
            self.group.rank,
            self.attention_ranks,
            self.expert_ranks,
            self.capacity,
            self.num_experts,
            self.top_k,
            self.intermediate,
            0,
            phase,
            True,
            False,
            self.format == "nvfp4",
            self.top_k,
            self.num_experts,
            0.0,
            1.0,
            None,
            compile_only,
        )

    def expert(self, operator, *, microbatches=(), compile_only=False):
        # The other microbatches' buffers fill the kernel's three further
        # buffer slots; an unused slot passes no storage and this buffer's
        # pointers.
        buffers = (*microbatches, *((None,) * (3 - len(microbatches))))
        self.native.mega_moe_m2n_eg(
            operator.descriptors,
            operator.pointers,
            operator.descriptors.numel() // (4 * 128),
            self.num_experts // self.expert_ranks,
            self.hidden,
            self.intermediate,
            operator.weight_bytes,
            self.format == "mxfp8",
            self.format == "nvfp4",
            None,
            operator.alpha,
            self.storage,
            *(None if buffer is None else buffer.storage for buffer in buffers),
            self.pointers,
            *(
                self.pointers if buffer is None else buffer.pointers
                for buffer in buffers
            ),
            self.group.rank,
            self.attention_ranks,
            self.expert_ranks,
            self.capacity,
            self.num_experts,
            self.top_k,
            self.max_tokens,
            None,
            None,
            None,
            None,
            True,
            0,
            0,
            None,
            compile_only,
        )

    def close(self):
        """Retire collectively after every caller drains its local readers."""
        if self.storage is None:
            return
        # A local stream drain does not retire a peer's remote memory reads.
        # Every owner reaches this host barrier after draining its own work.
        self.bootstrap.barrier()
        self.x = self.scales = self.ids = self.weights = None
        self.storage = None
        self.step = None
        self.operators.clear()


class _Step:
    """Native descriptors for a complete layer/microbatch expert sequence."""

    def __init__(self, buffers):
        self.buffers = tuple(buffers)
        self.operators = tuple(buffers[0].operators.values())
        self.descriptors = torch.stack(
            [op.descriptors for op in self.operators]
        )
        self.pointers = torch.cat([op.pointers for op in self.operators])
        self.weight_bytes = self.operators[0].weight_bytes
        self.alpha = (
            None
            if self.operators[0].alpha is None
            else torch.stack([op.alpha for op in self.operators])
        )
        buffers[0].expert(self, microbatches=buffers[1:], compile_only=True)

    def launch(self, operator):
        if operator is self.operators[0]:
            self.buffers[0].expert(self, microbatches=self.buffers[1:])


def _weights(linear, order, rows, columns, *, halves=1):
    """An expert projection's kernel operands: values and MN-major scales.

    Returns the encoded values ``[E, rows, columns / 2]`` (NVFP4) or
    ``[E, rows, columns]`` (MXFP8) in ``order`` and the block scales as
    ``[E, columns / 128, rows]`` 32-bit words of four scales, the MN-major
    order UTCCP's 128-row tiles load. A weight of the kernels' padded width
    is stored that way (``weight_layout``) and borrowed; a narrower one is
    copied into zero-padded operands, its up and gate halves padded apart.
    """
    weight = linear.weight
    if not isinstance(weight, QuantizedTensor):
        raise ValueError("split MegaMoE requires encoded expert weights")
    if weight.quantizer.format not in {"mxfp8", "nvfp4"}:
        raise ValueError("split MegaMoE requires MXFP8 or NVFP4 expert weights")
    if weight.shape[1:] != (rows, columns):
        original = weight.repack(
            row_order=RowOrder.LINEAR, scale_layout=ScaleLayout.LINEAR
        )
        experts, original_rows, _ = original.shape
        padded = weight.quantizer.empty(
            (experts, rows, columns), dtype=weight.dtype, device=weight.device
        )
        fields = dict(padded.buffers())
        for name, field in original.buffers().items():
            if name == "tensor_scale":
                fields[name] = field
                continue
            source = field.reshape(experts, original_rows, -1)
            target = fields[name].reshape(experts, rows, -1)
            target.zero_()
            # Up and gate occupy separate logical halves. Move each half
            # independently so padding cannot shift their pairing.
            source_rows, target_rows = original_rows // halves, rows // halves
            for half in range(halves):
                target[
                    :,
                    half * target_rows : half * target_rows + source_rows,
                    : source.shape[-1],
                ].copy_(
                    source[:, half * source_rows : (half + 1) * source_rows]
                )
        placed = weight.quantizer.from_tensors(
            fields, shape=(experts, rows, columns), dtype=weight.dtype
        ).repack(row_order=order, scale_layout=ScaleLayout.SWIZZLED_128X4_MN)
    else:
        placed = weight
    fields = placed.buffers()
    experts, rows, _ = placed.shape
    nvfp4 = placed.quantizer.format == "nvfp4"
    scales = (
        fields["block_scale" if nvfp4 else "scale"]
        .view(torch.int32)
        .reshape(experts, -1, rows)
    )
    values = fields["values"]
    return values.view(torch.int8 if nvfp4 else torch.float8_e4m3fn), scales


class _SplitOperator(Operator):
    def __init__(self, *, buffer, **kwargs):
        super().__init__(**kwargs)
        self.buffer = buffer
        self.alpha = None
        module = self.module
        if buffer.source:
            shape = (buffer.max_tokens, buffer.hidden)
            hidden = torch.empty(
                shape, dtype=torch.bfloat16, device=buffer.device
            )
            # Provision the public activation encoder before a peer can enter
            # the persistent expert kernel's device waits.
            module.up_gate.input_quantizer.quantize(
                hidden[:, : buffer.hidden_size].zero_()
            )
            ids = torch.empty(
                (buffer.max_tokens, buffer.top_k),
                dtype=torch.int32,
                device=buffer.device,
            )
            weights = torch.empty_like(ids, dtype=torch.float32)
            for phase in (1, 2):
                buffer.attention(
                    hidden, hidden, ids, weights, phase=phase, compile_only=True
                )
            return

        verify(module, weight_layout(module), "split MegaMoE")
        fc1 = _weights(
            module.up_gate,
            _ORDERS["up_gate"],
            2 * buffer.intermediate,
            buffer.hidden,
            halves=2,
        )
        fc2 = _weights(
            module.down, _ORDERS["down"], buffer.hidden, buffer.intermediate
        )
        for projection in (module.up_gate, module.down):
            if projection.weight.quantizer.format != buffer.format:
                raise ValueError(
                    "expert weights and activations must share an encoding"
                )
        # ``close`` releases the weight operands and their pointers.
        self.operands: tuple[Any, Any] | None = (fc1, fc2)
        self.descriptors = buffer.native.make_m2n_weight_descs(
            *fc1,
            *fc2,
            module.up_gate.num_experts,
            buffer.hidden,
            buffer.intermediate,
            16 if buffer.format == "nvfp4" else 32,
        ).to(buffer.device)
        self.pointers: torch.Tensor | None = torch.tensor(
            [fc1[0].data_ptr()], dtype=torch.int64, device=buffer.device
        )
        self.weight_bytes = fc1[0].numel() * fc1[0].element_size()
        if buffer.format == "nvfp4":
            first = module.up_gate.weight.buffers()["tensor_scale"].float()
            second = module.down.weight.buffers()["tensor_scale"].float()
            input_scale = module.up_gate.input_quantizer.calibrated_scale
            intermediate_scale = module.down.input_quantizer.calibrated_scale
            experts = module.up_gate.num_experts
            self.alpha = torch.stack(
                (
                    first.reshape(-1).expand(experts) * input_scale,
                    torch.full(
                        (experts,), 1 / intermediate_scale, device=buffer.device
                    ),
                    second.reshape(-1).expand(experts) * intermediate_scale,
                ),
                dim=-1,
            ).contiguous()
        buffer.expert(self, compile_only=True)

    def __call__(self, hidden, topk_ids, topk_weights, *, combine=True):
        self._validate(hidden, topk_ids, topk_weights)
        buffer = self.buffer
        count = hidden.shape[0]
        if count > buffer.max_tokens:
            raise ValueError("routed tokens exceed the split MegaMoE capacity")
        if not buffer.source:
            if buffer.step is None:
                buffer.expert(self)
            else:
                buffer.step.launch(self)
            buffer.yield_turn()
            return hidden.new_empty((0, buffer.hidden_size))

        if count:
            encoded = (
                hidden
                if isinstance(hidden, QuantizedTensor)
                else self.module.up_gate.input_quantizer.quantize(hidden)
            )
            fields = encoded.buffers()
            values = fields["values"].view(torch.uint8)
            buffer.x[:count, : values.shape[-1]].view(torch.uint8).copy_(values)
            scales = fields[
                "block_scale" if buffer.format == "nvfp4" else "scale"
            ]
            columns = buffer.hidden_size // (
                16 if buffer.format == "nvfp4" else 32
            )
            buffer.scales[:count].view(torch.uint8)[:, :columns].copy_(
                scales.reshape(count, columns)
            )
            buffer.weights[:count].copy_(topk_weights)
        output = torch.empty(
            (count, buffer.hidden), dtype=torch.bfloat16, device=buffer.device
        )
        # external_quant consumes the encoded buffer, so hidden's argument
        # only supplies the BF16 shape expected by the native launch contract.
        buffer.attention(output, output, topk_ids, topk_weights, phase=1)
        buffer.yield_turn()
        buffer.attention(output, output, topk_ids, topk_weights, phase=2)
        # Strip physical channel padding before exposing numerical rows.
        return output[:, : buffer.hidden_size].contiguous()

    def close(self):
        super().close()
        self.buffer = self.operands = self.descriptors = self.pointers = None
        self.alpha = None
