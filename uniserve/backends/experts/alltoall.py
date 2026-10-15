"""FlashInfer's NVLink all-to-all around each rank's grouped experts.

Every rank maps one workspace of its peers' memory (MNNVL), sized for
``max_tokens`` tokens per rank; FlashInfer caches it process-wide. A
dispatch writes each token's payload (states, expert ids, route weights)
into the slots of the ranks holding its experts; a combine sums the FP32
partial results of each token and rounds once to the output dtype.
"""

from __future__ import annotations

import torch

from uniserve.distributed import Communicator
from uniserve.quantization import QuantizedTensor, ScaleLayout

from . import ExpertTransport


class AllToAll(ExpertTransport):
    """One microbatch's all-to-all workspace over the expert group.

    Construction is collective over ``group``: the workspace maps peer
    memory, exchanging handles through ``bootstrap``. The first call of the
    transport needs no host barrier.
    """

    transport = "alltoall"

    def __init__(
        self,
        group: Communicator,
        bootstrap,
        *,
        max_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        device: torch.device,
        attention_ranks: int = 0,
        yield_turn=None,
    ) -> None:
        if attention_ranks:
            raise ValueError("the all-to-all serves colocated experts only")
        super().__init__(
            group,
            bootstrap,
            max_tokens=max_tokens,
            attention_ranks=0,
            barrier_first=False,
            yield_turn=yield_turn,
        )
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig
        from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll

        # FlashInfer's Mapping describes expert parallelism inside a
        # TP-sized container; attention stays data-parallel on every rank.
        mapping = Mapping(
            world_size=group.size,
            rank=group.rank,
            gpus_per_node=torch.cuda.device_count(),
            tp_size=group.size,
            moe_tp_size=1,
            moe_ep_size=group.size,
            enable_attention_dp=True,
        )
        with torch.cuda.device(device):
            self._exchange = MoeAlltoAll(
                mapping,
                max_tokens,
                top_k,
                num_experts,
                hidden_size=hidden_size,
                mnnvl_config=MnnvlConfig(comm_backend=bootstrap),
            )

    def dispatch(self, hidden, topk_ids, topk_weights, *, invalid_expert):
        """Route ``hidden`` rows by ``topk_ids``.

        ``hidden`` is a dense ``[T, H]`` tensor or a calibrated NVFP4
        ``QuantizedTensor`` with linear block scales: an encoded row travels
        as its packed E2M1 values and E4M3 block scales (9/32 of its BF16
        bytes) and arrives in the same encoding; the tensor scale is the
        static calibration every rank shares, so it stays local. Every rank
        of a step sends a layer's rows in one representation, because the
        all-to-all pairs the ranks' payload lists. Returns ``[P * C, ...]``
        rows for ``P`` ranks and the step capacity ``C``.
        """
        if hidden.shape[0] > self.capacity:
            raise ValueError("a rank's tokens exceed the step capacity")
        encoded = isinstance(hidden, QuantizedTensor)
        if encoded:
            quantizer = hidden.quantizer
            if (
                quantizer.format != "nvfp4"
                or quantizer.calibrated_scale is None
                or hidden.scale_layout is not ScaleLayout.LINEAR
            ):
                raise ValueError(
                    "encoded hidden states are exchanged as calibrated NVFP4 "
                    "with linear block scales"
                )
            fields = hidden.buffers()
            states = [fields["values"], fields["block_scale"]]
        else:
            states = [hidden]

        # The states, then the ids the all-to-all routes by, then the
        # weights, as SGLang's NVFP4 dispatch orders them.
        received = [
            value.flatten(0, 1)
            for value in self._exchange.dispatch(
                topk_ids,
                [*states, topk_ids, topk_weights],
                self.capacity,
                invalid_token_expert_id=invalid_expert,
                expert_id_payload_index=len(states),
            )
        ]
        ids, weights = received[-2:]
        if not encoded:
            return received[0], ids, weights
        return (
            quantizer.from_tensors(
                {
                    "values": received[0],
                    "block_scale": received[1],
                    "tensor_scale": fields["tensor_scale"],
                },
                shape=(ids.shape[0], hidden.shape[1]),
                dtype=hidden.dtype,
            ),
            ids,
            weights,
        )

    def combine(self, partial: torch.Tensor, tokens: int) -> torch.Tensor:
        output = self._exchange.combine(
            partial.view(self.group.size, self.capacity, partial.shape[-1]),
            self.capacity,
            payload_in_workspace=False,
            output_dtype=partial.dtype,
            use_low_precision=False,
        )
        if output.shape[0] != tokens:
            raise RuntimeError("the combine returned another token count")
        return output
