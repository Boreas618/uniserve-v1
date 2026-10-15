"""The fused MegaMoE transport.

MegaMoE exchanges tokens, runs the experts and combines in one launch per
layer, so the transport holds the fused kernel's buffer instead of
dispatching and combining: colocated experts share NVSHMEM symmetric
buffers (``backends.moe.megamoe``); dedicated expert ranks run split
MegaMoE over DeepGEMM (:mod:`.split`), on symmetric storage the serving
runtime maps.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from uniserve.distributed import Communicator
from uniserve.nn.moe import Activation

from . import ExpertTransport


class FusedExperts(ExpertTransport):
    """One microbatch's fused expert buffer over the expert group.

    Construction is collective over ``group``. With dedicated expert ranks
    the first call waits on a host barrier, so a source's preceding
    compilation cannot consume a device collective timeout.
    """

    transport = "megamoe"

    def __init__(
        self,
        group: Communicator,
        bootstrap,
        *,
        max_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Activation,
        device: torch.device,
        attention_ranks: int = 0,
        encoding: str | None = None,
        symmetric: Mapping[str, Any] | None = None,
        yield_turn=None,
    ) -> None:
        super().__init__(
            group,
            bootstrap,
            max_tokens=max_tokens,
            attention_ranks=attention_ranks,
            barrier_first=bool(attention_ranks),
            yield_turn=yield_turn,
        )
        if attention_ranks:
            from .split import Buffer

            if symmetric is None or encoding is None:
                raise ValueError(
                    "split MegaMoE requires symmetric storage and an encoding"
                )
            self.fused = Buffer(
                group,
                bootstrap,
                max_tokens=max_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden=hidden_size,
                intermediate=intermediate_size,
                activation=activation,
                attention_ranks=attention_ranks,
                device=device,
                format=encoding,
                symmetric=symmetric,
                yield_turn=self.yield_turn,
            )
        else:
            from ..moe.megamoe import MegaMoEBuffer

            self.fused = MegaMoEBuffer(
                group,
                bootstrap,
                max_tokens=max_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden=hidden_size,
                intermediate=intermediate_size,
                activation=activation,
                device=device,
            )

    def bind_microbatches(self, transports) -> None:
        if not self.attention_ranks:
            return
        if any(
            transport.transport != self.transport
            or transport.attention_ranks != self.attention_ranks
            for transport in transports
        ):
            raise ValueError("microbatches must use the same expert transport")
        self.fused.bind_microbatches([t.fused for t in transports])

    def close(self) -> None:
        # The colocated NVSHMEM heap stays mapped for the process lifetime.
        if self.attention_ranks and self.fused is not None:
            self.fused.close()
        self.fused = None
