"""The megamoe module's exported launch of FlashInfer's MegaMoE kernel.

The build compiles :class:`Launcher` with the CuTe DSL and exports it to C
(`export_to_c`) as `cute_dsl_mega_wrapper`, which `entry.cpp` calls once per
layer. The kernel reaches a peer's buffer by adding that peer's offset to the
local address (the pinned `src.sym_buffer.SymBufferHost`). The offsets arrive
as one tuple of 64-bit integers, one per rank, which the exported C function
takes as one `int64_t *` array: the same launcher serves every group size, and
the group's size is the tuple's length the build traces it with.

The pinned kernel's packages must be imported (`fetch.load`) before this
module, which names their `src` package.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from src.sym_buffer import SymBufferHost


class Launcher:
    """Launches the configured kernel over its persistent grid."""

    def __init__(self, kernel, clusters):
        #: The configured `Sm100MegaMoEKernel`.
        self.kernel = kernel
        #: The clusters of the kernel's persistent grid, which the kernel
        #: takes as a compile-time constant.
        self.clusters = clusters

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_norm_const: cute.Tensor,
        output_activation: cute.Tensor,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        # A tuple of `cutlass.Int64`, the byte distance from this rank's
        # window to each rank's, in rank order.
        offsets,
        rank: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        peers = SymBufferHost(
            # The kernel reads the offsets and the rank; it does not use the
            # payload's base address.
            base_addr=cutlass.Int64(0),
            offsets=offsets,
            rank_idx=rank,
            num_max_ranks=len(offsets),
        )
        self.kernel(
            activation,
            activation_sf,
            topk_idx,
            topk_weights,
            fc1_weight,
            fc1_weight_sf,
            fc2_weight,
            fc2_weight_sf,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            output_activation,
            local_workspace,
            shared_workspace,
            peers,
            self.clusters,
            stream,
        )
