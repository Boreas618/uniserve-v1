# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Derived from FlashInfer's MoE grouped GEMMs as `uniserve_kernels.grouped_gemm`
# states, which holds the computation this kernel shares with route_gemm's.

"""BF16 gather grouped GEMM with gated activation for routed experts (SM100).

FC1 of a mixture of experts over FlashInfer's MoE sort metadata: every
tile of permuted (token, route) rows belongs to one expert; the kernel
gathers the tile's token rows of the BF16 hidden states ``A [T, K]`` with
LDGSTS through ``token_id_mapping``, multiplies them by that expert's
BF16 ``B [N, K]`` rows with tcgen05 MMA accumulating in FP32 in tensor
memory, applies the gated activation to the FP32 products of each up and
gate column pair and stores the product in BF16 at the permuted row.

- ``B`` holds each expert's up rows followed by its gate rows. The
  kernel reads them in 64-row blocks interleaved as
  ``[up_0:64, gate_0:64, up_64:128, gate_64:128, ...]`` through a strided
  TMA view, so each 128-column block holds matching up and gate columns and
  ``C`` is ``[M, N / 2]``.
- Warp roles: epilogue (0-3), LDGSTS A (4-7), MMA (8), TMA B (9),
  scheduler (10), and with two-CTA MMA an A-arrival relay (11). The
  scheduler, the B load, the MMA and the tensor memory are
  :class:`uniserve_kernels.grouped_gemm.GroupedGemm`'s.
- Tiles at or past ``num_non_exiting_tiles`` are skipped, so padded
  metadata and outputs of a larger capacity are safe under CUDA graphs.
  Rows at or past a tile's ``mn_limit`` are not gathered; the tile's TMA
  store still writes them, with unspecified values, which no valid row
  depends on.

Every output element is one FP32 accumulation over K in a fixed order
followed by the activation and one BF16 rounding, so results are
independent of tile scheduling and repeat bit for bit.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from flashinfer.fused_moe.cute_dsl.blackwell.custom_pipeline import (
    PipelineCpAsyncUmma,
)
from flashinfer.fused_moe.cute_dsl.blackwell.utils import (
    gelu_tanh_f32,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)
from uniserve_kernels.grouped_gemm import GroupedGemm


class GatherGroupedGemmKernel(GroupedGemm):
    """Gather grouped GEMM of routed tokens with gated activation (FC1).

    ``mma_tiler_mn`` is the MMA tile (M, N): M 128 runs one-CTA MMA, M 256
    two-CTA MMA over a cluster pair; N is 128 or 256, one or two pairs of
    64-column up and gate blocks. The MoE sort's routing tile equals M.
    ``activation`` is
    ``"silu"`` (``silu(gate) * up``) or ``"gelu_tanh"``
    (``gelu_tanh(gate) * up``). ``topk`` converts the expanded
    (token, route) ids of ``token_id_mapping`` to token rows. The target's
    capacities are :class:`GroupedGemm`'s.
    """

    def __init__(
        self,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        topk: int,
        activation: str,
        smem_capacity: int,
        tmem_columns: int,
    ):
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"unsupported gated activation {activation!r}")
        super().__init__(
            mma_tiler_mn, cluster_shape_mn, smem_capacity, tmem_columns
        )
        self.activation = activation
        self.topk = topk
        # Each output column combines one up and one gate accumulator column.
        self.out_n_factor = 2

        self.epilog_warp_id = (0, 1, 2, 3)
        self.ldgsts_a_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        self.sync_transform_warp_id = 11
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                *self.ldgsts_a_warp_id,
                self.tma_b_warp_id,
                *self.epilog_warp_id,
                self.sched_warp_id,
                self.sync_transform_warp_id,
            )
        )
        self.warps_wo_sched = (
            len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    self.sync_transform_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
            if self.use_2cta_instrs
            else len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
        )
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched
        self._make_barriers()

    def _setup_attributes(self):
        """Derive the tiled MMA, tile shapes, stages and layouts from the
        operand dtypes and the kernel's tile configuration.
        """  # noqa: D205
        # A K tile is 128 bytes of each A row, the row width the LDGSTS
        # gather below copies per K tile.
        tiled_mma = self._setup_mma_tiles()
        self.mma_tiler_c = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1] // self.out_n_factor,
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        self._setup_cluster_layout(tiled_mma)

        # Epilogue subtile: 128 rows of 64 output columns (one up and one
        # gate accumulator subtile each).
        self.epi_tile = (128, 64)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile[1],
        )

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.num_smem_capacity,
            self.occupancy,
        )

        self._setup_ab_smem_layouts(tiled_mma)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Two accumulator stages of N FP32 columns each fit tensor memory's
        # 512 columns for N <= 256, so the next tile's MMA overlaps this
        # tile's epilogue.
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
        )

        # Each LDGSTS.128 copies 8 BF16 values; a K tile row is 8 copies.
        self.a_elements_per_ldgsts = 128 // self.a_dtype.width

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        multiprocessors: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Launch the kernel on ``stream``, of ``multiprocessors``.

        ``a`` is the ``(T, K, 1)`` hidden states, gathered by
        ``token_id_mapping_tensor[row] // topk`` for every permuted row below
        its tile's ``mn_limit``; ``b`` the ``(N, K, E)`` expert weights in
        interleaved 64-row up/gate block order; ``c`` the ``(M, N / 2, 1)``
        permuted output rows.
        """
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.c_dtype: type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        tiled_mma = self._make_tiled_mma()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for B
        tma_atom_b, tma_tensor_b, b_copy_size = self._make_tma_b(tiled_mma, b)
        self.num_tma_load_bytes = b_copy_size * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(
            self.c_smem_layout_staged, (None, None, 0)
        )
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size: the persistent schedule covers C's CTA tiles.
        c_shape = cute.slice_(self.cta_tile_shape_mnk_c, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        self.tile_sched_params, grid = self._compute_grid(
            num_ctas_mnl,
            self.cluster_shape_mn,
            multiprocessors,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage1cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            s_info: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            b_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            s_c: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            s_a: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            s_b: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorage2cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            s_info: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            a_sync_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            b_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            s_c: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            s_a: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            s_b: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = (
            SharedStorage2cta
            if cutlass.const_expr(self.use_2cta_instrs)
            else SharedStorage1cta
        )

        # Launch the kernel synchronously
        kernel = self.kernel(
            tiled_mma,
            a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        )
        self._launch(kernel, grid, stream)
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        m_a_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        m_b_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        m_c_mnl: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout | None,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """The persistent warp-specialized grouped GEMM (see the class)."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        (
            use_2cta_instrs,
            mma_tile_coord_v,
            is_leader_cta,
            block_in_cluster_coord_vmnk,
            tidx,
            storage,
        ) = self._cta_coordinates(tiled_mma, cluster_layout_vmnk)

        # Pipeline Init: Initialize A pipeline for LDGSTS operations
        # Producer: 4 warps (warps 4-7) with 128 threads total for LDGSTS
        # operations
        # Consumer: MMA warp for consuming A data
        a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 4,
        )

        a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize A SYNC Transform pipeline when
        # use_2cta_instrs is True
        # Producer: 1 warp (warp 11) for LDGSTS SYNC transformation operations
        # Consumer: MMA warp for consuming A data
        if cutlass.const_expr(self.use_2cta_instrs):
            a_sync_transform_pipeline_producer_group = (
                pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * cute.size(cluster_layout_vmnk, mode=[0]),
                )
            )
            a_sync_transform_pipeline = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.a_sync_transform_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=a_sync_transform_pipeline_producer_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        # Pipeline Init: Initialize B pipeline for TMA operations
        # PipelineTmaUmma for B, a TMA load with multicast support
        # Producer: TMA B warp (warp 9) - 1 warp issuing TMA operations
        # Consumer: MMA warp for consuming B data
        b_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        num_tma_producer = self.num_mcast_ctas_b
        b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,  # Bytes TMA loads per stage (B)
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        acc_pipeline, tile_info_pipeline = (
            self._make_acc_and_tile_info_pipelines(
                storage, cluster_layout_vmnk, use_2cta_instrs
            )
        )
        tmem = self._make_tmem_allocator(storage, use_2cta_instrs)

        # Cluster arrive after barrier init
        self._cluster_arrive()

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        s_c = storage.s_c.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        s_a, s_b = self._ab_smem_tensors(
            storage, a_smem_layout_staged, b_smem_layout_staged
        )
        s_info = self._tile_info_smem(storage)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = self._b_full_mcast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        g_a_mkl = cute.local_tile(
            m_a_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        g_b_nkl = self._local_tile_b(m_b_nkl)

        g_token_ml = cute.local_tile(
            token_id_mapping_tensor,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
            (None,),
        )

        # (bM, bN, loopM, loopN, loopL)
        g_c_mnl = cute.local_tile(
            m_c_mnl,
            cute.slice_(self.mma_tiler_c, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cutlass.Int32(cute.size(g_a_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        t_cg_b = thr_mma.partition_B(g_b_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        t_cg_c = thr_mma.partition_C(g_c_mnl)

        #
        # Partition the TMA load of B and the MMA's fragments
        #
        t_bs_b, t_bg_b, t_cr_a, t_cr_b, t_ct_acc_fake = (
            self._partition_b_and_fragments(
                tiled_mma,
                tma_atom_b,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                s_a,
                s_b,
                t_cg_b,
            )
        )

        #
        # Cluster wait before tensor memory alloc
        #
        self._cluster_wait()

        griddepcontrol_wait()

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            # Tiles at or past num_non_exiting_tiles are padding; clusters
            # raster along N, so the first such M tile ends the schedule.
            is_continue = cutlass.Boolean(1)
            while work_tile.is_valid_tile and is_continue:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape
                )
                if mma_tile_coord_m < num_non_exiting_tiles_value:
                    tile_info_pipeline.producer_acquire(
                        tile_info_producer_state
                    )
                    cur_tile_coord = work_tile.tile_idx
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                    self._publish_tile(
                        s_info,
                        tile_info_producer_state,
                        cur_tile_coord,
                        expert_idx,
                        work_tile.is_valid_tile,
                        mn_limit,
                    )

                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                else:
                    is_continue = cutlass.Boolean(0)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # The end of the schedule: an invalid tile.
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            self._publish_tile(
                s_info, tile_info_producer_state, work_tile.tile_idx, -1, 0, -1
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized LDGSTS A warps (warps 4-7)
        # These warps use LDGSTS instructions to load A from global to shared
        # memory
        # with gather/permutation capability enabled by token_id_mapping
        #
        if (
            warp_idx <= self.ldgsts_a_warp_id[-1]
            and warp_idx >= self.ldgsts_a_warp_id[0]
        ):
            #
            # Setup the LDGSTS copy atom for A: 8x LDGSTS.128 per thread with
            # swizzle_128B, 8 BF16 values each.
            #
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cpasync.LoadCacheMode.GLOBAL
                ),
                m_a_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout(
                (1, self.a_elements_per_ldgsts),
                stride=(self.a_elements_per_ldgsts, 1),
            )
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )

            tidx_in_warpgroup = tidx % 128

            s_a_tiled = cute.make_tensor(
                s_a.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        self.cta_tile_shape_mnk[2],
                        self.num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            t_as_a_tiled = a_thr_copy.partition_D(s_a_tiled)

            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)),
                cutlass.Boolean,
            )
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            is_valid_tile = self._read_tile_info(
                tile_info_pipeline,
                tile_info_consumer_state,
                s_info,
                tile_info,
                (0, 3, 4),
            )
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Load token IDs for the gather: each thread loads 8 token
                # offsets, one per LDGSTS.128 row.
                g_token_ml_tile = g_token_ml[(None, tile_info[0])]
                for i in range(8):
                    token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                    a_token_offset_tensor[i] = g_token_ml_tile[
                        token_ml_tile_offset
                    ]
                    a_predicate_tensor[i] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + token_ml_tile_offset
                        < tile_info[4]
                        else cutlass.Boolean(0)
                    )
                    a_token_offset_tensor[i] = (
                        a_token_offset_tensor[i] // self.topk
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + token_ml_tile_offset
                        < tile_info[4]
                        else 0
                    )

                t_ag_a = g_a_mkl[(None, None, 0, None, 0)]
                a_gmem_thread_offset = cute.assume(
                    (tidx_in_warpgroup % 8) * self.a_elements_per_ldgsts,
                    divby=self.a_elements_per_ldgsts,
                )

                # Peek (try_wait) A buffer empty
                a_producer_state.reset_count()
                peek_a_empty_status = cutlass.Boolean(1)
                if a_producer_state.count < k_tile_cnt:
                    peek_a_empty_status = a_pipeline.producer_try_acquire(
                        a_producer_state
                    )

                #
                # Load A with LDGSTS and gather: each K-tile iteration loads
                # one K tile of the tile's token rows from GMEM to SMEM.
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    # Conditionally wait for AB buffer empty
                    a_pipeline.producer_acquire(
                        a_producer_state, peek_a_empty_status
                    )

                    t_ag_a_ktile = t_ag_a[(None, None, a_producer_state.count)]
                    t_as_a_ktile = t_as_a_tiled[
                        (None, None, None, a_producer_state.index)
                    ]

                    for i in range(8):
                        #
                        # Load A matrix: 8x LDGSTS.128 per thread with
                        # swizzle_128B
                        # Each LDGSTS.128 loads a_elements_per_ldgsts values.
                        # Global memory address is computed using token offset
                        # for gather operation
                        # Predicate mask guards against invalid token IDs
                        # (padding tokens marked as -1)
                        #
                        a_gmem_slice_offset = (
                            a_gmem_thread_offset
                            + cute.assume(
                                a_token_offset_tensor[i]
                                * t_ag_a_ktile.layout[0].stride,
                                divby=self.a_elements_per_ldgsts,
                            )
                        )
                        a_gmem_slice_offset = cute.assume(
                            a_gmem_slice_offset,
                            divby=self.a_elements_per_ldgsts,
                        )
                        t_ag_a_slice_ptr = (
                            t_ag_a_ktile.iterator + a_gmem_slice_offset
                        )
                        t_ag_a_slice = cute.make_tensor(
                            t_ag_a_slice_ptr,
                            layout=cute.make_layout(
                                (self.a_elements_per_ldgsts,)
                            ),
                        )

                        t_as_a_slice = cute.make_tensor(
                            t_as_a_ktile[(None, i, None)].iterator,
                            layout=cute.make_layout(
                                (self.a_elements_per_ldgsts,)
                            ),
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        a_predicate_slice[0] = a_predicate_tensor[i]

                        cute.copy_atom_call(
                            a_atom_copy,
                            t_ag_a_slice,
                            t_as_a_slice,
                            pred=a_predicate_slice,
                        )

                    a_pipeline.producer_commit(a_producer_state)

                    # Peek (try_wait) A buffer empty for k_tile =
                    # prefetch_k_tile_cnt + k_tile + 1
                    a_producer_state.advance()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(
                            a_producer_state
                        )

                #
                # Advance to next tile
                #
                is_valid_tile = self._read_tile_info(
                    tile_info_pipeline,
                    tile_info_consumer_state,
                    s_info,
                    tile_info,
                    (0, 3, 4),
                )
                tile_info_consumer_state.advance()

            #
            # Wait A pipeline buffer empty
            #
            a_pipeline.producer_tail(a_producer_state)

        #
        # Specialized A Sync Transform Warp (warp 11) when use_2cta_instrs is
        # True: relays the A arrival of both CTAs to the leader's MMA warp.
        #
        if warp_idx == self.sync_transform_warp_id:
            if cutlass.const_expr(self.use_2cta_instrs):
                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                )
                # First tile
                work_tile = tile_sched.initial_work_tile_info()

                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # Get the first tile info
                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                is_valid_tile = self._read_tile_info(
                    tile_info_pipeline,
                    tile_info_consumer_state,
                    s_info,
                    tile_info,
                    (3,),
                )
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    # Peek (try_wait) A buffer full for k_tile = 0
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(
                            a_consumer_state
                        )
                    # Peek (try_wait) a sync transform buffer empty
                    a_sync_transform_producer_state.reset_count()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for A buffer full
                        a_pipeline.consumer_wait(
                            a_consumer_state, peek_a_full_status
                        )

                        a_sync_transform_pipeline.producer_commit(
                            a_sync_transform_producer_state
                        )
                        a_sync_transform_producer_state.advance()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(
                                a_consumer_state
                            )

                    #
                    # Advance to next tile
                    #
                    is_valid_tile = self._read_tile_info(
                        tile_info_pipeline,
                        tile_info_consumer_state,
                        s_info,
                        tile_info,
                        (3,),
                    )
                    tile_info_consumer_state.advance()

                #
                # Wait A sync transform buffer empty
                #
                a_sync_transform_pipeline.producer_tail(
                    a_sync_transform_producer_state
                )

        #
        # Specialized TMA B load warp (warp 9): loads B from global to shared
        # memory with multicast support to reduce L2 memory traffic
        #
        if warp_idx == self.tma_b_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            is_valid_tile = self._read_tile_info(
                tile_info_pipeline,
                tile_info_consumer_state,
                s_info,
                tile_info,
                (0, 1, 2, 3),
            )
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                t_bg_b_slice = t_bg_b[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile =
                # prefetch_k_tile_cnt
                b_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = b_pipeline.producer_try_acquire(
                        b_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    # Conditionally wait for B buffer empty
                    b_pipeline.producer_acquire(
                        b_producer_state, peek_ab_empty_status
                    )

                    t_bg_b_k = t_bg_b_slice[(None, b_producer_state.count)]
                    t_bs_b_pipe = t_bs_b[(None, b_producer_state.index)]

                    tma_bar = b_pipeline.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        t_bg_b_k,
                        t_bs_b_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile =
                    # prefetch_k_tile_cnt + k_tile + 1
                    b_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = b_pipeline.producer_try_acquire(
                            b_producer_state
                        )

                #
                # Advance to next tile
                #
                is_valid_tile = self._read_tile_info(
                    tile_info_pipeline,
                    tile_info_consumer_state,
                    s_info,
                    tile_info,
                    (0, 1, 2, 3),
                )
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            b_pipeline.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            # (MMA, MMA_M, MMA_N, STAGE)
            _, t_ct_acc_base = self._accumulator_base(tmem, t_ct_acc_fake)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )

            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out
            # tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            is_valid_tile = self._read_tile_info(
                tile_info_pipeline,
                tile_info_consumer_state,
                s_info,
                tile_info,
                (0, 1, 2, 3),
            )
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_consumer_state.reset_count()
                    peek_a_sync_transform_full_status = cutlass.Boolean(1)
                    if (
                        a_sync_transform_consumer_state.count < k_tile_cnt
                        and is_leader_cta
                    ):
                        peek_a_sync_transform_full_status = (
                            a_sync_transform_pipeline.consumer_try_wait(
                                a_sync_transform_consumer_state
                            )
                        )
                    a_consumer_state.reset_count()
                else:
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(
                            a_consumer_state
                        )

                b_consumer_state.reset_count()
                peek_b_full_status = cutlass.Boolean(1)
                if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_b_full_status = b_pipeline.consumer_try_wait(
                        b_consumer_state
                    )

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                t_ct_acc = t_ct_acc_base[
                    (None, None, None, acc_producer_state.index)
                ]

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_wait(
                                a_sync_transform_consumer_state,
                                peek_a_sync_transform_full_status,
                            )
                        else:
                            a_pipeline.consumer_wait(
                                a_consumer_state, peek_a_full_status
                            )
                        b_pipeline.consumer_wait(
                            b_consumer_state, peek_b_full_status
                        )

                        # t_ct_acc += t_cr_a * t_cr_b
                        num_kblocks = cute.size(t_cr_a, mode=[2])

                        for kblock_idx in cutlass.range(
                            num_kblocks, unroll_full=True
                        ):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                b_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                t_ct_acc,
                                t_cr_a[kblock_coord],
                                t_cr_b[kblock_coord],
                                t_ct_acc,
                            )
                            # Enable accumulate on t_ct_acc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        a_pipeline.consumer_release(a_consumer_state)
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_release(
                                a_sync_transform_consumer_state
                            )
                        b_pipeline.consumer_release(b_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.advance()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if a_sync_transform_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_a_sync_transform_full_status = (
                                    a_sync_transform_pipeline.consumer_try_wait(
                                        a_sync_transform_consumer_state
                                    )
                                )
                        a_consumer_state.advance()
                    else:
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(
                                a_consumer_state
                            )

                    b_consumer_state.advance()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_b_full_status = b_pipeline.consumer_try_wait(
                                b_consumer_state
                            )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                is_valid_tile = self._read_tile_info(
                    tile_info_pipeline,
                    tile_info_consumer_state,
                    s_info,
                    tile_info,
                    (0, 1, 2, 3),
                )
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            # (MMA, MMA_M, MMA_N, STAGE)
            tmem_ptr, t_ct_acc_base = self._accumulator_base(
                tmem, t_ct_acc_fake
            )

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                t_tr_t_acc_base,
                t_tr_r_acc_up,
                t_tr_r_acc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, t_ct_acc_base, t_cg_c, epi_tile, use_2cta_instrs
            )

            t_tr_r_c = cute.make_rmem_tensor(t_tr_r_acc_up.shape, self.c_dtype)
            tiled_copy_r2s, t_rs_r_c, t_rs_s_c = (
                self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, t_tr_r_c, epi_tidx, s_c
                )
            )
            (
                tma_atom_c,
                b_sg_s_c,
                b_sg_g_c_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, t_cg_c, epi_tile, s_c
            )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            is_valid_tile = self._read_tile_info(
                tile_info_pipeline,
                tile_info_consumer_state,
                s_info,
                tile_info,
                (0, 1, 2, 3),
            )
            tile_info_consumer_state.advance()

            log2_e = cutlass.Float32(1.4426950408889634)
            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                b_sg_g_c = b_sg_g_c_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                t_tr_t_acc = t_tr_t_acc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                t_tr_t_acc = cute.group_modes(
                    t_tr_t_acc, 3, cute.rank(t_tr_t_acc)
                )
                b_sg_g_c = cute.group_modes(b_sg_g_c, 1, cute.rank(b_sg_g_c))

                # Accumulator subtiles alternate up and gate columns: output
                # subtile j reads up subtile 2j and gate subtile 2j + 1.
                subtile_cnt = cute.size(t_tr_t_acc.shape, mode=[3])
                for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                    real_subtile_idx = subtile_idx // 2
                    t_tr_t_acc_mn_up = t_tr_t_acc[
                        (None, None, None, subtile_idx)
                    ]
                    t_tr_t_acc_mn_gate = t_tr_t_acc[
                        (None, None, None, subtile_idx + 1)
                    ]
                    cute.copy(tiled_copy_t2r, t_tr_t_acc_mn_up, t_tr_r_acc_up)
                    cute.copy(
                        tiled_copy_t2r, t_tr_t_acc_mn_gate, t_tr_r_acc_gate
                    )

                    # FP32 activation of each gate, times its up value.
                    acc_vec_up = t_tr_r_acc_up.load()
                    acc_vec_gate = t_tr_r_acc_gate.load()
                    t_compute = cute.make_rmem_tensor(
                        acc_vec_gate.shape, self.acc_dtype
                    )
                    for i in cutlass.range_constexpr(cute.size(t_tr_r_acc_up)):
                        gate = acc_vec_gate[i]
                        if cutlass.const_expr(self.activation == "gelu_tanh"):
                            activated = gelu_tanh_f32(gate, fastmath=True)
                        else:
                            # silu(g) = g / (1 + 2^(-g * log2(e)))
                            activated = gate * cute.arch.rcp_approx(
                                cute.math.exp2(-gate * log2_e, fastmath=True)
                                + cutlass.Float32(1.0)
                            )
                        t_compute[i] = acc_vec_up[i] * activated

                    acc_vec = tiled_copy_r2s.retile(t_compute).load()
                    t_rs_r_c.store(acc_vec.to(self.c_dtype))

                    num_prev_subtiles = num_prev_subtiles + 1
                    c_buffer = num_prev_subtiles % self.num_c_stage

                    cute.copy(
                        tiled_copy_r2s,
                        t_rs_r_c,
                        t_rs_s_c[(None, None, None, c_buffer)],
                    )
                    # Make the shared memory store visible to the TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            b_sg_s_c[(None, c_buffer)],
                            b_sg_g_c[(None, real_subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                is_valid_tile = self._read_tile_info(
                    tile_info_pipeline,
                    tile_info_consumer_state,
                    s_info,
                    tile_info,
                    (0, 1, 2, 3),
                )
                tile_info_consumer_state.advance()
            self._free_tensor_memory(tmem, tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        t_acc: cute.Tensor,
        g_c_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: cutlass.Boolean | bool,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """Partition the accumulators for tensor-memory-to-register loads.

        Returns the T2R tiled copy, this thread's partition of the staged
        accumulators, and register fragments for one up and one gate
        epilogue subtile.
        """
        tiled_copy_t2r, t_tr_t_acc, fragment = self._partition_accumulators(
            tidx,
            t_acc,
            g_c_mnl,
            epi_tile,
            use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        # (T2R, T2R_M, T2R_N)
        t_tr_r_acc_up = cute.make_rmem_tensor(fragment, self.acc_dtype)
        # (T2R, T2R_M, T2R_N)
        t_tr_r_acc_gate = cute.make_rmem_tensor(fragment, self.acc_dtype)
        return tiled_copy_t2r, t_tr_t_acc, t_tr_r_acc_up, t_tr_r_acc_gate

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        t_tr_r_c: cute.Tensor,
        tidx: cutlass.Int32,
        s_c: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition registers and shared memory for the C subtile store.

        Returns the R2S tiled copy, the register source ``t_tr_r_c``
        retiled for it and this thread's shared-memory destination.
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        t_rs_s_c = thr_copy_r2s.partition_D(s_c)
        # (R2S, R2S_M, R2S_N)
        t_rs_r_c = tiled_copy_r2s.retile(t_tr_r_c)
        return tiled_copy_r2s, t_rs_r_c, t_rs_s_c

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: cute.CopyAtom | cute.TiledCopy,
        g_c_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        s_c: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Partition shared and global C for the epilogue's TMA store.

        Returns the TMA atom with the shared-memory source and the global
        destination, both divided into epilogue subtiles.
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        g_c_epi = cute.flat_divide(
            g_c_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tma_atom_c = atom
        s_c_for_tma_partition = cute.group_modes(s_c, 0, 2)
        g_c_for_tma_partition = cute.group_modes(g_c_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        b_sg_s_c, b_sg_g_c = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            s_c_for_tma_partition,
            g_c_for_tma_partition,
        )
        return tma_atom_c, b_sg_s_c, b_sg_g_c

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        num_smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int, int]:
        """Return (accumulator, A/B, C, tile-info) stage counts.

        Two accumulator stages always fit tensor memory (N <= 256); A/B
        stages fill shared memory after the C stages and barriers, and the
        remainder adds C stages.
        """
        num_acc_stage = 2
        num_c_stage = 2
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, 1
        )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(
            c_dtype, c_smem_layout_staged_one
        )
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        tile_idx_to_expert_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        tokens: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        experts: cutlass.Int64,
        multiprocessors: cutlass.Int32,
        tile_size: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Bind row-major pointers and launch.

        ``a`` is ``[tokens, k]``, ``b`` ``[experts, n, k]`` with each
        expert's ``n / 2`` up rows followed by its ``n / 2`` gate rows, ``c``
        ``[m, n / 2]``; the tile metadata holds ``m / tile_size`` entries and
        ``token_id_mapping`` ``m``. ``multiprocessors`` is the device's count,
        which bounds the persistent grid.
        """
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((tokens, k, 1), order=(1, 0, 2)),
        )
        # The kernel's N order interleaves 64-row up and gate blocks: logical
        # row j + 64 * h + 128 * i is physical row j + 64 * i + h * n / 2
        # (h = 0 up, 1 gate). The hierarchical N mode lets TMA load each
        # 128-row tile as the matching up and gate blocks straight from the
        # linear weights.
        b = cute.make_tensor(
            b_ptr,
            layout=cute.make_layout(
                ((64, 2, n // 128), k, experts),
                stride=((k, (n // 2) * k, 64 * k), 1, n * k),
            ),
        )
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (m, n // self.out_n_factor, 1), order=(1, 0, 2)
            ),
        )
        tile_idx_to_expert_idx = cute.make_tensor(
            tile_idx_to_expert_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(
            token_id_mapping_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        return self(
            a,
            b,
            c,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            multiprocessors=multiprocessors,
            stream=stream,
        )
