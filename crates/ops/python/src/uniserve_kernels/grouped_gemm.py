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

"""The persistent grouped GEMM of routed experts on Blackwell (SM100).

Provenance: this module and the two kernels built on it, the ``gather_gemm``
(FC1) and ``route_gemm`` (FC2) packages' CuTe DSL sources, are derived from
FlashInfer 0.6.18's MoE grouped GEMMs,
``flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py``
and ``.../blockscaled_contiguous_grouped_gemm_finalize_fusion.py`` (themselves
from TensorRT-LLM's ``tensorrt_llm/_torch/cute_dsl_kernels/blackwell/``), with
the block-scaled MMA and its scale-factor loads replaced by BF16 tcgen05 MMA.
The pipelines and device functions they call are imported from the locked
``flashinfer-python`` distribution, which the packages declare.

Both kernels are persistent and warp-specialized over the MoE sort's tile
metadata: every tile of permuted (token, route) rows belongs to one expert, a
scheduler warp walks the static persistent schedule and publishes each tile
below ``num_non_exiting_tiles`` (expert, row limit) through a shared-memory
pipeline, a TMA warp loads the expert's weight rows, the MMA warp accumulates
in FP32 in tensor memory (two stages, so the next tile's MMA overlaps this
tile's epilogue) and the epilogue warps read the accumulators back.
:class:`GroupedGemm` holds what the two kernels compute alike: the MMA and its
tiles, the TMA load of the weights, the persistent grid, the pipelines and
tensor memory, the tile-information exchange and the MMA's K-block loop. The
subclasses own what differs: how the activations reach shared memory (an
LDGSTS gather of token rows, or a TMA load of permuted rows), the epilogue
(a gated activation stored by TMA, or each row bulk-copied to its output row)
and the warps those need.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


class GroupedGemm:
    """The shared computation of the FC1 and FC2 grouped GEMMs.

    ``mma_tiler_mn`` is the MMA tile (M, N): M 128 runs one-CTA MMA, M 256
    two-CTA MMA over a cluster pair, and the MoE sort's routing tile equals M.
    ``cluster_shape_mn`` is the cluster. A subclass states its warp roles
    (``epilog_warp_id``, ``mma_warp_id``, ``sched_warp_id`` and its own) and
    ``threads_per_cta`` before calling :meth:`_make_barriers`.
    """

    def __init__(
        self,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
    ):
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_mma_tiles
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO
            if self.use_2cta_instrs
            else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        self.threads_per_warp = 32
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512

    def _make_barriers(self):
        """The named barriers of the CTA, the epilogue, the tensor-memory
        pointer and the scheduler warp, for the subclass's warp roles.
        """  # noqa: D205
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

    def _make_tiled_mma(self) -> cute.TiledMma:
        """The tcgen05 MMA of the operand dtypes over the MMA tile."""
        return sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

    def _setup_mma_tiles(self) -> cute.TiledMma:
        """Derive the tiled MMA and its K extent: four MMA instructions per K
        tile, 64 BF16 values (128 bytes of each operand row).
        """  # noqa: D205
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])

        tiled_mma = self._make_tiled_mma()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        # One CTA's share of the MMA tile: half its rows under two-CTA MMA.
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        return tiled_mma

    def _setup_cluster_layout(self, tiled_mma: cute.TiledMma):
        """The cluster's (V, M, N, K) layout and the CTAs a B load multicasts
        to.
        """  # noqa: D205
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

    def _setup_ab_smem_layouts(self, tiled_mma: cute.TiledMma):
        """The staged shared-memory layouts of the A and B tiles."""
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )

    def _make_tma_b(self, tiled_mma: cute.TiledMma, b: cute.Tensor):
        """The TMA load of the experts' weight tiles, multicast across the
        cluster's M mode: its atom, its tensor and one stage's bytes.
        """  # noqa: D205
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(
            self.b_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        return tma_atom_b, tma_tensor_b, b_copy_size

    @staticmethod
    def _compute_grid(
        num_ctas_mnl: tuple,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """The persistent tile scheduler over ``num_ctas_mnl`` CTA tiles and
        the grid of at most ``max_active_clusters`` clusters it runs on.

        Clusters raster along N, so a CTA's tiles visit the M tiles in rising
        order and the scheduler warp stops at the first M tile at or past
        ``num_non_exiting_tiles``.
        """  # noqa: D205
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=False
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    def _launch(self, kernel, grid, stream):
        """Launch the kernel the subclass traced on ``stream``."""
        kernel.launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[union-attr]
            stream=stream,
            min_blocks_per_mp=1,
            # The kernel waits for its predecessor's grid before it reads
            # global memory and lets its successor launch once it is done.
            use_pdl=True,
        )

    def _cta_coordinates(
        self, tiled_mma: cute.TiledMma, cluster_layout_vmnk: cute.Layout
    ):
        """This CTA's MMA peer index, whether it leads its MMA and its
        coordinate in the cluster, and the shared storage it allocates.
        """  # noqa: D205
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory
        # dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        return (
            use_2cta_instrs,
            mma_tile_coord_v,
            is_leader_cta,
            block_in_cluster_coord_vmnk,
            tidx,
            storage,
        )

    def _make_acc_and_tile_info_pipelines(
        self, storage, cluster_layout_vmnk: cute.Layout, use_2cta_instrs
    ):
        """The accumulator pipeline (MMA warp to epilogue warps of both MMA
        peers) and the tile-information pipeline (scheduler warp to every
        other warp).
        """  # noqa: D205
        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        num_acc_consumer_threads = (
            len(self.epilog_warp_id)
            * self.threads_per_warp
            * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )
        return acc_pipeline, tile_info_pipeline

    def _make_tmem_allocator(self, storage, use_2cta_instrs):
        """The tensor-memory allocator the first epilogue warp drives."""
        return utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

    def _cluster_arrive(self):
        """Signal the cluster that this CTA's barriers are initialized."""
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

    def _ab_smem_tensors(
        self,
        storage,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        """The staged A and B tiles in shared memory."""
        # (MMA, MMA_M, MMA_K, STAGE)
        s_a = storage.s_a.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        s_b = storage.s_b.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        return s_a, s_b

    def _tile_info_smem(self, storage):
        """The published tiles in shared memory: (bidx, bidy, expert, valid,
        mn_limit) per tile-information stage.
        """  # noqa: D205
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        return storage.s_info.get_tensor(info_layout)

    def _b_full_mcast_mask(
        self, cluster_layout_vmnk, block_in_cluster_coord_vmnk
    ):
        """The CTAs a B tile's TMA load arrives at."""
        return cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

    def _local_tile_b(self, m_b_nkl: cute.Tensor) -> cute.Tensor:
        """The experts' weights in MMA tiles: (bN, bK, loopN, loopK, loopL)."""
        return cute.local_tile(
            m_b_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )

    def _partition_b_and_fragments(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_b: cute.CopyAtom,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk,
        s_a: cute.Tensor,
        s_b: cute.Tensor,
        t_cg_b: cute.Tensor,
    ):
        """The TMA load of B partitioned over the cluster's M mode, and the
        MMA's A, B and staged accumulator fragments.
        """  # noqa: D205
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        t_bs_b, t_bg_b = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(s_b, 0, 3),
            cute.group_modes(t_cg_b, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        t_cr_a = tiled_mma.make_fragment_A(s_a)
        # (MMA, MMA_N, MMA_K, STAGE)
        t_cr_b = tiled_mma.make_fragment_B(s_b)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        t_ct_acc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        return t_bs_b, t_bg_b, t_cr_a, t_cr_b, t_ct_acc_fake

    def _cluster_wait(self):
        """Wait until every CTA of the cluster initialized its barriers."""
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

    def _publish_tile(
        self,
        s_info: cute.Tensor,
        tile_info_producer_state,
        cur_tile_coord,
        expert_idx,
        is_valid_tile,
        mn_limit,
    ):
        """Write one tile's information into its stage (one thread) and make
        it visible to the async proxy.
        """  # noqa: D205
        with cute.arch.elect_one():
            s_info[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
            s_info[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
            s_info[(2, tile_info_producer_state.index)] = expert_idx
            s_info[(3, tile_info_producer_state.index)] = cutlass.Int32(
                is_valid_tile
            )
            s_info[(4, tile_info_producer_state.index)] = mn_limit
            # fence view async shared
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )

    @staticmethod
    def _read_tile_info(
        tile_info_pipeline,
        tile_info_consumer_state,
        s_info: cute.Tensor,
        tile_info: cute.Tensor,
        fields: tuple[int, ...],
    ):
        """Read the next published tile's ``fields`` into ``tile_info`` and
        release its stage; answers whether the tile is valid.

        The caller advances ``tile_info_consumer_state``: the DSL carries a
        loop's state through the names its body assigns or calls methods on,
        so the advance stays in the loop that iterates over tiles.
        """  # noqa: D205
        tile_info_pipeline.consumer_wait(tile_info_consumer_state)
        for field in fields:
            tile_info[field] = s_info[(field, tile_info_consumer_state.index)]
        is_valid_tile = tile_info[3] == 1
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        tile_info_pipeline.consumer_release(tile_info_consumer_state)
        return is_valid_tile

    def _accumulator_base(self, tmem, t_ct_acc_fake: cute.Tensor):
        """The staged accumulators in the tensor memory the epilogue's warp
        allocated.
        """  # noqa: D205
        #
        # Bar sync for retrieve tensor memory ptr from shared memory
        #
        tmem.wait_for_alloc()

        #
        # Retrieving tensor memory ptr and make accumulator tensor
        #
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        t_ct_acc_base = cute.make_tensor(tmem_ptr, t_ct_acc_fake.layout)
        return tmem_ptr, t_ct_acc_base

    def _free_tensor_memory(self, tmem, tmem_ptr):
        """Give the tensor memory back once every epilogue warp is done."""
        #
        # Dealloc the tensor memory buffer
        #
        tmem.relinquish_alloc_permit()
        self.epilog_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr)

    def _partition_accumulators(
        self,
        tidx: cutlass.Int32,
        t_acc: cute.Tensor,
        g_c_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: cutlass.Boolean | bool,
        c_layout: utils.LayoutEnum,
        c_dtype: type[cutlass.Numeric],
    ):
        """Partition the accumulators for tensor-memory-to-register loads.

        Returns the T2R tiled copy, this thread's partition of the staged
        accumulators, and the shape of one epilogue subtile's register
        fragment.
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        t_acc_epi = cute.flat_divide(
            t_acc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, t_acc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        t_tr_t_acc = thr_copy_t2r.partition_S(t_acc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        g_c_mnl_epi = cute.flat_divide(
            g_c_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        t_tr_g_c = thr_copy_t2r.partition_D(g_c_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        fragment = t_tr_g_c[(None, None, None, 0, 0, 0, 0, 0)].shape
        return tiled_copy_t2r, t_tr_t_acc, fragment
