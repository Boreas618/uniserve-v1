"""The mxfp8_gemm package's DeepGEMM build module: one descriptor -> one module.

The module is DeepGEMM's SM100 block-scaled product over MXFP8 operands,
``sm100_fp8_fp4_gemm_1d1d_impl`` of the pinned headers, instantiated with the
tile this module's :func:`choose` picks for the descriptor and compiled with
the package's entry unit (``src/gemm.cu``), which encodes the kernel's tensor
maps and launches it. DeepGEMM picks a tile at run time from the call's shape
(its ``get_best_config`` with ``SM100ArchSpec``, ``csrc/jit_kernels/
heuristics/sm100.hpp`` at the pinned commit); a descriptor states the row
class the tile is chosen for, so the same rule runs here, once, and a serving
host compiles nothing.

The steps:

1. fetch the ``deepgemm`` pin and stage a copy with
   ``patches/launched-grid.patch`` applied: the persistent scheduler strides by
   the launched grid rather than the compiled multiprocessor count;
2. choose the tile (:func:`choose`) for ``m`` the descriptor's row class, the
   descriptor's ``n``, ``k`` and ``groups`` (a grouped product is DeepGEMM's
   batched form, whose blocks are every group's), and the manifest's
   ``multiprocessors``;
3. compile the entry unit with the tile as its definitions, against the staged
   headers and the CUTLASS of the ``flashinfer-python`` distribution, as
   DeepGEMM's run-time compiler compiles its kernels (the pin's
   ``[compile]``), and link it with the driver API into ``module.so``.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from __future__ import annotations

import dataclasses
import math

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's unit: the operator's entry and the kernel's instance.
UNIT = "src/gemm.cu"

#: The package's patch of the pinned scheduler.
PATCH = "patches/launched-grid.patch"

#: Where the CUTLASS the kernel includes is installed.
CUTLASS = ("flashinfer-python", "flashinfer/data/cutlass/include")

#: DeepGEMM's statement of SM100's shared memory per block, which its stage
#: count is planned against (``SM100ArchSpec::smem_capacity``).
SHARED_MEMORY = 232448

#: The contraction one pipeline stage loads: 128 elements of 8-bit operands.
BLOCK_K = 128

#: The rows of the MMA's A/D layout, and the N step of a swapped store.
LAYOUT_AD_M = 128
UMMA_STEP_N = 16

#: Tensor-core operand scales are loaded in whole blocks of this many rows.
UTCCP_ROWS = 128

#: The 32-bit columns of tensor memory a multiprocessor has.
TENSOR_MEMORY_COLUMNS = 512

#: DeepGEMM's barrier bound: its stages' barriers are allocated for this many.
MAX_STAGES = 32

#: The element bytes of the operands (E4M3) and of the output (BF16).
OPERAND_BYTES = 1
OUTPUT_BYTES = 2

#: The threads of a block: the TMA and MMA warps, then the epilogue's.
NON_EPILOGUE_THREADS = 128
EPILOGUE_THREADS = 128


@dataclasses.dataclass(frozen=True)
class Layout:
    """One tile candidate (DeepGEMM's ``Layout``)."""

    swap_ab: bool
    block_m: int
    block_n: int
    cluster_m: int
    cluster_n: int

    @property
    def cluster(self) -> int:
        return self.cluster_m * self.cluster_n


@dataclasses.dataclass(frozen=True)
class Tile:
    """The chosen tile with what follows from it: DeepGEMM's ``GemmConfig``."""

    layout: Layout
    load_block_m: int
    load_block_n: int
    store_block_m: int
    store_block_n: int
    swizzle_cd: int
    stages: int
    store_stages: int
    shared_memory: int


def _align(value: int, granule: int) -> int:
    return (value + granule - 1) // granule * granule


def _swizzle(bytes_: int) -> int:
    """The widest swizzle that divides a row (``get_swizzle_mode``)."""
    for mode in (128, 64, 32, 16):
        if bytes_ % mode == 0:
            return mode
    raise SystemExit(f"a row of {bytes_} bytes has no swizzle")


def _candidates(m: int, n: int, k: int, multiprocessors: int) -> list[Layout]:
    """The tiles DeepGEMM enumerates for a normal K-major MXFP8 product."""
    found = []
    for swap_ab in (False, True):
        if swap_ab:
            blocks_m = list(range(16, 257, 16))
            blocks_n = [LAYOUT_AD_M]
        else:
            # A smaller block M keeps a small product's loads inside its rows.
            blocks_m = [32 if m <= 32 else 64 if m <= 64 else 128]
            # Fewer store blocks overlap a short contraction's stores better.
            end = 128 if k <= 256 else 256
            blocks_n = [16, *range(32, end + 1, 32)]
        # A cluster along the rows multicasts B to pairs of row blocks, so it
        # needs an even count of them; the rows of a call vary within its
        # class, which DeepGEMM's run-time choice (made for the call's own
        # rows) never meets, so no tile clusters along them here.
        for cluster_m in (1,):
            for cluster_n in (1, 2):
                if swap_ab and cluster_m > 1:
                    continue
                if not swap_ab and cluster_n > 1:
                    continue
                if cluster_m * cluster_n > 2:
                    continue
                if multiprocessors % (cluster_m * cluster_n) != 0:
                    continue
                for block_m in blocks_m:
                    if (block_m // cluster_n) % 8 != 0:
                        continue
                    if math.ceil(m / block_m) % cluster_m != 0:
                        continue
                    for block_n in blocks_n:
                        if (block_n // cluster_m) % 8 != 0:
                            continue
                        if math.ceil(n / block_n) % cluster_n != 0:
                            continue
                        if swap_ab and block_n != LAYOUT_AD_M:
                            continue
                        # The accumulator's columns and both operands' scale
                        # columns share the tensor memory.
                        scale_columns = (
                            _align(block_m, UTCCP_ROWS)
                            + _align(block_n, UTCCP_ROWS)
                        ) // 32
                        accumulator = block_m if swap_ab else block_n
                        if accumulator + scale_columns > TENSOR_MEMORY_COLUMNS:
                            continue
                        found.append(
                            Layout(
                                swap_ab, block_m, block_n, cluster_m, cluster_n
                            )
                        )
    if not found:
        raise SystemExit(f"DeepGEMM has no tile for a [{m}, {n}, {k}] product")
    return found


def _rank(m: int, n: int, groups: int, multiprocessors: int, layout: Layout):
    """DeepGEMM's comparison of two tiles, as a sort key (smaller is better).

    A grouped product's blocks are every group's (``get_layout_info`` counts
    ``get_expected_num_groups``).
    """
    blocks = (
        math.ceil(m / layout.block_m) * math.ceil(n / layout.block_n) * groups
    )
    waves = math.ceil(blocks / multiprocessors)
    last = blocks % multiprocessors or multiprocessors
    return (
        # A single wave is always better.
        waves != 1,
        # Multicasting is better.
        -layout.cluster,
        waves,
        # A fuller last wave is better.
        -last,
        layout.block_m + layout.block_n,
        layout.block_m * layout.block_n,
    )


def choose(m: int, n: int, k: int, groups: int, multiprocessors: int) -> Tile:
    """The tile DeepGEMM's run-time heuristic picks for ``groups`` products.

    Each product is ``[m, n, k]``: one, or the groups of a batched product. The
    candidates are taken in DeepGEMM's enumeration order and the first of
    the best is kept, as its comparison keeps the incumbent on a tie.
    """
    candidates = _candidates(m, n, k, multiprocessors)
    layout = min(
        candidates, key=lambda one: _rank(m, n, groups, multiprocessors, one)
    )

    load_block_m = layout.block_m // layout.cluster_n
    load_block_n = layout.block_n // layout.cluster_m
    store_block_m = (
        UMMA_STEP_N if layout.swap_ab else min(LAYOUT_AD_M, layout.block_m)
    )
    store_block_n = layout.block_n
    swizzle_cd = _swizzle(store_block_n * OUTPUT_BYTES)

    # The pipeline: the C/D staging of two TMA stores, the barriers and the
    # tensor-memory pointer, then as many A/B/scale stages as fit.
    store_stages = 2
    staged_cd = (
        store_block_m * store_block_n * OUTPUT_BYTES
        if layout.swap_ab
        else store_block_m * swizzle_cd
    ) * store_stages
    barriers = MAX_STAGES * 8 * 3 + 2 * 8 * 3 + 8
    extra = staged_cd + barriers + 4
    per_stage = (
        load_block_m * BLOCK_K * OPERAND_BYTES
        + load_block_n * BLOCK_K * OPERAND_BYTES
        + _align(layout.block_m, UTCCP_ROWS) * BLOCK_K // 32
        + _align(layout.block_n, UTCCP_ROWS) * BLOCK_K // 32
    )
    stages = min((SHARED_MEMORY - extra) // per_stage, MAX_STAGES)
    return Tile(
        layout=layout,
        load_block_m=load_block_m,
        load_block_n=load_block_n,
        store_block_m=store_block_m,
        store_block_n=store_block_n,
        swizzle_cd=swizzle_cd,
        stages=stages,
        store_stages=store_stages,
        shared_memory=extra + stages * per_stage,
    )


def definitions(request: Request, tile: Tile) -> dict[str, cuda.Definition]:
    """The entry's compile-time values: the descriptor's extents, the tile."""
    layout = tile.layout
    return {
        "UNISERVE_N": request.field("n"),
        "UNISERVE_K": request.field("k"),
        "UNISERVE_GROUPS": request.field("groups"),
        "UNISERVE_BLOCK_M": layout.block_m,
        "UNISERVE_BLOCK_N": layout.block_n,
        "UNISERVE_BLOCK_K": BLOCK_K,
        "UNISERVE_SWAP_AB": layout.swap_ab,
        "UNISERVE_CLUSTER": layout.cluster,
        "UNISERVE_MULTICAST_ON_A": layout.cluster_n > 1,
        "UNISERVE_LOAD_BLOCK_M": tile.load_block_m,
        "UNISERVE_LOAD_BLOCK_N": tile.load_block_n,
        "UNISERVE_STORE_BLOCK_M": tile.store_block_m,
        "UNISERVE_STORE_BLOCK_N": tile.store_block_n,
        "UNISERVE_SWIZZLE_CD": tile.swizzle_cd,
        "UNISERVE_STAGES": tile.stages,
        "UNISERVE_STORE_STAGES": tile.store_stages,
        "UNISERVE_SHARED_BYTES": tile.shared_memory,
        "UNISERVE_NON_EPILOGUE_THREADS": NON_EPILOGUE_THREADS,
        "UNISERVE_EPILOGUE_THREADS": EPILOGUE_THREADS,
        "UNISERVE_PLANNED_MULTIPROCESSORS": request.parameters[
            "multiprocessors"
        ],
    }


def compile(request: Request) -> None:
    """Compile the product for the request's descriptor into ``module.so``.

    A fetch, patch or compiler failure refuses the build, with its output in
    the build log.
    """
    tile = choose(
        request.field("rows"),
        request.field("n"),
        request.field("k"),
        request.field("groups"),
        request.parameters["multiprocessors"],
    )
    if tile.shared_memory > request.target.shared_memory:
        raise SystemExit(
            f"the tile takes {tile.shared_memory} bytes of shared memory and "
            f"{request.target} gives a block {request.target.shared_memory}"
        )
    print(f"the tile: {tile}", flush=True)

    pinned = fetch.pin("deepgemm")
    with cuda.scratch(request) as staged:
        try:
            names = list(fetch.manifest(pinned, request.cache))
            fetch.fetch(pinned, request.cache, names, request.jobs)
            fetch.stage(pinned, request.cache, staged, names, copy=True)
            fetch.apply_patch(staged, request.package / PATCH)
            headers = [
                *fetch.includes(pinned, staged),
                fetch.installed(*CUTLASS),
            ]
        except fetch.FetchError as error:
            raise SystemExit(
                f"staging the deepgemm pin failed: {error}"
            ) from error

        # The kernel's body is tcgen05 code behind a guard of the architecture
        # number alone, which no portable PTX can hold.
        unit = cuda.Unit(
            source=request.package / UNIT,
            definitions=definitions(request, tile),
            includes=headers,
            specific=True,
            **pinned.compile,
        )
        # The entry encodes the kernel's tensor maps through the driver API.
        cuda.native(request, [unit], libraries=["cuda"])
