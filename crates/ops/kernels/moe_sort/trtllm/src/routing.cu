// SPDX-License-Identifier: Apache-2.0
//
// Derived from TensorRT-LLM's routing pipeline,
// `cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingKernel.cuh`
// (Apache-2.0, NVIDIA CORPORATION): the histogram/offsets kernels and
// `routingMainKernel`'s block path. TensorRT-LLM counts the routes per local
// expert in shared memory, turns the counts into tile offsets and per-tile live
// row limits, and writes the two permutation maps; so does this file, with the
// same ordering — the permuted rows of one expert in ascending expanded-route
// order — and without the cluster path, which TensorRT-LLM needs only for token
// counts that outgrow one grid.
//
// The entry (`entry.cpp`) validates the call; this file only launches.
//
// Two kernels, because the tile metadata of every expert must be known before a
// single permutation index can be written:
//
//   1. `sort_metadata_kernel` — one block. Counts the routes per local expert,
//      writes `tile_to_expert`, `tile_to_limit`, `total_padded_tokens` and
//      `num_non_exiting_tiles`, and fills `expanded_to_permuted` with -1, the
//      value a route this rank does not hold keeps.
//   2. `sort_permutation_kernel` — one block per local expert. Each block
//      recomputes the counts (they are the block's own prefix over experts),
//      then walks the routes in order and writes its expert's permuted rows.
//
// The counts are recomputed rather than carried in a scratch buffer because the
// call structure has none: `ops::moe::sort::Call` hands the module exactly the
// buffers the GEMMs and the combine read, and their sizes do not bound the
// expert count. The cost is one pass over the routes per block, which the
// pipeline pays anyway.

#include <cstdint>

#include <cuda_runtime.h>

#include "../include/uniserve_moe_sort.h"

namespace {

/// The threads of one block: the histogram pass uses them all, and the
/// permutation pass partitions the routes among them.
constexpr int kThreads = 256;

/// The tiles `count` rows of one expert occupy.
__device__ __forceinline__ int64_t tiles_of(int64_t count, int64_t tile_tokens) {
  return (count + tile_tokens - 1) / tile_tokens;
}

/// The local expert of one route, or -1 when this rank does not hold it.
__device__ __forceinline__ int64_t local_expert_of(const uniserve_moe_sort_call& call,
                                                   int64_t route) {
  const int64_t local = static_cast<int64_t>(call.ids[route]) - call.local_offset;
  return (local >= 0 && local < call.local_experts) ? local : -1;
}

/// Counts the routes per local expert into `counts`, over the whole block.
__device__ __forceinline__ void count_routes(const uniserve_moe_sort_call& call,
                                             int64_t* counts) {
  const int64_t expanded = call.tokens * call.top;
  for (int64_t route = threadIdx.x; route < expanded; route += blockDim.x) {
    const int64_t local = local_expert_of(call, route);
    if (local >= 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&counts[local]), 1ull);
    }
  }
}

/// The first pass: counts, the tile metadata, the two scalars, and the -1 an
/// out-of-window route keeps.
__global__ void sort_metadata_kernel(uniserve_moe_sort_call call) {
  extern __shared__ int64_t shared[];
  int64_t* counts = shared;

  for (int64_t expert = threadIdx.x; expert < call.local_experts; expert += blockDim.x) {
    counts[expert] = 0;
  }
  __syncthreads();
  count_routes(call, counts);
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }
  // Every expert's tiles are whole tiles: an expert's permuted rows pad to the
  // next tile boundary, and the next expert starts on a tile (design 5.2's
  // grouped GEMM reads one expert per tile group).
  int64_t tile_offset = 0;
  for (int64_t expert = 0; expert < call.local_experts; ++expert) {
    const int64_t count = counts[expert];
    const int64_t tiles = tiles_of(count, call.tile_tokens);
    for (int64_t tile = 0; tile < tiles; ++tile) {
      // The limit is the row the tile's live rows end at, counted from the
      // first permuted row of the whole routing — TensorRT-LLM's `mnLimit`, and
      // the GEMMs compute a tile's rows as `limit − tile · tile_tokens`.
      const int64_t end_of_tile = (tile_offset + tile + 1) * call.tile_tokens;
      const int64_t end_of_rows = tile_offset * call.tile_tokens + count;
      call.tile_to_expert[tile_offset + tile] = static_cast<int32_t>(expert);
      call.tile_to_limit[tile_offset + tile] =
          static_cast<int32_t>(end_of_tile < end_of_rows ? end_of_tile : end_of_rows);
    }
    tile_offset += tiles;
  }
  call.total_padded_tokens[0] = static_cast<int32_t>(tile_offset * call.tile_tokens);
  call.num_non_exiting_tiles[0] = static_cast<int32_t>(tile_offset);
}

/// The second pass, one block per local expert: the permutation of that
/// expert's routes, in ascending expanded-route order, and the -1 that fills
/// the rest of its last tile.
__global__ void sort_permutation_kernel(uniserve_moe_sort_call call) {
  extern __shared__ int64_t shared[];
  const int64_t local_experts = call.local_experts;
  const int64_t expanded = call.tokens * call.top;
  int64_t* counts = shared;
  int64_t* per_thread = shared + local_experts;
  int64_t* thread_offsets = per_thread + blockDim.x;

  const int64_t expert = blockIdx.x;
  for (int64_t one = threadIdx.x; one < local_experts; one += blockDim.x) {
    counts[one] = 0;
  }
  __syncthreads();
  count_routes(call, counts);
  __syncthreads();

  // The block's first permuted row: every earlier expert's tiles, whole.
  int64_t base = 0;
  for (int64_t one = 0; one < expert; ++one) {
    base += tiles_of(counts[one], call.tile_tokens) * call.tile_tokens;
  }
  const int64_t own = counts[expert];
  const int64_t limit = tiles_of(own, call.tile_tokens) * call.tile_tokens;

  // A blocked partition keeps the ascending order: a thread's routes are one
  // contiguous range of expanded indexes, and the threads' ranges follow each
  // other, so the permuted rows come out ordered by expanded index exactly as
  // one thread walking the routes would write them.
  const int64_t chunk = (expanded + blockDim.x - 1) / blockDim.x;
  const int64_t first_route = static_cast<int64_t>(threadIdx.x) * chunk;
  const int64_t past_route = (first_route + chunk < expanded) ? first_route + chunk : expanded;
  int64_t matches = 0;
  for (int64_t route = first_route; route < past_route; ++route) {
    if (local_expert_of(call, route) == expert) {
      ++matches;
    }
  }
  per_thread[threadIdx.x] = matches;
  __syncthreads();

  if (threadIdx.x == 0) {
    int64_t running = 0;
    for (int index = 0; index < blockDim.x; ++index) {
      thread_offsets[index] = running;
      running += per_thread[index];
    }
  }
  __syncthreads();

  int64_t rank = thread_offsets[threadIdx.x];
  for (int64_t route = first_route; route < past_route; ++route) {
    if (local_expert_of(call, route) == expert) {
      call.expanded_to_permuted[route] = static_cast<int32_t>(base + rank);
      call.permuted_to_expanded[base + rank] = static_cast<int32_t>(route);
      ++rank;
    }
  }
  __syncthreads();

  // The rows of the expert's last tile past its live ones are padding: unread
  // by the GEMMs, and -1 so a reader that walks the tile sees no route.
  for (int64_t row = own + threadIdx.x; row < limit; row += blockDim.x) {
    call.permuted_to_expanded[base + row] = -1;
  }
}

}  // namespace

int32_t uniserve_moe_sort_launch(const uniserve_moe_sort_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  const int64_t local_experts = call->local_experts;
  const int64_t tile_tokens = call->tile_tokens;
  const int64_t expanded = call->tokens * call->top;
  // One shared array of counts per block, plus the permutation pass's
  // per-thread tallies: an expert count that outgrows that asks for a smaller
  // batch or the cluster path, and is refused by name.
  constexpr int64_t kMaxSharedBytes = 46 * 1024;
  const int64_t shared_bytes = (local_experts + 2 * kThreads) * sizeof(int64_t);
  if (local_experts <= 0 || tile_tokens <= 0) {
    return 2;
  }
  if (shared_bytes > kMaxSharedBytes) {
    return 3;
  }
  if (expanded > 0) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
    sort_metadata_kernel<<<1, kThreads, static_cast<size_t>(local_experts * sizeof(int64_t)), cs>>>(
        *call);
    const cudaError_t metadata = cudaGetLastError();
    if (metadata != cudaSuccess) {
      return 4;
    }
    sort_permutation_kernel<<<static_cast<unsigned int>(local_experts), kThreads,
                              static_cast<size_t>(shared_bytes), cs>>>(*call);
    const cudaError_t permutation = cudaGetLastError();
    if (permutation != cudaSuccess) {
      return 5;
    }
  } else {
    call->total_padded_tokens[0] = 0;
    call->num_non_exiting_tiles[0] = 0;
  }
  return 0;
}
