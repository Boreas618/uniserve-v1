// SPDX-License-Identifier: Apache-2.0
//
// Derived from TensorRT-LLM's routing pipeline,
// `cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingKernel.cuh`
// (Apache-2.0, NVIDIA CORPORATION). TensorRT-LLM counts the routes per local
// expert, turns the counts into tile offsets and per-tile live row limits, and
// writes the permutation maps; so does this file, with the same ordering: the
// permuted rows of one expert in ascending expanded-route order.
//
// The entry (`entry.cpp`) validates the call; this file only launches.
//
// The sort runs once per expert layer of every step, so a decode step pays for
// it dozens of times over and its cost is its latency. One block sorts a batch of
// up to `kChunk` routes in one launch:
//
//   1. Each warp owns one contiguous slice of the routes and counts them by
//      expert: `__match_any_sync` finds the lanes that name the same expert, and
//      one of them adds the group's size to the warp's own row of counts, so no
//      atomic is needed and a slice's order is kept.
//   2. A prefix over the warps turns each warp's counts into where its slice
//      starts inside each expert, which is what keeps the routes ascending.
//   3. A scan over the experts gives every expert its first tile.
//   4. Each warp walks its slice again and places its routes.
//
// A larger batch needs more than one block, so the same steps are split over
// three launches: `sort_count_kernel` counts each chunk, `sort_tiles_kernel` turns
// the counts into each chunk's start in each expert and writes the tile metadata,
// and `sort_place_kernel` places each chunk. The chunks' counts travel in
// `expanded_to_permuted`, at the head of the chunk's own routes: the call has no
// scratch buffer, and a block overwrites that row only after reading it.
//
// So a route outside this rank's local window has an entry in neither permuted
// map, and no reader walks one: the combine runs after the exchange, over the
// rows a rank was sent, and the GEMMs are bounded by `tile_to_limit`.

#include <cstdint>

#include <cuda_runtime.h>

#include "../include/uniserve_moe_sort.h"

namespace {

/// The threads of one block.
constexpr int kThreads = 512;
/// The warps of one block, each owning a slice of a chunk's routes.
constexpr int kWarps = kThreads / 32;
/// The routes one block sorts: what one launch serves, and a chunk of a larger batch.
constexpr int64_t kChunk = 4096;
constexpr unsigned kAll = 0xffffffffu;

/// The tiles `count` rows of one expert occupy.
__device__ __forceinline__ int32_t tiles_of(int32_t count, int32_t tile_tokens) {
  return (count + tile_tokens - 1) / tile_tokens;
}

/// The routes the batch fills.
///
/// A captured bucket is wider than the batch that fills it, so a route of a row past the batch's
/// extent belongs to no token: it is not counted into a tile, it receives no permuted row, and the
/// map states it as `-2` rather than letting a reader dereference an index that names nothing.
/// A null count states no count, and every row is live.
__device__ __forceinline__ int64_t live_routes(const uniserve_moe_sort_call& call) {
  const int64_t expanded = call.tokens * call.top;
  return call.live_tokens == nullptr ? expanded : call.live_tokens[0] * call.top;
}

/// The local expert of one route of a live row, or -1 when this rank does not hold it.
__device__ __forceinline__ int32_t local_expert_of(const uniserve_moe_sort_call& call,
                                                   int64_t route) {
  const int64_t local = static_cast<int64_t>(call.ids[route]) - call.local_offset;
  return (local >= 0 && local < call.local_experts) ? static_cast<int32_t>(local) : -1;
}

/// The slice of `[first, past)` warp `warp` owns: whole groups of 32 routes, in order.
__device__ __forceinline__ void slice_of(int64_t first, int64_t past, int warp, int64_t* start,
                                         int64_t* end) {
  const int64_t groups = (past - first + 31) / 32;
  const int64_t own = (groups + kWarps - 1) / kWarps;
  *start = first + warp * own * 32;
  *end = *start + own * 32;
  if (*end > past) {
    *end = past;
  }
}

/// Counts the routes of `[first, past)` into each warp's row of `warp_counts`, then turns the rows
/// into where each warp's slice starts within the chunk, and leaves the chunk's counts in `totals`.
__device__ __forceinline__ void count_chunk(const uniserve_moe_sort_call& call, int64_t first,
                                            int64_t past, int64_t live, int32_t* warp_counts,
                                            int32_t* totals) {
  const int local_experts = static_cast<int>(call.local_experts);
  for (int index = threadIdx.x; index < kWarps * local_experts; index += blockDim.x) {
    warp_counts[index] = 0;
  }
  __syncthreads();

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  int32_t* counts = warp_counts + warp * local_experts;
  int64_t start;
  int64_t end;
  slice_of(first, past, warp, &start, &end);
  for (int64_t base = start; base < end; base += 32) {
    const int64_t route = base + lane;
    const int32_t local = (route < end && route < live) ? local_expert_of(call, route) : -1;
    const unsigned peers = __match_any_sync(kAll, local);
    if (local >= 0 && lane == __ffs(peers) - 1) {
      counts[local] += __popc(peers);
    }
  }
  __syncthreads();

  for (int expert = threadIdx.x; expert < local_experts; expert += blockDim.x) {
    int32_t running = 0;
    for (int one = 0; one < kWarps; ++one) {
      const int32_t count = warp_counts[one * local_experts + expert];
      warp_counts[one * local_experts + expert] = running;
      running += count;
    }
    totals[expert] = running;
  }
  __syncthreads();
}

/// Writes each expert's first tile into `bases` from the counts in `totals`, the tile metadata, the
/// two scalars and the padding rows of each expert's last tile.
__device__ __forceinline__ void write_tiles(const uniserve_moe_sort_call& call,
                                            const int32_t* totals, int32_t* bases) {
  const int local_experts = static_cast<int>(call.local_experts);
  const int32_t tile_tokens = static_cast<int32_t>(call.tile_tokens);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  // The first warp scans the experts, 32 at a time, and the last lane carries the running sum.
  if (warp == 0) {
    int32_t carried = 0;
    for (int first = 0; first < local_experts; first += 32) {
      const int expert = first + lane;
      const int32_t tiles = expert < local_experts ? tiles_of(totals[expert], tile_tokens) : 0;
      int32_t inclusive = tiles;
      for (int offset = 1; offset < 32; offset *= 2) {
        const int32_t below = __shfl_up_sync(kAll, inclusive, offset);
        if (lane >= offset) {
          inclusive += below;
        }
      }
      if (expert < local_experts) {
        bases[expert] = carried + inclusive - tiles;
      }
      carried += __shfl_sync(kAll, inclusive, 31);
    }
    if (lane == 0) {
      call.total_padded_tokens[0] = carried * tile_tokens;
      call.num_non_exiting_tiles[0] = carried;
    }
  }
  __syncthreads();

  // One warp per expert. The limit is the row the tile's live rows end at, counted from the first
  // permuted row of the whole routing: TensorRT-LLM's `mnLimit`, and the GEMMs compute a tile's
  // rows as `limit - tile * tile_tokens`.
  for (int expert = warp; expert < local_experts; expert += kWarps) {
    const int32_t count = totals[expert];
    if (count == 0) {
      continue;
    }
    const int32_t first_tile = bases[expert];
    const int32_t tiles = tiles_of(count, tile_tokens);
    const int32_t first_row = first_tile * tile_tokens;
    for (int32_t tile = lane; tile < tiles; tile += 32) {
      const int32_t end_of_tile = (first_tile + tile + 1) * tile_tokens;
      const int32_t end_of_rows = first_row + count;
      call.tile_to_expert[first_tile + tile] = expert;
      call.tile_to_limit[first_tile + tile] =
          end_of_tile < end_of_rows ? end_of_tile : end_of_rows;
    }
    // The rows of the last tile past the live ones are padding: unread by the GEMMs, and -1 so a
    // reader that walks the tile sees no route.
    for (int32_t row = count + lane; row < tiles * tile_tokens; row += 32) {
      call.permuted_to_expanded[first_row + row] = -1;
      call.permuted_to_token[first_row + row] = -1;
    }
  }
}

/// Places the routes of `[first, past)`: `warp_counts` holds where each warp's slice starts within
/// the chunk, and `starts` the first permuted row of each expert for this chunk.
__device__ __forceinline__ void place_chunk(const uniserve_moe_sort_call& call, int64_t first,
                                            int64_t past, int64_t live, int32_t* warp_counts,
                                            const int32_t* starts) {
  const int local_experts = static_cast<int>(call.local_experts);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  int32_t* counts = warp_counts + warp * local_experts;
  const unsigned below = (1u << lane) - 1u;
  int64_t start;
  int64_t end;
  slice_of(first, past, warp, &start, &end);
  for (int64_t base = start; base < end; base += 32) {
    const int64_t route = base + lane;
    const bool inside = route < end;
    const bool filled = inside && route < live;
    const int32_t local = filled ? local_expert_of(call, route) : -1;
    const unsigned peers = __match_any_sync(kAll, local);
    if (local >= 0) {
      const int32_t row = starts[local] + counts[local] + __popc(peers & below);
      call.expanded_to_permuted[route] = row;
      call.permuted_to_expanded[row] = static_cast<int32_t>(route);
      call.permuted_to_token[row] = static_cast<int32_t>(route / call.top);
    } else if (inside) {
      // A route this rank does not hold is stated absent here, so no caller has to pre-fill the
      // map. A route of a token row the batch does not fill is a different absence and says so.
      call.expanded_to_permuted[route] = filled ? -1 : -2;
    }
    __syncwarp();
    if (local >= 0 && lane == __ffs(peers) - 1) {
      counts[local] += __popc(peers);
    }
    __syncwarp();
  }
}

/// A whole batch of at most `kChunk` routes.
__global__ void __launch_bounds__(kThreads) sort_block_kernel(uniserve_moe_sort_call call) {
  extern __shared__ int32_t shared[];
  const int local_experts = static_cast<int>(call.local_experts);
  int32_t* warp_counts = shared;
  int32_t* totals = warp_counts + kWarps * local_experts;
  int32_t* starts = totals + local_experts;
  const int64_t past = call.tokens * call.top;
  const int64_t live = live_routes(call);

  count_chunk(call, 0, past, live, warp_counts, totals);
  write_tiles(call, totals, starts);
  __syncthreads();
  for (int expert = threadIdx.x; expert < local_experts; expert += blockDim.x) {
    starts[expert] *= static_cast<int32_t>(call.tile_tokens);
  }
  __syncthreads();
  place_chunk(call, 0, past, live, warp_counts, starts);
}

/// The chunk `blockIdx.x` owns, as `[first, past)`.
__device__ __forceinline__ void chunk_of(const uniserve_moe_sort_call& call, int64_t* first,
                                         int64_t* past) {
  const int64_t expanded = call.tokens * call.top;
  *first = static_cast<int64_t>(blockIdx.x) * kChunk;
  *past = *first + kChunk < expanded ? *first + kChunk : expanded;
}

/// The first of three launches: each chunk's counts, at the head of its own routes in the map.
__global__ void __launch_bounds__(kThreads) sort_count_kernel(uniserve_moe_sort_call call) {
  extern __shared__ int32_t shared[];
  const int local_experts = static_cast<int>(call.local_experts);
  int32_t* warp_counts = shared;
  int32_t* totals = warp_counts + kWarps * local_experts;
  int64_t first;
  int64_t past;
  chunk_of(call, &first, &past);

  count_chunk(call, first, past, live_routes(call), warp_counts, totals);
  for (int expert = threadIdx.x; expert < local_experts; expert += blockDim.x) {
    call.expanded_to_permuted[first + expert] = totals[expert];
  }
}

/// The second: turns each chunk's counts into its first row in each expert, and writes the tiles.
__global__ void __launch_bounds__(kThreads) sort_tiles_kernel(uniserve_moe_sort_call call,
                                                              int chunks) {
  extern __shared__ int32_t shared[];
  const int local_experts = static_cast<int>(call.local_experts);
  int32_t* totals = shared;
  int32_t* bases = totals + local_experts;

  for (int expert = threadIdx.x; expert < local_experts; expert += blockDim.x) {
    int32_t running = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      int32_t* slot = call.expanded_to_permuted + chunk * kChunk + expert;
      const int32_t count = *slot;
      *slot = running;
      running += count;
    }
    totals[expert] = running;
  }
  __syncthreads();
  write_tiles(call, totals, bases);
  __syncthreads();
  for (int expert = threadIdx.x; expert < local_experts; expert += blockDim.x) {
    const int32_t first_row = bases[expert] * static_cast<int32_t>(call.tile_tokens);
    for (int chunk = 0; chunk < chunks; ++chunk) {
      call.expanded_to_permuted[chunk * kChunk + expert] += first_row;
    }
  }
}

/// The third: each chunk reads its row before it overwrites the routes that row sits in.
__global__ void __launch_bounds__(kThreads) sort_place_kernel(uniserve_moe_sort_call call) {
  extern __shared__ int32_t shared[];
  const int local_experts = static_cast<int>(call.local_experts);
  int32_t* warp_counts = shared;
  int32_t* totals = warp_counts + kWarps * local_experts;
  int32_t* starts = totals + local_experts;
  int64_t first;
  int64_t past;
  chunk_of(call, &first, &past);

  for (int expert = threadIdx.x; expert < local_experts; expert += blockDim.x) {
    starts[expert] = call.expanded_to_permuted[first + expert];
  }
  const int64_t live = live_routes(call);
  count_chunk(call, first, past, live, warp_counts, totals);
  place_chunk(call, first, past, live, warp_counts, starts);
}

}  // namespace

int32_t uniserve_moe_sort_launch(const uniserve_moe_sort_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  const int64_t local_experts = call->local_experts;
  const int64_t expanded = call->tokens * call->top;
  // A chunk's counts sit at the head of its routes, so an expert count past a chunk has no room.
  constexpr int64_t kMaxSharedBytes = 46 * 1024;
  const int64_t shared_bytes = (kWarps + 2) * local_experts * sizeof(int32_t);
  if (local_experts <= 0 || call->tile_tokens <= 0) {
    return 2;
  }
  if (shared_bytes > kMaxSharedBytes || local_experts > kChunk) {
    return 3;
  }
  cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
  const size_t shared = static_cast<size_t>(shared_bytes);
  if (expanded <= kChunk) {
    sort_block_kernel<<<1, kThreads, shared, cs>>>(*call);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
  }
  const int chunks = static_cast<int>((expanded + kChunk - 1) / kChunk);
  sort_count_kernel<<<chunks, kThreads, shared, cs>>>(*call);
  if (cudaGetLastError() != cudaSuccess) {
    return 4;
  }
  sort_tiles_kernel<<<1, kThreads, shared, cs>>>(*call, chunks);
  if (cudaGetLastError() != cudaSuccess) {
    return 4;
  }
  sort_place_kernel<<<chunks, kThreads, shared, cs>>>(*call);
  return cudaGetLastError() == cudaSuccess ? 0 : 5;
}
