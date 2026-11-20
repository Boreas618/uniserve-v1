// SPDX-License-Identifier: Apache-2.0
//
// The deepgemm module's configuration: what DeepGEMM's host side derives for one MegaMoE launch
// (`csrc/jit_kernels/heuristics/mega_moe.hpp`, `csrc/apis/mega_moe.hpp` at the pinned commit),
// restated as constant expressions of the descriptor's extents and one tile height (BLOCK_M), so
// the kernel instances (`kernel.cu`) and the transport (`entry.cu`) agree by construction.
//
// The tile height is the build's (`src/build.py`, DeepGEMM's `get_block_config_for_mega_moe`);
// everything else follows from it:
//
// - the tile: 128 output columns in two-CTA clusters multicasting the rows, one 128-byte step of
//   the contraction, the epilogue's store heights, and the scale tiles padded to the 128 rows a
//   tensor-core scale copy moves;
// - the pipeline: as many operand stages as a block's shared memory holds beside the dispatch's
//   counters and pull buffers, the epilogue's staging, the barriers and the schedule's payloads;
// - the transport: the rows a rank's buffer holds (the descriptor's, rounded up to the least
//   common multiple of every tile height), and the routed rings, sized for the worst tile height
//   by the scheduler's bound on live pool blocks (`sched::get_num_max_live_pool_blocks`).
//
// The ring and the buffer layout are DeepGEMM's own constexpr helpers, so the window this module
// creates is the one the pinned kernel addresses.

#pragma once

#include <bit>
#include <cstdint>
#include <limits>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#if !defined(UNISERVE_HIDDEN) || !defined(UNISERVE_INTERMEDIATE) || !defined(UNISERVE_TOP) || \
    !defined(UNISERVE_EXPERTS) || !defined(UNISERVE_MAX_TOKENS) || !defined(UNISERVE_RANKS) ||  \
    !defined(UNISERVE_MULTIPROCESSORS) || !defined(UNISERVE_SHARED) ||                          \
    !defined(UNISERVE_LIMIT_BITS)
#error "the build defines the descriptor's extents (src/build.py)"
#endif

namespace uniserve::mega {

// The descriptor's extents.
constexpr int kHidden = UNISERVE_HIDDEN;
constexpr int kIntermediate = UNISERVE_INTERMEDIATE;
constexpr int kTop = UNISERVE_TOP;
constexpr int kExperts = UNISERVE_EXPERTS;
constexpr int kRanks = UNISERVE_RANKS;
constexpr int kLocalExperts = kExperts / kRanks;
constexpr int kMultiprocessors = UNISERVE_MULTIPROCESSORS;
constexpr int kShared = UNISERVE_SHARED;
constexpr int kSharedIntermediate = kShared * kIntermediate;

// The rows one rank sends at most: the transport's, as the descriptor states them.
constexpr int kStatedTokens = UNISERVE_MAX_TOKENS;

// The rows a rank's buffers hold: DeepGEMM's `SymmBuffer` rounds the stated rows up to a whole
// multiple of every tile height, which the kernel is compiled for.
constexpr int kMaxTokens = (kStatedTokens + deep_gemm::layout::kLCMCandidateBlockM - 1) /
                           deep_gemm::layout::kLCMCandidateBlockM *
                           deep_gemm::layout::kLCMCandidateBlockM;

// The tile's fixed extents: 128 output columns, one 128-byte (128-element E4M3) step of the
// contraction, 32-element scale blocks, and the rows of a tensor-core scale copy.
constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kGranK = 32;
constexpr int kUtccpRows = 128;

// The threads of a block: four dispatch warps, the TMA and MMA warps, two epilogue warpgroups.
constexpr int kDispatchThreads = 128;
constexpr int kNonEpilogueThreads = 128;
constexpr int kEpilogueThreads = 256;
constexpr int kThreads = kDispatchThreads + kNonEpilogueThreads + kEpilogueThreads;

// DeepGEMM's statement of SM100's shared memory per block (`SM100ArchSpec::smem_capacity`).
constexpr int kSharedMemory = 232448;

// The swizzles of the operands' shared-memory tiles: 128 bytes for the E4M3 rows (and the E2M1
// weights unpacked to bytes), 64 for the gated rows the first epilogue stores (half a tile).
constexpr int kSwizzleOperands = 128;
constexpr int kSwizzleGated = 64;

constexpr int align(int value, int granule) { return (value + granule - 1) / granule * granule; }

// The bytes a dispatch warp pulls of a peer's row at a time: the row halved until it is at most
// 8 KiB (`get_mega_moe_config`).
constexpr int bytes_per_pull() {
  int bytes = kHidden;
  while (bytes > 8192) {
    bytes /= 2;
  }
  return bytes;
}
constexpr int kBytesPerPull = bytes_per_pull();

// The clamp of the gated activation as the kernel's template states it: none is infinity.
constexpr float clamp_of(uint32_t bits) {
  return bits == 0 ? std::numeric_limits<float>::infinity() : std::bit_cast<float>(bits);
}
constexpr float kClamp = clamp_of(static_cast<uint32_t>(UNISERVE_LIMIT_BITS));

// The routed rings: every pool block the scheduler may hold live at once, for the worst tile
// height, rounded to the tile heights' common multiple (`build_symm_buffer_layout`).
constexpr int ring_tokens() {
  const int active = kTop < kLocalExperts ? kTop : kLocalExperts;
  const int routed = kMaxTokens * kRanks * active;
  int tokens = 0;
  for (const int height : deep_gemm::layout::kCandidateBlockM) {
    const int pool = (routed + height - 1) / height + kLocalExperts;
    const int live = deep_gemm::sched::get_num_max_live_pool_blocks(pool, kMultiprocessors,
                                                                     kHidden, kIntermediate);
    tokens = tokens > live * height ? tokens : live * height;
  }
  return align(tokens, deep_gemm::layout::kLCMCandidateBlockM);
}
constexpr int kRingTokens = ring_tokens();

// The rows of the routed rings' scale tables: each pool block's rows padded to whole scale
// copies, for the worst tile height.
constexpr int sf_ring_tokens() {
  int tokens = 0;
  for (const int height : deep_gemm::layout::kCandidateBlockM) {
    const int rows = deep_gemm::layout::get_num_sf_ring_tokens(kRingTokens, height);
    tokens = tokens > rows ? tokens : rows;
  }
  return tokens;
}
constexpr int kSfRingTokens = sf_ring_tokens();

// The rows of the shared experts' scale tables: every tile height's pages of the rank's rows.
constexpr int kSharedSfTokens = deep_gemm::layout::get_num_max_shared_sf_tokens(kMaxTokens);

// The tile of one height, and the pipeline it leaves room for.
struct Tile {
  int block_m;
  // The rows each block of the cluster loads: the cluster multicasts the rows' halves.
  int load_block_m;
  // The rows one store of the first and of the second product's epilogue writes.
  int store_block_m_l1;
  int store_block_m_l2;
  // The scale tiles' rows, padded to whole tensor-core scale copies.
  int sf_block_m;
  int sf_block_n;
  int stages;
  int shared_bytes;
};

// `get_block_config_for_mega_moe` and `get_pipeline_config_for_mega_moe` for FP8 x FP4 at tile
// height `block_m`.
constexpr Tile tile(int block_m) {
  Tile tile{};
  tile.block_m = block_m;
  tile.load_block_m = block_m / 2;
  // Smaller, asymmetric store blocks at large tiles keep shared memory for the pipeline.
  tile.store_block_m_l1 = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 : 24;
  tile.store_block_m_l2 = block_m == 240 ? 8 : tile.store_block_m_l1;
  tile.sf_block_m = align(block_m, kUtccpRows);
  tile.sf_block_n = align(kBlockN, kUtccpRows);

  constexpr int kAlignment = 1024;
  constexpr int kEpilogueStages = 2;
  constexpr int kStoreStages = 2;
  constexpr int kScheduleStages = 2;
  constexpr int kDispatchWarps = kDispatchThreads / 32;
  constexpr int kEpilogueWarps = kEpilogueThreads / 32;
  constexpr int kEpilogueWarpgroups = kEpilogueWarps / 4;

  // The dispatch's per-expert counters and one pull buffer a dispatch warp.
  const int dispatch = align(kExperts * 4, kAlignment) + align(kBytesPerPull * kDispatchWarps,
                                                               kAlignment);
  // The epilogue's staging: the larger of the gated E4M3 rows and the second product's BF16 rows.
  const int staged_l1 = kEpilogueWarpgroups * tile.store_block_m_l1 * (kBlockN / 2) * kStoreStages;
  const int staged_l2 = kEpilogueWarpgroups * tile.store_block_m_l2 * kBlockN * 2;
  const int staged = align(staged_l1 > staged_l2 ? staged_l1 : staged_l2, kAlignment);
  const int task_info =
      kScheduleStages * static_cast<int>(sizeof(deep_gemm::sched::TaskInfo<true>));
  const int barriers =
      (kDispatchWarps + kEpilogueStages * 2 + kEpilogueWarps * 2 + kScheduleStages * 2) * 8;
  // The SwiGLU's cross-warp amax exchange, and the tensor-memory pointer.
  const int amax = tile.store_block_m_l1 * kEpilogueWarps * 4;
  const int fixed = dispatch + staged + amax + barriers + task_info + 4;

  // One stage: the rows' and the weights' 128-byte K step, their scales, two barriers.
  const int per_stage = tile.load_block_m * kBlockK + kBlockN * kBlockK +
                        tile.sf_block_m * (kBlockK / kGranK) +
                        tile.sf_block_n * (kBlockK / kGranK) + 2 * 8;
  tile.stages = (kSharedMemory - fixed) / per_stage;
  tile.shared_bytes = fixed + tile.stages * per_stage;
  return tile;
}

// The window one rank creates, its offsets those of DeepGEMM's `MegaMoEBuffer` at address zero.
inline deep_gemm::layout::MegaMoEBuffer buffer_layout() {
  return deep_gemm::layout::MegaMoEBuffer(nullptr, kHidden, kIntermediate, kRanks, kExperts,
                                          kMaxTokens, kTop, kRingTokens, kSfRingTokens,
                                          /*with_sf=*/true, kShared);
}

static_assert(kExperts % kRanks == 0, "the ranks hold equal windows of the experts");
static_assert(kHidden % 256 == 0 && (2 * kIntermediate) % 256 == 0,
              "both products' widths are whole two-CTA tiles");
static_assert(kSharedIntermediate % 128 == 0, "the shared experts' width is whole scale words");
static_assert(kSfRingTokens % 4 == 0, "a scale column is whole 16-byte rows");
static_assert(kHidden % kBytesPerPull == 0, "the pull divides a row");
static_assert(static_cast<int64_t>(kMaxTokens) * (kHidden / 256) < (int64_t{1} << 31) &&
                  static_cast<int64_t>(kSharedSfTokens) * (kHidden / 128) < (int64_t{1} << 31),
              "the staging's 32-bit indices hold a rank's rows and scales");

}  // namespace uniserve::mega
