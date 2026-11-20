// SPDX-License-Identifier: Apache-2.0

// The index_select module's CUDA backend: the `kept` best pooled scores of each row, exactly as
// the operator states them (`ops::attention::IndexSelect`).
//
//   pooled[t, b] = max_{c < POOL} s[t, b * POOL + c]       +inf for b = (visible[t] - 1) / POOL
//                                                           when POOL > 1 and visible[t] > 0
//   ranked[t]    = the blocks below ceil(visible[t] / POOL); every column of candidate blocks
//   out[t, i]    = the ranked blocks of the kept largest pooled scores: the strictly greater
//                  ones by column, then the ties at the kept-th score by column; -1 for a -inf
//                  score, an unnamed candidate block or a slot past the ranked blocks; mapped
//                  through candidate blocks of CANDIDATES groups as candidates[t, c / B] * B +
//                  c % B
//
// Scores are FP32 or BF16, the call's type choosing between the module's two instantiations; a
// BF16 score is ranked by its FP32 value. Each is ranked by its monotone 32-bit key (negative
// values inverted, positive ones with their sign bit set), so equal keys are equal scores and the
// tie rule is the column order.
//
// One row is one thread-block cluster of `S` CTAs (the entry chooses `S` so that the rows'
// clusters fill the device), each CTA owning a contiguous slice of the row's ranked blocks, so
// the slices are in column order by cluster rank; a row of few ranked blocks is split over only
// as many of its CTAs as give each a slice of at least 8192 blocks. The selection is a radix
// select over the keys in three levels of 11, 11 and 10 bits:
//
// 1. every CTA reads its slice once and counts its keys' top 11 bits into a shared histogram;
//    the cluster sums its CTAs' histograms through distributed shared memory, and every CTA
//    finds the same bin `b1` the kept-th largest key lies in;
// 2. the keys of `b1` and above are kept in shared memory, in column order: a slice that fits
//    the buffer whole is cached there during the first read (no second read), else the second
//    read collects its keys of `b1` and above (fewer than `kept` lie above `b1`), else (a bin
//    too full for the buffer) the slice is read again for each further level;
// 3. two more levels over the kept keys (bits 20..10 of `b1`'s keys, then 9..0) find the
//    threshold key `T` exactly, with the count of greater keys and of ties the kept slots take;
// 4. each CTA publishes its counts of keys above `T` and equal to it; a CTA's slots start after
//    the lower ranks' counts, and within a slice the keys leave in column order (an ordered
//    block scan per tile), so the output is the operator's slot for slot.
//
// A row of no more ranked blocks than kept slots keeps every ranked block in column order.
// Nothing is allocated and nothing synchronizes with the host, so CUDA graphs capture a launch.

#include <uniserve_index_select.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#if !defined(UNISERVE_POOL) || !defined(UNISERVE_CANDIDATES)
#error "the manifest defines the descriptor's pool and candidate block"
#endif

namespace {

namespace cg = cooperative_groups;

// The descriptor's values.
constexpr int64_t kPool = UNISERVE_POOL;
constexpr int64_t kCandidates = UNISERVE_CANDIDATES;

static_assert(kPool >= 1 && (kPool & (kPool - 1)) == 0, "the pool is a power of two");

// The groups of a candidate block a kept column divides by: one when there are none, so the
// unrestricted module's division is by one and never taken.
constexpr int64_t kBlockGroups = kCandidates > 0 ? kCandidates : 1;
static_assert(kPool == 1 || kCandidates == 0, "the block mode ranks unrestricted scores");

// A CTA: 32 warps, each thread holding 8 consecutive keys of a tile.
constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kPerThread = 8;
constexpr int64_t kTile = int64_t{kThreads} * kPerThread;

// The radix levels: bits 31..21, 20..10 and 9..0 of a key.
constexpr int kBins = 2048;
constexpr int kShift1 = 21;
constexpr int kShift2 = 10;
constexpr uint32_t kMask2 = 2047;
constexpr uint32_t kMask3 = 1023;

// The key of -inf.
constexpr uint32_t kNegativeInfinity = 0x007FFFFFu;

// The most clustered CTAs a row is split over, and the fewest pooled entries a CTA of a split
// row takes: a short row is one CTA's.
constexpr int kMostCluster = 16;
constexpr int64_t kLeastSlice = 8192;

// The shared buffer: a slice of keys cached whole, or the keys of `b1` and above with their
// pooled indices (keys first, then indices).
constexpr int kBufferWords = 45056;
constexpr int64_t kCacheCapacity = kBufferWords;
constexpr int64_t kCollectCapacity = kBufferWords / 2;

struct Shared {
  // Each level's histogram of this CTA's keys, and the cluster's sums of them.
  uint32_t local[3][kBins];
  uint32_t total[3][kBins];
  // A block scan's warp sums, its total, and values one thread finds for all.
  uint32_t warp[kWarps];
  uint32_t scan_total;
  uint32_t found[2];
  // This CTA's counts of keys above the threshold and equal to it, for the higher ranks.
  uint32_t published[2];
  alignas(16) uint32_t buffer[kBufferWords];
};

// What the entry hands the kernel.
struct Params {
  const void* scores;
  const uint32_t* visible;
  const int32_t* candidates;
  int32_t* out;
  int64_t columns;
  int64_t kept;
};

__device__ __forceinline__ uint32_t monotone(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
}

// One row's scores, BF16 ones when `kBf16`: its first byte, its columns, its pinned block (or -1)
// and whether its columns may be read 16 bytes at a time.
template <bool kBf16>
struct Row {
  const uint8_t* base;
  int64_t columns;
  int64_t pin;
  bool aligned;

  __device__ __forceinline__ float column(int64_t c) const {
    if constexpr (kBf16) {
      const uint16_t bits = reinterpret_cast<const uint16_t*>(base)[c];
      return __uint_as_float(static_cast<uint32_t>(bits) << 16);
    } else {
      return reinterpret_cast<const float*>(base)[c];
    }
  }

  // Eight consecutive columns from `first`, a multiple of eight, all below `columns`.
  __device__ __forceinline__ void eight(int64_t first, float (&value)[8]) const {
    if constexpr (kBf16) {
      const uint4 word = *reinterpret_cast<const uint4*>(base + first * 2);
      const uint32_t parts[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        value[2 * j] = __uint_as_float(parts[j] << 16);
        value[2 * j + 1] = __uint_as_float(parts[j] & 0xFFFF0000u);
      }
    } else {
      const float4 low = *reinterpret_cast<const float4*>(base + first * 4);
      const float4 high = *reinterpret_cast<const float4*>(base + first * 4 + 16);
      value[0] = low.x, value[1] = low.y, value[2] = low.z, value[3] = low.w;
      value[4] = high.x, value[5] = high.y, value[6] = high.z, value[7] = high.w;
    }
  }

  // The pooled score of block `block`: the largest of its columns, -inf past the row, +inf
  // pinned.
  __device__ __forceinline__ float pooled(int64_t block) const {
    if (block == pin) {
      return __uint_as_float(0x7F800000u);
    }
    const int64_t first = block * kPool;
    float best = __uint_as_float(0xFF800000u);
    if constexpr (kPool >= 8) {
      if (aligned && first + kPool <= columns) {
#pragma unroll
        for (int64_t c = 0; c < kPool; c += 8) {
          float value[8];
          eight(first + c, value);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            best = fmaxf(best, value[j]);
          }
        }
        return best;
      }
    }
    for (int64_t c = first; c < first + kPool && c < columns; ++c) {
      best = fmaxf(best, column(c));
    }
    return best;
  }

  // The keys of pooled blocks `first` .. `first + 8`, `first` a multiple of eight.
  __device__ __forceinline__ void keys(int64_t first, uint32_t (&key)[8]) const {
    if constexpr (kPool == 1) {
      if (aligned && first + 8 <= columns) {
        float value[8];
        eight(first, value);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          key[j] = monotone(value[j]);
        }
        return;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      key[j] = monotone(pooled(first + j));
    }
  }
};

// The exclusive prefix of `value` over the block's threads in thread order, and the total. Every
// thread calls it.
__device__ __forceinline__ uint32_t block_scan(Shared& shared, uint32_t value, uint32_t& total) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  uint32_t inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, inclusive, offset);
    if (lane >= offset) {
      inclusive += other;
    }
  }
  __syncthreads();
  if (lane == 31) {
    shared.warp[warp] = inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    const uint32_t sum = shared.warp[lane];
    uint32_t prefix = sum;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, prefix, offset);
      if (lane >= offset) {
        prefix += other;
      }
    }
    shared.warp[lane] = prefix - sum;
    if (lane == 31) {
      shared.scan_total = prefix;
    }
  }
  __syncthreads();
  total = shared.scan_total;
  return shared.warp[warp] + inclusive - value;
}

// The largest bin `bin` of `histogram` (`bins`, 1024 or 2048) whose suffix count reaches
// `need`, and the count of keys in the bins above it. Every thread calls it; the cluster's sums
// hold at least `need` keys.
__device__ void threshold_bin(Shared& shared, const uint32_t* histogram, int bins, uint32_t need,
                              uint32_t& bin, uint32_t& above) {
  // Thread t holds `per` bins, descending from the top: thread order is descending bin order.
  const int per = bins / kThreads;
  const int top = bins - 1 - static_cast<int>(threadIdx.x) * per;
  uint32_t sum = 0;
  for (int b = top; b > top - per; --b) {
    sum += histogram[b];
  }
  uint32_t total = 0;
  uint32_t running = block_scan(shared, sum, total);
  for (int b = top; b > top - per; --b) {
    if (running < need && running + histogram[b] >= need) {
      shared.found[0] = static_cast<uint32_t>(b);
      shared.found[1] = running;
    }
    running += histogram[b];
  }
  __syncthreads();
  bin = shared.found[0];
  above = shared.found[1];
  __syncthreads();
}

// The sum of `value` over the block's threads.
__device__ __forceinline__ uint32_t block_sum(Shared& shared, uint32_t value) {
  uint32_t total = 0;
  block_scan(shared, value, total);
  return total;
}

// Where a pass reads a slice's keys: the row itself, the slice cached whole, or the keys of `b1`
// and above collected with their pooled indices.
enum class Source { Global, Cached, Collected };

// Calls `visit(key, block, valid)` once per tile with every thread of the block: thread t holds
// the keys of 8 consecutive entries of the tile (in column order across the block's threads)
// and their pooled blocks, `valid` the mask of those inside the slice.
template <typename Row, typename Visit>
__device__ __forceinline__ void walk(Source source, const Row& row, const Shared& shared,
                                     int64_t lo, int64_t hi, int64_t collected, Visit&& visit) {
  uint32_t key[kPerThread];
  uint32_t block[kPerThread];
  if (source == Source::Global) {
    for (int64_t start = lo & ~int64_t{7}; start < hi; start += kTile) {
      const int64_t first = start + int64_t{threadIdx.x} * kPerThread;
      uint32_t valid = 0;
      if (first < hi && first + kPerThread > lo) {
        row.keys(first, key);
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
          const int64_t entry = first + j;
          block[j] = static_cast<uint32_t>(entry);
          valid |= static_cast<uint32_t>(entry >= lo && entry < hi) << j;
        }
      }
      visit(key, block, valid);
    }
    return;
  }

  const int64_t count = source == Source::Cached ? hi - lo : collected;
  const uint32_t* keys = shared.buffer;
  const uint32_t* blocks = shared.buffer + kCollectCapacity;
  for (int64_t start = 0; start < count; start += kTile) {
    const int64_t first = start + int64_t{threadIdx.x} * kPerThread;
    uint32_t valid = 0;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int64_t at = first + j;
      if (at < count) {
        key[j] = keys[at];
        block[j] = source == Source::Cached ? static_cast<uint32_t>(lo + at) : blocks[at];
        valid |= 1u << j;
      }
    }
    visit(key, block, valid);
  }
}

// The entry a kept pooled block names: -1 for a -inf score or an unnamed candidate block.
__device__ __forceinline__ int32_t entry_of(uint32_t key, uint32_t block, const int32_t* named) {
  if (key == kNegativeInfinity) {
    return -1;
  }
  if constexpr (kCandidates > 0) {
    const int32_t candidate = named[block / kBlockGroups];
    return candidate < 0
               ? -1
               : candidate * static_cast<int32_t>(kBlockGroups) +
                     static_cast<int32_t>(block % kBlockGroups);
  } else {
    return static_cast<int32_t>(block);
  }
}

template <bool kBf16>
__global__ void __launch_bounds__(kThreads, 1) select_kernel(Params params) {
  extern __shared__ __align__(16) uint8_t memory[];
  Shared& shared = *reinterpret_cast<Shared*>(memory);
  cg::cluster_group cluster = cg::this_cluster();
  const int ranks = static_cast<int>(cluster.num_blocks());
  const int rank = static_cast<int>(cluster.block_rank());
  const int64_t token = blockIdx.x / ranks;

  const int64_t columns = params.columns;
  const int64_t kept = params.kept;
  const int64_t blocks = (columns + kPool - 1) / kPool;
  const int64_t seen = params.visible[token];
  // A row of the request's own groups ranks the blocks its visible groups lie in; a row of
  // candidate columns ranks every column.
  const int64_t seen_blocks = (seen + kPool - 1) / kPool;
  const int64_t ranked = kCandidates > 0 ? blocks : (blocks < seen_blocks ? blocks : seen_blocks);
  const size_t element = kBf16 ? 2 : 4;
  const uint8_t* base =
      static_cast<const uint8_t*>(params.scores) + static_cast<size_t>(token * columns) * element;
  const Row<kBf16> row{base, columns, kPool > 1 && seen > 0 ? (seen - 1) / kPool : -1,
                       reinterpret_cast<uintptr_t>(base) % 16 == 0};
  const int32_t* named =
      kCandidates > 0 ? params.candidates + token * (columns / kBlockGroups) : nullptr;
  int32_t* out = params.out + token * kept;

  // ---- A row of no more ranked blocks than slots keeps them all, in column order. ----
  if (ranked <= kept) {
    for (int64_t slot = rank * kThreads + threadIdx.x; slot < kept;
         slot += int64_t{ranks} * kThreads) {
      out[slot] = slot < ranked ? entry_of(monotone(row.pooled(slot)),
                                           static_cast<uint32_t>(slot), named)
                                : -1;
    }
    return;
  }

  // The CTAs that take a slice of the row: the cluster is sized for the columns, but a row of
  // fewer ranked blocks (a short request in a long table) is split over no more CTAs than give
  // each the least slice, and the others take empty slices and only meet the cluster's
  // barriers. Every CTA of the row computes the same count.
  int active = ranks;
  while (active > 1 && ranked / active < kLeastSlice) {
    active /= 2;
  }

  // This CTA's slice of the ranked blocks, empty past the active CTAs.
  const int64_t lo = rank < active ? ranked * rank / active : ranked;
  const int64_t hi = rank < active ? ranked * (rank + 1) / active : ranked;

  for (int b = threadIdx.x; b < 3 * kBins; b += kThreads) {
    (&shared.local[0][0])[b] = 0;
  }
  __syncthreads();

  // ---- Level 1: the top 11 bits of every key, the slice cached when it fits. ----
  const bool cached = hi - lo <= kCacheCapacity;
  walk(Source::Global, row, shared, lo, hi, 0,
       [&](const uint32_t (&key)[kPerThread], const uint32_t (&block)[kPerThread],
           uint32_t valid) {
#pragma unroll
         for (int j = 0; j < kPerThread; ++j) {
           if ((valid >> j) & 1u) {
             atomicAdd(&shared.local[0][key[j] >> kShift1], 1u);
             if (cached) {
               shared.buffer[block[j] - lo] = key[j];
             }
           }
         }
       });

  // The cluster's sums of one level: every CTA reads the active CTAs' histograms. A CTA writes no
  // histogram of a level after its sum, so the reads need no second synchronization, and the
  // last one before the kernel ends keeps every CTA's memory alive for the others' reads.
  const auto sum_level = [&](int level, int bins) {
    cluster.sync();
    for (int b = threadIdx.x; b < bins; b += kThreads) {
      uint32_t sum = 0;
      for (int r = 0; r < active; ++r) {
        sum += cluster.map_shared_rank(&shared.local[level][0], r)[b];
      }
      shared.total[level][b] = sum;
    }
    __syncthreads();
  };

  sum_level(0, kBins);
  uint32_t bin1 = 0;
  uint32_t above1 = 0;
  threshold_bin(shared, shared.total[0], kBins, static_cast<uint32_t>(kept), bin1, above1);

  // ---- Where the further levels read: the cache, the collected keys, or the row again. ----
  Source source = Source::Cached;
  int64_t collected = 0;
  if (!cached) {
    uint32_t mine = 0;
    for (int b = threadIdx.x; b < kBins; b += kThreads) {
      mine += b >= static_cast<int>(bin1) ? shared.local[0][b] : 0;
    }
    const uint32_t count = block_sum(shared, mine);
    source = count <= kCollectCapacity ? Source::Collected : Source::Global;
  }
  if (source == Source::Collected) {
    uint32_t* keys = shared.buffer;
    uint32_t* indices = shared.buffer + kCollectCapacity;
    walk(Source::Global, row, shared, lo, hi, 0,
         [&](const uint32_t (&key)[kPerThread], const uint32_t (&block)[kPerThread],
             uint32_t valid) {
           uint32_t taken = 0;
#pragma unroll
           for (int j = 0; j < kPerThread; ++j) {
             taken |= static_cast<uint32_t>(((valid >> j) & 1u) && (key[j] >> kShift1) >= bin1)
                      << j;
           }
           // A tile of no key at or above the bin appends nothing.
           if (__syncthreads_or(taken != 0) == 0) {
             return;
           }
           uint32_t total = 0;
           uint32_t at = static_cast<uint32_t>(collected) +
                         block_scan(shared, static_cast<uint32_t>(__popc(taken)), total);
#pragma unroll
           for (int j = 0; j < kPerThread; ++j) {
             if ((taken >> j) & 1u) {
               keys[at] = key[j];
               indices[at] = block[j];
               ++at;
             }
           }
           collected += total;
         });
    __syncthreads();
  }

  // ---- Level 2: bits 20..10 of the keys of bin 1. ----
  walk(source, row, shared, lo, hi, collected,
       [&](const uint32_t (&key)[kPerThread], const uint32_t (&)[kPerThread], uint32_t valid) {
#pragma unroll
         for (int j = 0; j < kPerThread; ++j) {
           if (((valid >> j) & 1u) && (key[j] >> kShift1) == bin1) {
             atomicAdd(&shared.local[1][(key[j] >> kShift2) & kMask2], 1u);
           }
         }
       });
  sum_level(1, kBins);
  uint32_t bin2 = 0;
  uint32_t above2 = 0;
  const uint32_t need2 = static_cast<uint32_t>(kept) - above1;
  threshold_bin(shared, shared.total[1], kBins, need2, bin2, above2);

  // ---- Level 3: bits 9..0 of the keys of bins 1 and 2: the threshold key itself. ----
  const uint32_t prefix = (bin1 << (kShift1 - kShift2)) | bin2;
  walk(source, row, shared, lo, hi, collected,
       [&](const uint32_t (&key)[kPerThread], const uint32_t (&)[kPerThread], uint32_t valid) {
#pragma unroll
         for (int j = 0; j < kPerThread; ++j) {
           if (((valid >> j) & 1u) && (key[j] >> kShift2) == prefix) {
             atomicAdd(&shared.local[2][key[j] & kMask3], 1u);
           }
         }
       });
  sum_level(2, 1024);
  uint32_t bin3 = 0;
  uint32_t above3 = 0;
  const uint32_t need3 = need2 - above2;
  threshold_bin(shared, shared.total[2], 1024, need3, bin3, above3);
  const uint32_t threshold = (prefix << kShift2) | bin3;
  const uint32_t above = above1 + above2 + above3;

  // ---- This CTA's keys above the threshold and equal to it, and the lower ranks'. ----
  uint32_t greater = 0;
  for (int b = threadIdx.x; b < kBins; b += kThreads) {
    greater += b > static_cast<int>(bin1) ? shared.local[0][b] : 0;
    greater += b > static_cast<int>(bin2) ? shared.local[1][b] : 0;
    greater += b < 1024 && b > static_cast<int>(bin3) ? shared.local[2][b] : 0;
  }
  greater = block_sum(shared, greater);
  if (threadIdx.x == 0) {
    shared.published[0] = greater;
    shared.published[1] = shared.local[2][bin3];
  }
  cluster.sync();
  uint32_t greater_before = 0;
  uint32_t equal_before = 0;
  for (int r = 0; r < rank && r < active; ++r) {
    const uint32_t* other = cluster.map_shared_rank(&shared.published[0], r);
    greater_before += other[0];
    equal_before += other[1];
  }

  // ---- The kept entries in column order: the greater keys from this CTA's first slot, the
  // ties after every greater key and the lower ranks' ties while slots remain. ----
  uint32_t greater_run = greater_before;
  uint32_t equal_run = above + equal_before;
  const uint32_t slots = static_cast<uint32_t>(kept);
  walk(source, row, shared, lo, hi, collected,
       [&](const uint32_t (&key)[kPerThread], const uint32_t (&block)[kPerThread],
           uint32_t valid) {
         uint32_t over = 0;
         uint32_t tie = 0;
#pragma unroll
         for (int j = 0; j < kPerThread; ++j) {
           const bool live = (valid >> j) & 1u;
           over |= static_cast<uint32_t>(live && key[j] > threshold) << j;
           tie |= static_cast<uint32_t>(live && key[j] == threshold) << j;
         }
         if (__syncthreads_or((over | tie) != 0) == 0) {
           return;
         }
         // The counts of both kinds in one word: a tile holds at most 8192 of each.
         uint32_t total = 0;
         const uint32_t before =
             block_scan(shared, static_cast<uint32_t>(__popc(over)) |
                                    (static_cast<uint32_t>(__popc(tie)) << 16),
                        total);
         uint32_t at_over = greater_run + (before & 0xFFFFu);
         uint32_t at_tie = equal_run + (before >> 16);
#pragma unroll
         for (int j = 0; j < kPerThread; ++j) {
           if ((over >> j) & 1u) {
             out[at_over++] = entry_of(key[j], block[j], named);
           } else if ((tie >> j) & 1u) {
             if (at_tie < slots) {
               out[at_tie] = entry_of(key[j], block[j], named);
             }
             ++at_tie;
           }
         }
         greater_run += total & 0xFFFFu;
         equal_run += total >> 16;
       });

  // Every CTA's memory stays alive until the cluster's reads of it are done.
  cluster.sync();
}

// The cluster a row is split over: rows of a short table are one CTA's; a long row is split so
// that the rows' clusters fill the device, at most 16 CTAs and a slice of at least 8192 blocks
// each.
int cluster_size(int64_t tokens, int64_t blocks, int64_t multiprocessors, int largest) {
  int size = 1;
  while (size * 2 <= largest && tokens * size * 2 <= multiprocessors &&
         blocks / (size * 2) >= kLeastSlice) {
    size *= 2;
  }
  return size;
}

// The largest cluster the device can hold resident with the kernel's shared memory, once per
// device and score type.
template <bool kBf16>
int largest_cluster(int device, cudaStream_t stream) {
  static int known[64] = {};
  if (device >= 0 && device < 64 && known[device] != 0) {
    return known[device];
  }
  int largest = 8;
  if (cudaFuncSetAttribute(select_kernel<kBf16>, cudaFuncAttributeNonPortableClusterSizeAllowed,
                           1) == cudaSuccess) {
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(kMostCluster, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = sizeof(Shared);
    config.stream = stream;
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = kMostCluster;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, select_kernel<kBf16>, &config) == cudaSuccess &&
        clusters > 0) {
      largest = kMostCluster;
    }
  }
  cudaGetLastError();
  if (device >= 0 && device < 64) {
    known[device] = largest;
  }
  return largest;
}

int refuse(uniserve_index_select_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

// The launch of the instantiation of the call's score type: each row one cluster of the size
// `cluster_size` chooses.
template <bool kBf16>
int launch(uniserve_index_select_call* call, cudaStream_t stream) {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }
  status = cudaFuncSetAttribute(select_kernel<kBf16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(sizeof(Shared)));
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }
  const int64_t blocks = (call->columns + kPool - 1) / kPool;
  const int size = cluster_size(call->tokens, blocks, call->multiprocessors,
                                largest_cluster<kBf16>(device, stream));
  if (call->tokens * size > INT32_MAX) {
    return refuse(call, "the rows' clusters exceed the grid");
  }

  const Params params{call->scores, static_cast<const uint32_t*>(call->visible),
                      static_cast<const int32_t*>(call->candidates),
                      static_cast<int32_t*>(call->out), call->columns, call->kept};
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(call->tokens * size), 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = sizeof(Shared);
  config.stream = stream;
  cudaLaunchAttribute attribute = {};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = static_cast<unsigned>(size);
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  status = cudaLaunchKernelEx(&config, select_kernel<kBf16>, params);
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }
  return 0;
}

}  // namespace

extern "C" int uniserve_index_select(uniserve_index_select_call* call, void* stream_handle) {
  if (call == nullptr) {
    return 1;
  }
  std::memset(call->error, 0, sizeof(call->error));
  if (call->pool != kPool || call->candidate_block != kCandidates ||
      (call->scores_dtype != UNISERVE_DTYPE_F32 && call->scores_dtype != UNISERVE_DTYPE_BF16)) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "this module is compiled for blocks of %lld over candidates of %lld of FP32 or "
                  "BF16 scores, and was handed %lld, %lld and dtype %lld",
                  static_cast<long long>(kPool), static_cast<long long>(kCandidates),
                  static_cast<long long>(call->pool),
                  static_cast<long long>(call->candidate_block),
                  static_cast<long long>(call->scores_dtype));
    return refuse(call, message);
  }
  if (call->tokens < 0 || call->columns <= 0 || call->columns > INT32_MAX || call->kept <= 0 ||
      call->kept > (call->columns + kPool - 1) / kPool) {
    return refuse(call, "a selection keeps between one and every block of rows of at most 2^31 "
                        "columns");
  }
  if ((kCandidates > 0) != (call->candidates != nullptr) ||
      call->columns % kBlockGroups != 0) {
    return refuse(call, "candidates are passed exactly when the module maps through them, and "
                        "cover the columns in whole blocks");
  }
  if (call->scores == nullptr || call->visible == nullptr || call->out == nullptr) {
    return refuse(call, "the scores, the visible counts and the output are device addresses");
  }
  if (call->tokens == 0) {
    return 0;
  }
  // The slots a tile counts are 16-bit fields of one word.
  if (call->kept >= (int64_t{1} << 30)) {
    return refuse(call, "a selection keeps fewer than 2^30 slots");
  }

  const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  return call->scores_dtype == UNISERVE_DTYPE_BF16 ? launch<true>(call, stream)
                                                    : launch<false>(call, stream);
}
