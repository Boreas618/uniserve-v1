// SPDX-License-Identifier: Apache-2.0

// One token per row drawn from the row's shaped logits under its top-k width,
// nucleus threshold and min-p threshold, without ordering the vocabulary.
//
// The rule is the operator's (`ops::sampler::SamplingFromLogits`). In the row's
// descending order (the order-preserving key of each FP32 logit, equal keys in
// ascending token id), each of the three filters keeps a prefix:
//
// - the top-k width keeps the first `limit` positions;
// - the nucleus keeps position j while the mass of the positions before it is
//   at most `top_p` of the top-k set's mass;
// - min-p keeps the positions whose logit is at least `max + ln(min_p)`.
//
// Their intersection is the shortest of the three prefixes, so the kept set is
// named by one boundary: every key above `bound`, and the first `quota` tokens
// by id whose key equals it. The draw walks the vocabulary in ascending token
// id and takes the first kept token whose cumulative mass exceeds
// `draw * total`: the token an inverse CDF over the kept candidates in token
// order selects.
//
// A mass is `exp(x - max)` in FP32, held as an integer in units of 2^-40 of the
// largest logit's, so every sum is exact and independent of the order a block
// adds it in: the nucleus's comparison and the draw's cumulative agree to the
// unit, a launch is deterministic, and a draw below one always lands on a kept
// token. Sums stay below 2^64 for any vocabulary the kernel takes.
//
// One block owns one row, and every pass reads the row from global memory (the
// shaping launch before this one leaves it in L2) in chunks of 128 tokens a
// warp reads as one float4 per lane. The boundaries are found by radix
// selection over the keys, digits of 11, 11 and 10 bits from the most
// significant end, a histogram of counts per digit for the top-k width and of
// masses for the nucleus. When the first digit already bounds the candidates
// to `kCapacity` tokens - a top-k width or a nucleus of a few thousand tokens
// at most - the block lists them in shared memory, in token order, and selects
// and draws there: two passes over the row for a top-k width, three for a
// nucleus alone. Otherwise the selection continues digit by digit over the
// row, and the draw is one more pass.
//
// The module is a cubin: the package's manifest defines the warps of a block
// (`UNISERVE_WARPS`), and the kernel reads its arguments in the operator's
// packed order. Launches synchronize nothing, so CUDA graphs capture them.

#include <cuda_runtime.h>

#include <cstdint>

#if !defined(UNISERVE_WARPS)
#error "the package's manifest defines UNISERVE_WARPS"
#endif

namespace {

using u32 = unsigned int;
using u64 = unsigned long long;

constexpr int kWarp = 32;
constexpr int kWarps = UNISERVE_WARPS;
constexpr int kThreads = kWarps * kWarp;
constexpr u32 kFull = 0xffffffffu;

// The tokens one warp reads at a time, a float4 per lane, and the chunks a warp
// has in flight at once: a pass over a row is bound by its loads' latency.
constexpr int kChunk = 4 * kWarp;
constexpr int kUnroll = 8;

// The digits of a key, most significant first: 11, 11 and 10 bits.
constexpr int kLevels = 3;
constexpr int kBins = 2048;
constexpr int kBinsPerThread = kBins / kThreads;
static_assert(kBins % kThreads == 0, "a block's threads split the bins evenly");

// The candidates the block lists in shared memory, two per thread.
constexpr int kCapacity = 2048;
constexpr int kListedPerThread = kCapacity / kThreads;
static_assert(kCapacity % kThreads == 0, "a block's threads split the list evenly");

// The widest row: the draw's segments of `kUnroll` chunks are at most one per
// thread. Its masses sum below 2^60.
constexpr int kMaxVocabulary = kThreads * kUnroll * kChunk;
static_assert(kThreads <= kBins, "a segment's sums fit the histogram");

__device__ __forceinline__ int digit_shift(int level) {
  return level == 0 ? 21 : (level == 1 ? 10 : 0);
}

__device__ __forceinline__ u32 digit_mask(int level) { return level == 2 ? 0x3ffu : 0x7ffu; }

// The parameter columns of a row (`proto::module::sampler`).
constexpr int kParameterColumns = 8;
constexpr int kTopK = 1;
constexpr int kTopP = 2;
constexpr int kMinP = 3;

// The keys of the infinities: a key above `kPositiveInfinity` is a positive
// NaN's, a key below `kNegativeInfinity` a negative NaN's.
constexpr u32 kPositiveInfinity = 0xff800000u;
constexpr u32 kNegativeInfinity = 0x007fffffu;

// A mass's unit: 2^-40 of the largest logit's.
constexpr float kMassScale = 1099511627776.0f;

/// The order-preserving key of an FP32 value: negatives invert, positives set
/// the sign bit, so keys compare as the values do and `+0.0` sits above `-0.0`.
__device__ __forceinline__ u32 key_of(float value) {
  const u32 bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/// The FP32 value of a key.
__device__ __forceinline__ float value_of(u32 key) {
  return __uint_as_float((key & 0x80000000u) ? (key ^ 0x80000000u) : ~key);
}

/// A logit's mass: `exp(x - max)` in FP32, in units of 2^-40, rounded to the
/// nearest unit. The largest logit holds exactly 2^40.
__device__ __forceinline__ u64 mass_of(float value, float maximum) {
  return __float2ull_rn(expf(value - maximum) * kMassScale);
}

/// `floor(fraction * total)` for a fraction in [0, 1): exact, since an FP32
/// fraction times 2^64 is an integer.
__device__ __forceinline__ u64 share(u64 total, float fraction) {
  const u64 scaled = static_cast<u64>(static_cast<double>(fraction) * 18446744073709551616.0);
  return __umul64hi(total, scaled);
}

/// One digit's histogram. A mass is added as two 32-bit words, the low word's
/// carries into the high one, since a 32-bit shared atomic is one instruction
/// and a contended 64-bit one is not; `masses` holds their sum once a pass
/// ends. The draw reuses `counts` and `masses` for its segments.
struct Histogram {
  u32 counts[kBins];
  u32 mass_low[kBins];
  u32 mass_high[kBins];
  u64 masses[kBins];
};

/// The listed candidates: their keys and ids in token order, and the order the
/// selection sorts, `~key << 32 | id` ascending (keys descending, equal keys in
/// ascending id).
struct List {
  u32 keys[kCapacity];
  u32 ids[kCapacity];
  u64 order[kCapacity];
};

/// The block's working memory.
struct Shared {
  union {
    Histogram histogram;
    List list;
  };
  // One word per warp for the block's scans.
  u64 warp_words[kWarps];
  u32 warp_counts[kWarps];
  // The values one thread found for the whole block.
  u64 found_mass;
  u32 found_count;
  int found_bin;
};

/// Loads the lane's four tokens of `chunk`: one float4 when the row is 16-byte
/// aligned and the chunk lies inside it, else one token at a time, with zero
/// past the vocabulary.
__device__ __forceinline__ void load_chunk(const float* row, int vocabulary, int chunk,
                                           float (&out)[4]) {
  const int first = chunk * kChunk + 4 * (threadIdx.x % kWarp);
  if ((reinterpret_cast<uintptr_t>(row) & 15u) == 0 && first + 3 < vocabulary) {
    const float4 four = *reinterpret_cast<const float4*>(row + first);
    out[0] = four.x;
    out[1] = four.y;
    out[2] = four.z;
    out[3] = four.w;
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) out[j] = first + j < vocabulary ? row[first + j] : 0.0f;
  }
}

/// Calls `visit(token, value)` for every token of the row, each warp reading
/// the chunks `warp, warp + kWarps, ...`, `kUnroll` of them at a time.
template <typename Visit>
__device__ __forceinline__ void for_each_token(const float* row, int vocabulary, Visit&& visit) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int chunks = (vocabulary + kChunk - 1) / kChunk;
  for (int base = warp; base < chunks; base += kWarps * kUnroll) {
    float values[kUnroll][4];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) load_chunk(row, vocabulary, base + u * kWarps, values[u]);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int first = (base + u * kWarps) * kChunk + 4 * lane;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (first + j < vocabulary) visit(first + j, values[u][j]);
      }
    }
  }
}

/// The exclusive prefix sum of `value` over the block's threads in thread
/// order, and the block's total in `total`. Every thread calls it.
template <typename T>
__device__ __forceinline__ T exclusive_scan(T value, T* warp_words, T& total) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  __shared__ T block_total;

  T inclusive = value;
#pragma unroll
  for (int offset = 1; offset < kWarp; offset *= 2) {
    const T other = __shfl_up_sync(kFull, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  // The words may still be read by the block's previous scan.
  __syncthreads();
  if (lane == kWarp - 1) warp_words[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const T own = lane < kWarps ? warp_words[lane] : T(0);
    T before = own;
#pragma unroll
    for (int offset = 1; offset < kWarp; offset *= 2) {
      const T other = __shfl_up_sync(kFull, before, offset);
      if (lane >= offset) before += other;
    }
    if (lane < kWarps) warp_words[lane] = before - own;
    if (lane == kWarp - 1) block_total = before;
  }
  __syncthreads();
  total = block_total;
  return warp_words[warp] + inclusive - value;
}

/// The sum of `value` over the block's threads.
template <typename T>
__device__ __forceinline__ T block_sum(T value, T* warp_words) {
  T total;
  exclusive_scan<T>(value, warp_words, total);
  return total;
}

/// Zeroes the histogram.
__device__ __forceinline__ void clear(Histogram& histogram) {
  for (int bin = threadIdx.x; bin < kBins; bin += kThreads) {
    histogram.counts[bin] = 0;
    histogram.mass_low[bin] = 0;
    histogram.mass_high[bin] = 0;
    histogram.masses[bin] = 0;
  }
  __syncthreads();
}

/// Adds `mass` to `bin`'s two words.
__device__ __forceinline__ void add_mass(Histogram& histogram, int bin, u64 mass) {
  const u32 low = static_cast<u32>(mass);
  const u32 high = static_cast<u32>(mass >> 32);
  const u32 before = atomicAdd(&histogram.mass_low[bin], low);
  const u32 carry = before + low < before ? 1u : 0u;
  if (high + carry > 0) atomicAdd(&histogram.mass_high[bin], high + carry);
}

/// Folds each bin's two words into its mass, once every thread has added.
__device__ __forceinline__ void fold_masses(Histogram& histogram) {
  __syncthreads();
  for (int bin = threadIdx.x; bin < kBins; bin += kThreads) {
    const u64 high = static_cast<u64>(histogram.mass_high[bin]) << 32;
    histogram.masses[bin] = high + histogram.mass_low[bin];
  }
  __syncthreads();
}

/// The bins a thread owns, in descending digit order: thread 0 owns the top
/// ones, so the block's scans in thread order run down the row's order.
__device__ __forceinline__ int owned_bin(int slot) {
  return kBins - 1 - (threadIdx.x * kBinsPerThread + slot);
}

/// After a count histogram: the bin holding the `rank`-th largest of the
/// counted tokens (one-based), and in `above` how many lie in bins above it.
__device__ int count_crossing(Shared& shared, u32 rank, u32& above) {
  const Histogram& histogram = shared.histogram;
  u32 own = 0;
#pragma unroll
  for (int slot = 0; slot < kBinsPerThread; ++slot) own += histogram.counts[owned_bin(slot)];
  u32 total;
  u32 before = exclusive_scan<u32>(own, shared.warp_counts, total);
#pragma unroll
  for (int slot = 0; slot < kBinsPerThread; ++slot) {
    const int bin = owned_bin(slot);
    const u32 count = histogram.counts[bin];
    if (before < rank && rank <= before + count) {
      shared.found_bin = bin;
      shared.found_count = before;
    }
    before += count;
  }
  __syncthreads();
  above = shared.found_count;
  const int bin = shared.found_bin;
  __syncthreads();
  return bin;
}

/// After a mass histogram of tokens preceded by `above` mass: the lowest bin
/// with mass whose first token's predecessors hold at most `threshold`, and in
/// `above` that predecessors' mass. The last token of positive mass the
/// threshold keeps lies in that bin.
__device__ int mass_crossing(Shared& shared, u64 threshold, u64& above) {
  const Histogram& histogram = shared.histogram;
  u64 own = 0;
#pragma unroll
  for (int slot = 0; slot < kBinsPerThread; ++slot) own += histogram.masses[owned_bin(slot)];
  if (threadIdx.x == 0) shared.found_bin = kBins;
  u64 total;
  u64 before = above + exclusive_scan<u64>(own, shared.warp_words, total);
  int lowest = kBins;
  u64 start = 0;
#pragma unroll
  for (int slot = 0; slot < kBinsPerThread; ++slot) {
    const int bin = owned_bin(slot);
    const u64 mass = histogram.masses[bin];
    if (mass > 0 && before <= threshold) {
      lowest = bin;
      start = before;
    }
    before += mass;
  }
  if (lowest < kBins) atomicMin(&shared.found_bin, lowest);
  __syncthreads();
  const int bin = shared.found_bin;
  if (lowest == bin) shared.found_mass = start;
  __syncthreads();
  above = shared.found_mass;
  __syncthreads();
  return bin;
}

/// How many tokens the histogram counts in `bin` and the bins above it.
__device__ u32 counted_from(Shared& shared, int bin) {
  u32 own = 0;
#pragma unroll
  for (int slot = 0; slot < kBinsPerThread; ++slot) {
    if (owned_bin(slot) >= bin) own += shared.histogram.counts[owned_bin(slot)];
  }
  return block_sum<u32>(own, shared.warp_counts);
}

/// Whether `key` shares `prefix`'s digits above `level`.
__device__ __forceinline__ bool under(u32 key, u32 prefix, int level) {
  if (level == 0) return true;
  const int high = digit_shift(level - 1);
  return (key >> high) == (prefix >> high);
}

/// Writes the row's outputs: its token, or token 0 and an unusable flag.
__device__ __forceinline__ void finish(int32_t* tokens, int32_t* valid, int row, int token,
                                       bool usable) {
  tokens[row] = usable ? token : 0;
  valid[row] = usable ? 1 : 0;
}

/// Lists every token whose key is at least `floor` in shared memory as its
/// sort word, `~key << 32 | id`, and answers how many. The caller has counted
/// them: at most `kCapacity`. Each warp appends a chunk's candidates at once,
/// so the list's order is the warps' and the selection sorts it.
__device__ int list_candidates(Shared& shared, const float* row, int vocabulary, u32 floor) {
  List& list = shared.list;
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int chunks = (vocabulary + kChunk - 1) / kChunk;
  if (threadIdx.x == 0) shared.found_count = 0;
  __syncthreads();

  for (int base = warp; base < chunks; base += kWarps * kUnroll) {
    float values[kUnroll][4];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) load_chunk(row, vocabulary, base + u * kWarps, values[u]);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int first = (base + u * kWarps) * kChunk + 4 * lane;
      u32 flags = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (first + j < vocabulary && key_of(values[u][j]) >= floor) flags |= 1u << j;
      }
      // Most chunks hold no candidate.
      if (!__any_sync(kFull, flags != 0)) continue;
      const u32 count = __popc(flags);
      u32 inclusive = count;
#pragma unroll
      for (int offset = 1; offset < kWarp; offset *= 2) {
        const u32 other = __shfl_up_sync(kFull, inclusive, offset);
        if (lane >= offset) inclusive += other;
      }
      u32 slot = 0;
      if (lane == kWarp - 1) slot = atomicAdd(&shared.found_count, inclusive);
      slot = __shfl_sync(kFull, slot, kWarp - 1) + inclusive - count;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (flags >> j & 1u) {
          const u64 key = ~key_of(values[u][j]);
          list.order[slot] = (key << 32) | static_cast<u32>(first + j);
          ++slot;
        }
      }
    }
  }
  __syncthreads();
  return static_cast<int>(shared.found_count);
}

/// Sorts the first `count` words of `words` ascending, padding them to a power
/// of two with words above every other.
__device__ void sort_words(u64* words, int count) {
  int width = 1;
  while (width < count) width *= 2;
  for (int slot = count + threadIdx.x; slot < width; slot += kThreads) words[slot] = ~0ull;
  __syncthreads();
  // Bitonic: each stage compares the pairs `stride` apart, in runs of `size`
  // alternately ascending and descending until the last.
  for (int size = 2; size <= width; size *= 2) {
    for (int stride = size / 2; stride > 0; stride /= 2) {
      for (int slot = threadIdx.x; slot < width; slot += kThreads) {
        const int partner = slot ^ stride;
        if (partner > slot) {
          const u64 a = words[slot];
          const u64 b = words[partner];
          const bool ascending = (slot & size) == 0;
          if ((a > b) == ascending) {
            words[slot] = b;
            words[partner] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

/// Selects and draws among the listed candidates, which hold every token the
/// row keeps. `limit` is the top-k width; the nucleus keeps `top_p` of the
/// listed top-k set's mass when `share_top_k` holds, and `nucleus` mass
/// otherwise (`~0` for none).
__device__ void draw_listed(Shared& shared, int listed, u32 limit, bool share_top_k, float top_p,
                            u64 nucleus, bool min_p_applies, float threshold, float maximum,
                            float draw, int32_t* tokens, int32_t* valid, int row) {
  List& list = shared.list;
  sort_words(list.order, listed);

  // Each thread's positions in the descending order, and their masses: the
  // top-k width's positions only.
  const int first = threadIdx.x * kListedPerThread;
  float logits[kListedPerThread];
  u64 masses[kListedPerThread];
  u64 own = 0;
#pragma unroll
  for (int i = 0; i < kListedPerThread; ++i) {
    const int position = first + i;
    const bool counted = position < listed && static_cast<u32>(position) < limit;
    logits[i] = value_of(~static_cast<u32>(list.order[position] >> 32));
    masses[i] = counted ? mass_of(logits[i], maximum) : 0ull;
    own += masses[i];
  }

  // The nucleus keeps a position while the mass before it is at most its
  // share; min-p keeps the logits at or above its threshold. With the width,
  // each keeps a prefix, so the kept set is the first `kept` positions.
  u64 top_k_mass;
  u64 before = exclusive_scan<u64>(own, shared.warp_words, top_k_mass);
  if (share_top_k) nucleus = share(top_k_mass, top_p);
  u32 kept = 0;
#pragma unroll
  for (int i = 0; i < kListedPerThread; ++i) {
    const int position = first + i;
    const bool inside = position < listed && static_cast<u32>(position) < limit;
    const bool meets = !min_p_applies || logits[i] >= threshold;
    kept += inside && before <= nucleus && meets;
    before += masses[i];
  }
  kept = block_sum<u32>(kept, shared.warp_counts);

  // The kept positions in token order: their keys and ids move out of the
  // sort words, which then sort `id << 32 | position`.
  for (int position = threadIdx.x; position < static_cast<int>(kept); position += kThreads) {
    list.keys[position] = ~static_cast<u32>(list.order[position] >> 32);
    list.ids[position] = static_cast<u32>(list.order[position]);
  }
  __syncthreads();
  for (int position = threadIdx.x; position < static_cast<int>(kept); position += kThreads) {
    const u64 id = list.ids[position];
    list.order[position] = (id << 32) | static_cast<u32>(position);
  }
  __syncthreads();
  sort_words(list.order, static_cast<int>(kept));

  // The draw: the first kept token by id whose cumulative mass exceeds the
  // draw's share of the kept mass.
  u64 drawn[kListedPerThread];
  u64 mass_own = 0;
#pragma unroll
  for (int i = 0; i < kListedPerThread; ++i) {
    const int slot = first + i;
    drawn[i] = slot < static_cast<int>(kept)
                   ? mass_of(value_of(list.keys[static_cast<u32>(list.order[slot])]), maximum)
                   : 0ull;
    mass_own += drawn[i];
  }
  u64 total;
  u64 start = exclusive_scan<u64>(mass_own, shared.warp_words, total);
  const u64 target = share(total, fminf(fmaxf(draw, 0.0f), 0.99999994f));
#pragma unroll
  for (int i = 0; i < kListedPerThread; ++i) {
    if (drawn[i] > 0 && start <= target && target < start + drawn[i]) {
      finish(tokens, valid, row, static_cast<int>(list.order[first + i] >> 32), true);
    }
    start += drawn[i];
  }
}

/// The nucleus's boundary over the row, digit by digit from `level`: the
/// lowest key and how many tokens holding it the nucleus keeps, out of the
/// top-k set that `bound` and `quota` name (`count` tokens hold `bound`).
/// `prefix`, `above` and `nucleus` carry the digits a caller has already
/// resolved; at level 0 the nucleus is taken as `top_p` of the top-k set's mass.
__device__ void nucleus_boundary(Shared& shared, const float* values, int vocabulary,
                                 float maximum, u32& bound, u32& quota, u32 count, float top_p,
                                 int level, u32 prefix, u64 above, u64 nucleus) {
  Histogram& histogram = shared.histogram;
  // The tokens holding the top-k bound past its quota are counted in the
  // histograms and taken back out of the bound's bin: they all hold its mass.
  const u32 excess = count > quota ? count - quota : 0u;
  const u64 bound_mass = mass_of(value_of(bound), maximum);
  const u32 floor = bound;
  int bin = 0;
  for (; level < kLevels; ++level) {
    clear(histogram);
    for_each_token(values, vocabulary, [&](int, float value) {
      const u32 key = key_of(value);
      if (key >= floor && under(key, prefix, level)) {
        const u64 mass = mass_of(value, maximum);
        const int digit = (key >> digit_shift(level)) & digit_mask(level);
        if (mass > 0) add_mass(histogram, digit, mass);
        atomicAdd(&histogram.counts[digit], 1u);
      }
    });
    fold_masses(histogram);
    if (threadIdx.x == 0 && excess > 0 && under(floor, prefix, level)) {
      const int digit = (floor >> digit_shift(level)) & digit_mask(level);
      histogram.masses[digit] -= excess * bound_mass;
      histogram.counts[digit] -= excess;
    }
    __syncthreads();
    if (level == 0) {
      u64 own = 0;
#pragma unroll
      for (int slot = 0; slot < kBinsPerThread; ++slot) own += histogram.masses[owned_bin(slot)];
      nucleus = share(block_sum<u64>(own, shared.warp_words), top_p);
    }
    bin = mass_crossing(shared, nucleus, above);
    prefix |= static_cast<u32>(bin) << digit_shift(level);
  }
  // The tokens of the boundary key the nucleus keeps: those whose predecessors
  // hold at most the nucleus's mass, of the ones the top-k set holds.
  const u64 mass = mass_of(value_of(prefix), maximum);
  const u32 held = histogram.counts[bin];
  const u64 fits = (nucleus - above) / mass + 1;
  bound = prefix;
  quota = fits < held ? static_cast<u32>(fits) : held;
  __syncthreads();
}

/// Draws over the row: the kept tokens' masses in segments of `kUnroll`
/// chunks, the segment the draw's share of their total falls in, and the token
/// in it, which one warp finds walking the segment in token order.
__device__ void draw_row(Shared& shared, const float* values, int vocabulary, float maximum,
                         u32 bound, u32 quota, float draw, int32_t* tokens, int32_t* valid,
                         int row) {
  Histogram& histogram = shared.histogram;
  const u64 bound_mass = mass_of(value_of(bound), maximum);
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int chunks = (vocabulary + kChunk - 1) / kChunk;
  const int segments = (chunks + kUnroll - 1) / kUnroll;

  // A segment's mass above the bound, and its tokens at the bound.
  clear(histogram);
  for (int segment = warp; segment < segments; segment += kWarps) {
    float lane_values[kUnroll][4];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      load_chunk(values, vocabulary, segment * kUnroll + u, lane_values[u]);
    }
    u64 mass = 0;
    u32 at = 0;
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int first = (segment * kUnroll + u) * kChunk + 4 * lane;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const u32 key = key_of(lane_values[u][j]);
        const bool live = first + j < vocabulary;
        if (live && key > bound) mass += mass_of(lane_values[u][j], maximum);
        at += live && key == bound;
      }
    }
    if (__any_sync(kFull, mass > 0 || at > 0)) {
#pragma unroll
      for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        mass += __shfl_xor_sync(kFull, mass, offset);
        at += __shfl_xor_sync(kFull, at, offset);
      }
    }
    if (lane == 0) {
      histogram.masses[segment] = mass;
      histogram.counts[segment] = at;
    }
  }
  __syncthreads();

  // Each thread's segments, at most one: the tokens at the bound before it,
  // which give it its kept mass, then the kept mass before it.
  const int segment = threadIdx.x;
  const bool owns = segment < segments;
  const u32 at = owns ? histogram.counts[segment] : 0u;
  u32 at_total;
  const u32 at_before = exclusive_scan<u32>(at, shared.warp_counts, at_total);
  u64 mass = 0;
  if (owns) {
    const u32 kept_at = at_before >= quota ? 0u : min(at, quota - at_before);
    mass = histogram.masses[segment] + static_cast<u64>(kept_at) * bound_mass;
  }
  u64 total;
  const u64 before = exclusive_scan<u64>(mass, shared.warp_words, total);
  const u64 target = share(total, fminf(fmaxf(draw, 0.0f), 0.99999994f));
  if (threadIdx.x == 0) shared.found_bin = -1;
  __syncthreads();
  if (mass > 0 && before <= target && target < before + mass) {
    shared.found_bin = segment;
    shared.found_mass = before;
    shared.found_count = at_before;
  }
  __syncthreads();
  const int chosen = shared.found_bin;
  if (warp != 0) return;
  if (chosen < 0) {
    // No kept mass: the largest logit is always kept, so only a defect of the
    // selection reaches here, and the row reports itself unusable.
    if (lane == 0) finish(tokens, valid, row, 0, false);
    return;
  }

  // The chosen segment's chunks in token order, until the cumulative mass
  // passes the target.
  u64 running = shared.found_mass;
  u32 at_rank = shared.found_count;
  for (int chunk = chosen * kUnroll; chunk < min(chunks, (chosen + 1) * kUnroll); ++chunk) {
    float lane_values[4];
    load_chunk(values, vocabulary, chunk, lane_values);
    const int first = chunk * kChunk + 4 * lane;
    u64 masses[4];
    u32 at_lane = 0;
    u32 at_flags = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const u32 key = key_of(lane_values[j]);
      const bool live = first + j < vocabulary;
      masses[j] = live && key > bound ? mass_of(lane_values[j], maximum) : 0ull;
      if (live && key == bound) {
        at_flags |= 1u << j;
        ++at_lane;
      }
    }
    // The tokens at the bound before this lane's, in token order.
    u32 at_inclusive = at_lane;
#pragma unroll
    for (int offset = 1; offset < kWarp; offset *= 2) {
      const u32 other = __shfl_up_sync(kFull, at_inclusive, offset);
      if (lane >= offset) at_inclusive += other;
    }
    u32 lane_rank = at_rank + at_inclusive - at_lane;
    u64 lane_mass = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (at_flags >> j & 1u) {
        if (lane_rank < quota) masses[j] = bound_mass;
        ++lane_rank;
      }
      lane_mass += masses[j];
    }
    u64 inclusive = lane_mass;
#pragma unroll
    for (int offset = 1; offset < kWarp; offset *= 2) {
      const u64 other = __shfl_up_sync(kFull, inclusive, offset);
      if (lane >= offset) inclusive += other;
    }
    u64 cumulative = running + inclusive - lane_mass;
    int token = -1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      cumulative += masses[j];
      if (token < 0 && masses[j] > 0 && cumulative > target) token = first + j;
    }
    const u32 crossed = __ballot_sync(kFull, token >= 0);
    if (crossed != 0) {
      if (lane == __ffs(crossed) - 1) finish(tokens, valid, row, token, true);
      return;
    }
    running += __shfl_sync(kFull, inclusive, kWarp - 1);
    at_rank += __shfl_sync(kFull, at_inclusive, kWarp - 1);
  }
}

}  // namespace

// The arguments are the operator's packed launch parameters: the shaped logits
// and their row stride in elements, the row count, the vocabulary (at most
// `kMaxVocabulary`), each row's parameters (`kParameterColumns` FP32: the
// top-k width at 1, the nucleus threshold at 2 and the min-p threshold at 3),
// each row's uniform draw in [0, 1), and the two outputs: each row's token, and
// 1 for a row whose kept candidates form a distribution. A row that keeps
// none, or whose kept candidates hold a NaN or +inf, reports 0 and holds
// token 0. One block per row.
extern "C" __global__ void __launch_bounds__(kThreads)
    uniserve_sampling_from_logits(const float* logits, int64_t row_stride, int rows, int vocabulary,
                                  const float* parameters, const float* draws, int32_t* tokens,
                                  int32_t* valid) {
  __shared__ Shared shared;
  Histogram& histogram = shared.histogram;

  // A vocabulary past the draw's segments would be drawn from in part, and
  // a cubin has no error channel: the launch fails instead.
  if (vocabulary <= 0 || vocabulary > kMaxVocabulary) __trap();

  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* values = logits + static_cast<int64_t>(row) * row_stride;
  const float top_k = parameters[row * kParameterColumns + kTopK];
  const float top_p = parameters[row * kParameterColumns + kTopP];
  const float min_p = parameters[row * kParameterColumns + kMinP];
  const float draw = draws[row];

  // The top-k width: a width that is not positive keeps the whole row.
  const u32 width = static_cast<u32>(vocabulary);
  const u32 limit =
      top_k > 0.0f ? static_cast<u32>(floorf(fminf(top_k, static_cast<float>(vocabulary)))) : width;
  const bool selects = limit < width;
  const bool nucleus = top_p < 1.0f && top_p == top_p;

  // ---- The largest key, the negative NaNs, and the first digit's counts for
  // a top-k width. ----
  if (selects) clear(histogram);
  u32 largest = 0;
  u32 negative_nans = 0;
  for_each_token(values, vocabulary, [&](int, float value) {
    const u32 key = key_of(value);
    largest = max(largest, key);
    negative_nans += key < kNegativeInfinity;
    if (selects) atomicAdd(&histogram.counts[key >> digit_shift(0)], 1u);
  });
  largest = __reduce_max_sync(kFull, largest);
  negative_nans = __reduce_add_sync(kFull, negative_nans);
  __syncthreads();
  if (threadIdx.x % kWarp == 0) shared.warp_counts[threadIdx.x / kWarp] = largest;
  __syncthreads();
  if (threadIdx.x < kWarp) {
    largest = __reduce_max_sync(kFull, threadIdx.x < kWarps ? shared.warp_counts[threadIdx.x] : 0u);
    if (threadIdx.x == 0) shared.found_count = largest;
  }
  __syncthreads();
  largest = shared.found_count;
  negative_nans = block_sum<u32>(threadIdx.x % kWarp == 0 ? negative_nans : 0u, shared.warp_counts);

  // A row keeps nothing usable when its largest key is +inf or a NaN, when it
  // holds no finite logit, when its width keeps nothing, when its top-k set
  // reaches a negative NaN, when its nucleus is negative, or when its min-p
  // threshold is above its largest logit (or NaN).
  const float maximum = value_of(largest);
  const bool min_p_applies = !(min_p <= 0.0f);
  const float threshold = maximum + logf(min_p);
  const bool usable = largest < kPositiveInfinity && largest > kNegativeInfinity && limit > 0 &&
                      limit <= width - negative_nans && !(top_p < 0.0f) &&
                      (!min_p_applies || maximum >= threshold);
  if (!usable) {
    if (threadIdx.x == 0) finish(tokens, valid, row, 0, false);
    return;
  }

  // The kept prefix so far, every token: the keys above `bound`, and `quota`
  // of the tokens at it, the lowest ids first.
  u32 bound = 0;
  u32 quota = width;

  if (selects) {
    // ---- The top-k width's first digit, and the candidates from it on. ----
    u32 above;
    int bin = count_crossing(shared, limit, above);
    if (above + histogram.counts[bin] <= kCapacity) {
      __syncthreads();
      const u32 floor = static_cast<u32>(bin) << digit_shift(0);
      const int listed = list_candidates(shared, values, vocabulary, floor);
      draw_listed(shared, listed, limit, nucleus, top_p, ~0ull, min_p_applies, threshold, maximum,
                  draw, tokens, valid, row);
      return;
    }

    // The width's boundary digit by digit over the row.
    u32 prefix = static_cast<u32>(bin) << digit_shift(0);
    u32 rank = limit - above;
    u32 count = 0;
    for (int level = 1; level < kLevels; ++level) {
      clear(histogram);
      for_each_token(values, vocabulary, [&](int, float value) {
        const u32 key = key_of(value);
        if (under(key, prefix, level)) {
          atomicAdd(&histogram.counts[(key >> digit_shift(level)) & digit_mask(level)], 1u);
        }
      });
      __syncthreads();
      bin = count_crossing(shared, rank, above);
      rank -= above;
      prefix |= static_cast<u32>(bin) << digit_shift(level);
      count = histogram.counts[bin];
      __syncthreads();
    }
    bound = prefix;
    quota = rank;
    if (nucleus) {
      nucleus_boundary(shared, values, vocabulary, maximum, bound, quota, count, top_p, 0, 0, 0, 0);
    }
  } else if (nucleus) {
    // ---- The nucleus's first digit over the whole row, and the candidates
    // from it on. ----
    clear(histogram);
    for_each_token(values, vocabulary, [&](int, float value) {
      const u32 key = key_of(value);
      const u64 mass = mass_of(value, maximum);
      const int digit = key >> digit_shift(0);
      if (mass > 0) add_mass(histogram, digit, mass);
      atomicAdd(&histogram.counts[digit], 1u);
    });
    fold_masses(histogram);
    u64 own = 0;
#pragma unroll
    for (int slot = 0; slot < kBinsPerThread; ++slot) own += histogram.masses[owned_bin(slot)];
    const u64 kept_mass = share(block_sum<u64>(own, shared.warp_words), top_p);
    u64 above = 0;
    const int bin = mass_crossing(shared, kept_mass, above);
    if (counted_from(shared, bin) <= kCapacity) {
      const u32 floor = static_cast<u32>(bin) << digit_shift(0);
      const int listed = list_candidates(shared, values, vocabulary, floor);
      draw_listed(shared, listed, width, false, top_p, kept_mass, min_p_applies, threshold,
                  maximum, draw, tokens, valid, row);
      return;
    }
    nucleus_boundary(shared, values, vocabulary, maximum, bound, quota, 0, top_p, 1,
                     static_cast<u32>(bin) << digit_shift(0), above, kept_mass);
  }

  // ---- Min-p, against the row's largest logit: every key at or above the
  // threshold's. `-0.0` stands for a zero threshold, which both zeros meet. ----
  if (min_p_applies) {
    const u32 lowest = key_of(threshold == 0.0f ? -0.0f : threshold);
    if (lowest > bound) {
      bound = lowest;
      quota = width;
    }
  }

  draw_row(shared, values, vocabulary, maximum, bound, quota, draw, tokens, valid, row);
}
