// SPDX-License-Identifier: Apache-2.0

// Expert routing: each token's scores from its router logits (a softmax, or a
// sqrt-softplus per expert), the top-k experts by the scores plus an optional
// selection bias, optional renormalization and an optional per-expert scale.
//
// The operator `topk_softmax` documents the arithmetic, and the numbers are the
// ones UniServe's tensor composition defined. One warp owns one token and
// reproduces that composition's FP32 arithmetic order:
//
// - softmax as PyTorch's warp softmax evaluates a row: lane l holds scores
//   l + 32 * i, the maximum and the sum of expf(score - max) reduce by xor
//   butterflies, and each probability is exp / sum; or, compiled for
//   sqrt-softplus (UNISERVE_SQRTSOFTPLUS), each score is
//   sqrtf(log1pf(expf(logit))), the logit itself above PyTorch's softplus
//   threshold of 20;
// - the k largest selection keys in descending order, equal ones in ascending
//   expert order, by repeated warp arg-max; a key is the score plus the
//   expert's selection bias when there is one, and a route's weight is the
//   unbiased score;
// - their sum by the xor butterfly over k lanes that PyTorch's row sum of a
//   [tokens, k] tensor uses divides them: clamped below by FLT_EPSILON after a
//   softmax, plus 1e-20 after sqrt-softplus, as the reference router does;
// - a per-expert scale multiplies the result in FP32;
// - a route's expert is written as the slot the logical-to-physical map names,
//   when there is a map. The selection above is over logical ids, so the picks
//   and the order of equal scores do not depend on where the experts lie;
// - with an expert load, each route of a live token (a row below the live
//   count, every row without one) adds one to its logical expert's 64-bit
//   counter, so the counters hold the tokens each expert took however the
//   experts are laid out.
//
// Launches synchronize nothing, so CUDA graphs capture them.
//
// The module is a cubin: the package's manifest defines the descriptor's
// expert count (`UNISERVE_EXPERTS`) and top-k width (`UNISERVE_TOP`), from which
// the kernel derives the scores' per-lane iteration count (the expert count in
// whole warps) and the butterfly width (the top-k width rounded up to a power of
// two), and the warps of a block (`UNISERVE_WARPS`), so the kernel reads its
// arguments in the operator's packed order and nothing else. A block of
// `UNISERVE_WARPS` warps routes as many tokens, which is the grid the operator
// computes from the module's metadata.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

#if !defined(UNISERVE_EXPERTS) || !defined(UNISERVE_TOP) || !defined(UNISERVE_WARPS) || \
    !defined(UNISERVE_SQRTSOFTPLUS)
#error "the package's manifest defines UNISERVE_EXPERTS, UNISERVE_TOP, UNISERVE_WARPS and UNISERVE_SQRTSOFTPLUS"
#endif

namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = UNISERVE_WARPS;

/// The smallest power of two at or above `top`: the width of the butterfly
/// that sums `top` picks.
constexpr int butterfly(int top) {
  int width = 1;
  while (width < top) width *= 2;
  return width;
}

/// The scores a lane walks: the expert count in whole warps.
constexpr int kIterations = (UNISERVE_EXPERTS + kWarp - 1) / kWarp;
constexpr int kTop = butterfly(UNISERVE_TOP);

// Whether the scores are sqrt(softplus(logit)) rather than the row's softmax.
constexpr bool kSqrtSoftplus = UNISERVE_SQRTSOFTPLUS != 0;

// PyTorch's softplus threshold: above it softplus(x) is x itself.
constexpr float kSoftplusThreshold = 20.f;

// What the reference router adds to the selected scores' sum before dividing by it.
constexpr float kRenormalizeEpsilon = 1e-20f;

// Storage kinds of the scores and of the per-expert scale.
constexpr int kAbsent = -1;
constexpr int kFloat = 0;
constexpr int kBFloat16 = 1;
constexpr int kHalf = 2;

__device__ __forceinline__ float load_value(const void* data, int kind, int64_t index) {
  if (kind == kFloat) return static_cast<const float*>(data)[index];
  if (kind == kHalf) return __half2float(static_cast<const __half*>(data)[index]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

}  // namespace

// The arguments are the operator's packed launch parameters: the logits, their
// storage kind, the row stride in elements, the token count, the expert count,
// the top-k width, the renormalize flag, the per-expert scale and its kind, the
// FP32 selection bias (null for none), the two outputs, the logical-to-physical
// expert map (null for none), the count of live token rows (null: every row is
// live) and the expert load the live rows' routes add into (null for none).
extern "C" __global__ void uniserve_topk_softmax(const void* scores, int score_kind,
                                                 int64_t row_stride, int rows, int experts,
                                                 int top, int renormalize, const void* scale,
                                                 int scale_kind, const float* bias, int32_t* ids,
                                                 float* weights, const int32_t* map,
                                                 const int32_t* live, unsigned long long* load) {
  // `experts` and `top` arrive as launch arguments while the iteration count and
  // the butterfly width are compiled into this cubin. A call arriving with a
  // different expert count would reduce over part of the row and write
  // valid-looking ids; a wider top-k would read past the pick arrays. A cubin
  // has no error channel, so a mismatch traps and fails the launch loudly
  // rather than routing wrongly for the rest of the process's life.
  if (experts != UNISERVE_EXPERTS || top > UNISERVE_TOP) {
    __trap();
  }

  const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) return;  // uniform per warp

  // Scores past the expert count are -inf, as PyTorch pads a row.
  float values[kIterations];
#pragma unroll
  for (int it = 0; it < kIterations; ++it) {
    const int expert = lane + it * kWarp;
    values[it] = expert < experts
                     ? load_value(scores, score_kind, static_cast<int64_t>(row) * row_stride + expert)
                     : -INFINITY;
  }

  if constexpr (kSqrtSoftplus) {
    // Each expert's score on its own; a padding lane's stays -inf, which no key beats.
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const float logit = values[it];
      const float softplus = logit > kSoftplusThreshold ? logit : log1pf(expf(logit));
      values[it] = lane + it * kWarp < experts ? sqrtf(softplus) : -INFINITY;
    }
  } else {
    float maximum = values[0];
#pragma unroll
    for (int it = 0; it < kIterations; ++it) maximum = maximum > values[it] ? maximum : values[it];
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      const float other = __shfl_xor_sync(0xffffffffu, maximum, offset);
      maximum = maximum < other ? other : maximum;
    }
    float sum = 0.f;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      values[it] = expf(values[it] - maximum);
      sum += values[it];
    }
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      sum = sum + __shfl_xor_sync(0xffffffffu, sum, offset);
    }
#pragma unroll
    for (int it = 0; it < kIterations; ++it) values[it] = values[it] / sum;
  }

  // The keys the top-k selects by: the scores, plus the selection bias when there is one.
  float keys[kIterations];
#pragma unroll
  for (int it = 0; it < kIterations; ++it) {
    const int expert = lane + it * kWarp;
    keys[it] = bias != nullptr && expert < experts ? values[it] + bias[expert] : values[it];
  }

  // Top-k by repeated warp arg-max over the keys; every lane ends with all k
  // picks and their unbiased scores, in descending order of key with equal keys
  // in ascending expert order.
  float top_value[kTop];
  int top_expert[kTop];
  unsigned taken = 0u;
#pragma unroll
  for (int j = 0; j < kTop; ++j) {
    if (j >= top) {
      top_value[j] = 0.f;
      top_expert[j] = 0;
      continue;
    }
    float best = -INFINITY;
    float best_value = 0.f;
    int best_expert = INT_MAX;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int expert = lane + it * kWarp;
      if (expert < experts && !(taken >> it & 1u) &&
          (keys[it] > best || (keys[it] == best && expert < best_expert))) {
        best = keys[it];
        best_value = values[it];
        best_expert = expert;
      }
    }
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      const float other = __shfl_xor_sync(0xffffffffu, best, offset);
      const float other_value = __shfl_xor_sync(0xffffffffu, best_value, offset);
      const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, offset);
      if (other > best || (other == best && other_expert < best_expert)) {
        best = other;
        best_value = other_value;
        best_expert = other_expert;
      }
    }
    top_value[j] = best_value;
    top_expert[j] = best_expert;
    if (best_expert % kWarp == lane) taken |= 1u << (best_expert / kWarp);
  }

  // Lane j of the first k lanes stores pick j.
  if (lane >= top) return;
  float weight = 0.f;
  int expert = 0;
#pragma unroll
  for (int j = 0; j < kTop; ++j) {
    if (j == lane) {
      weight = top_value[j];
      expert = top_expert[j];
    }
  }
  if (renormalize) {
    // The butterfly over k lanes: at each step lane j adds lane j ^ offset.
    float partial[kTop];
#pragma unroll
    for (int j = 0; j < kTop; ++j) partial[j] = top_value[j];
#pragma unroll
    for (int offset = kTop / 2; offset > 0; offset /= 2) {
      float next[kTop];
#pragma unroll
      for (int j = 0; j < kTop; ++j) next[j] = partial[j] + partial[j ^ offset];
#pragma unroll
      for (int j = 0; j < kTop; ++j) partial[j] = next[j];
    }
    const float total = kSqrtSoftplus ? partial[0] + kRenormalizeEpsilon
                                      : (partial[0] < FLT_EPSILON ? FLT_EPSILON : partial[0]);
    weight = weight / total;
  }
  if (scale_kind != kAbsent) weight = weight * load_value(scale, scale_kind, expert);
  // A padding row of a captured bucket holds no scores: its picks may name no
  // expert, and `moe_sort` drops such an id. The map is read for real ids alone.
  const bool named = expert >= 0 && expert < experts;
  const bool mapped = map != nullptr && named;
  ids[static_cast<int64_t>(row) * top + lane] = mapped ? map[expert] : expert;
  weights[static_cast<int64_t>(row) * top + lane] = weight;

  // The load counts the routes of the batch's own tokens: a captured bucket's
  // rows past the live count route nothing a forward keeps. The counter is the
  // logical expert's, read before the map, and the device-wide atomic keeps
  // every warp's and every replay's additions.
  if (load != nullptr && named && (live == nullptr || row < live[0])) {
    atomicAdd(load + expert, 1ull);
  }
}
