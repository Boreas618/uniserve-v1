// SPDX-License-Identifier: Apache-2.0

// Expert routing: softmax over each token's router scores, the top-k experts,
// optional renormalization and an optional per-expert scale.
//
// The operator `topk_softmax` documents the contract, and the numbers are the
// ones UniServe's tensor composition defined. One warp owns one token and
// reproduces that composition's FP32 arithmetic order:
//
// - softmax as PyTorch's warp softmax evaluates a row: lane l holds scores
//   l + 32 * i, the maximum and the sum of expf(score - max) reduce by xor
//   butterflies, and each probability is exp / sum;
// - the k largest probabilities in descending order, equal ones in ascending
//   expert order, by repeated warp arg-max;
// - their sum by the xor butterfly over k lanes that PyTorch's row sum of a
//   [tokens, k] tensor uses, clamped below by FLT_EPSILON, divides them;
// - a per-expert scale multiplies the result in FP32.
//
// Launches synchronize nothing, so CUDA graphs capture them.
//
// The module is a cubin: the descriptor bakes the scores' per-lane iteration
// count (`UNISERVE_ITERATIONS`, the expert count rounded up to 32) and the
// butterfly width (`UNISERVE_TOP`, the top-k width rounded up to a power of
// two), so the kernel reads its arguments in the operator's packed order and
// nothing else. Four warps route four tokens, which is the grid the operator
// computes.
//
// |was| `uniserve_kernels/src/uniserve_kernels/csrc/topk_softmax.cu`

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

#ifndef UNISERVE_ITERATIONS
#define UNISERVE_ITERATIONS 4
#endif

#ifndef UNISERVE_TOP
#define UNISERVE_TOP 8
#endif

namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kIterations = UNISERVE_ITERATIONS;
constexpr int kTop = UNISERVE_TOP;

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

// The arguments are the operator's packed launch parameters: the scores, their
// storage kind, the row stride in elements, the token count, the expert count,
// the top-k width, the renormalize flag, the per-expert scale and its kind, and
// the two outputs.
extern "C" __global__ void uniserve_topk_softmax(const void* scores, int score_kind,
                                                 int64_t row_stride, int rows, int experts,
                                                 int top, int renormalize, const void* scale,
                                                 int scale_kind, int32_t* ids, float* weights) {
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

  // Top-k by repeated warp arg-max; every lane ends with all k picks, in
  // descending order with equal values in ascending expert order.
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
    float best = -1.f;
    int best_expert = INT_MAX;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int expert = lane + it * kWarp;
      if (expert < experts && !(taken >> it & 1u) && values[it] > best) {
        best = values[it];
        best_expert = expert;
      }
    }
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      const float other = __shfl_xor_sync(0xffffffffu, best, offset);
      const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, offset);
      if (other > best || (other == best && other_expert < best_expert)) {
        best = other;
        best_expert = other_expert;
      }
    }
    top_value[j] = best;
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
    const float total = partial[0] < FLT_EPSILON ? FLT_EPSILON : partial[0];
    weight = weight / total;
  }
  if (scale_kind != kAbsent) weight = weight * load_value(scale, scale_kind, expert);
  ids[static_cast<int64_t>(row) * top + lane] = expert;
  weights[static_cast<int64_t>(row) * top + lane] = weight;
}
