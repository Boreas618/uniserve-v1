// SPDX-License-Identifier: Apache-2.0

// Expert routing: softmax over each token's router scores, the top-k experts,
// optional renormalization and an optional per-expert scale.
//
// The operator `routing_softmax_topk` documents the contract, and the numbers
// are the ones UniServe's tensor composition defined. One warp owns one token
// and reproduces that composition's FP32 arithmetic order:
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
// The module is a native library: this host-side entry selects the score
// layout and top-k specialization, launches on the caller's stream and reports
// launch errors, so the operator's Rust side only passes tensors and scalars.
// The kernel itself is the one the torch extension compiled, unchanged. This
// file imports no torch type.
//
// |was| `uniserve_kernels/src/uniserve_kernels/csrc/topk_softmax.cu`

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

#include "topk_softmax.h"

namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;

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

// kIterations = next_power_of_two(experts) / 32 scores per lane (at least
// one); kTop = next_power_of_two(top) picks, the width of the renormalizing
// butterfly (picks past `top` enter it as zeros).
template <int kIterations, int kTop>
__global__ void topk_softmax_kernel(const void* scores, int score_kind, int64_t row_stride,
                                    int rows, int experts, int top, int renormalize,
                                    const void* scale, int scale_kind, int32_t* ids,
                                    float* weights) {
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

struct Launch {
  const void* scores;
  int score_kind;
  int64_t row_stride;
  int rows;
  int experts;
  int top;
  int renormalize;
  const void* scale;
  int scale_kind;
  int32_t* ids;
  float* weights;
};

template <int kIterations, int kTop>
void launch(const Launch& a, cudaStream_t stream) {
  const dim3 grid((a.rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  topk_softmax_kernel<kIterations, kTop><<<grid, kWarpsPerBlock * kWarp, 0, stream>>>(
      a.scores, a.score_kind, a.row_stride, a.rows, a.experts, a.top, a.renormalize, a.scale,
      a.scale_kind, a.ids, a.weights);
}

template <int kIterations>
void dispatch_top(const Launch& a, int top, cudaStream_t stream) {
  if (top <= 1) return launch<kIterations, 1>(a, stream);
  if (top <= 2) return launch<kIterations, 2>(a, stream);
  if (top <= 4) return launch<kIterations, 4>(a, stream);
  return launch<kIterations, 8>(a, stream);
}

}  // namespace

extern "C" int uniserve_routing_softmax_topk(const uniserve_routing_softmax_topk_args* args) {
  // The operator's Rust side validated its operands: score rows are
  // [tokens, experts] with at most 256 experts, top is 1..8, and both outputs
  // are [tokens, top]. These checks guard the specialization only.
  if (args == nullptr || args->scores == nullptr || args->ids == nullptr ||
      args->weights == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (args->rows <= 0 || args->experts <= 0 || args->experts > 256 || args->top < 1 ||
      args->top > 8) {
    return cudaErrorInvalidValue;
  }

  Launch launch_args{};
  launch_args.experts = args->experts;
  launch_args.rows = args->rows;
  launch_args.top = args->top;
  launch_args.scores = args->scores;
  launch_args.score_kind = args->score_kind;
  launch_args.row_stride = args->row_stride;
  launch_args.renormalize = args->renormalize ? 1 : 0;
  launch_args.scale = args->scale;
  launch_args.scale_kind = args->scale != nullptr ? args->scale_kind : kAbsent;
  launch_args.ids = args->ids;
  launch_args.weights = args->weights;

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(args->stream);
  const int per_lane =
      launch_args.experts <= 32 ? 1 : launch_args.experts <= 64 ? 2 : launch_args.experts <= 128 ? 4 : 8;
  switch (per_lane) {
    case 1:
      dispatch_top<1>(launch_args, launch_args.top, stream);
      break;
    case 2:
      dispatch_top<2>(launch_args, launch_args.top, stream);
      break;
    case 4:
      dispatch_top<4>(launch_args, launch_args.top, stream);
      break;
    default:
      dispatch_top<8>(launch_args, launch_args.top, stream);
      break;
  }
  return static_cast<int>(cudaGetLastError());
}
