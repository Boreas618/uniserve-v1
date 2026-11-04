// SPDX-License-Identifier: Apache-2.0

// The C entry of the routing_softmax_topk module. The operator's Rust side
// mirrors this struct field for field and calls the entry through the loaded
// library; nothing else crosses the boundary.

#pragma once

#include <cstdint>

// Storage kinds of a scores or scale buffer.
enum uniserve_routing_softmax_topk_kind {
  UNISERVE_ROUTING_SOFTMAX_TOPK_FLOAT = 0,
  UNISERVE_ROUTING_SOFTMAX_TOPK_BFLOAT16 = 1,
  UNISERVE_ROUTING_SOFTMAX_TOPK_HALF = 2,
};

// One routing launch. Buffers are device addresses; `stream` is a CUstream
// the caller made current. `scores` holds `[rows, experts]` rows whose expert
// stride is 1 and whose row stride is `row_stride` elements; `scale` holds
// `[experts]` values of `scale_kind` or is null; `ids` is int32 and `weights`
// float32, both `[rows, top]`. `renormalize` divides each row's picks by their
// sum clamped below by FLT_EPSILON.
struct uniserve_routing_softmax_topk_args {
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
  void* stream;
};

// Launch the specialized routing kernel. Returns 0, or a CUDA error code.
extern "C" int uniserve_routing_softmax_topk(const uniserve_routing_softmax_topk_args* args);
