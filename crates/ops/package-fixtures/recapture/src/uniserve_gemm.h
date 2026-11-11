// SPDX-License-Identifier: Apache-2.0
//
// The `gemm` call structure, mirrored from `crates/ops/src/gemm.rs`.
//
// The main process builds one of these per launch and passes it to
// `uniserve_gemm`; the layout is pinned there by `GEMM_CALL_LAYOUT`, by the size
// and offset assertions beside it, and by this header's own assertions.
#ifndef UNISERVE_GEMM_H
#define UNISERVE_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
#include <cstddef>
#define UNISERVE_STATIC_ASSERT static_assert
#else
#include <assert.h>
#define UNISERVE_STATIC_ASSERT _Static_assert
#endif

typedef struct UniserveGemmCall {
  // `[m, n]` row-major, the product.
  void* out;
  // `[m, k]` row-major, the activations.
  const void* input;
  // The weight in the descriptor's order: `[k, n]` (kxn) or `[n, k]` (nxk).
  const void* weight;
  // `[n]` when the epilogue adds a bias, null otherwise.
  const void* bias;
  // Rows of the input, and of the product.
  int64_t m;
  // Columns of the product.
  int64_t n;
  // The contraction length.
  int64_t k;
  // Elements between consecutive input rows.
  int64_t input_row_stride;
  // Elements between consecutive weight rows of the stored order.
  int64_t weight_row_stride;
  // Elements between consecutive output rows.
  int64_t out_row_stride;
  // A scalar applied to the product before the epilogue; 1.0 for a plain product.
  float alpha;
  // This entry's message on a nonzero return.
  char error[512];
} UniserveGemmCall;

UNISERVE_STATIC_ASSERT(sizeof(UniserveGemmCall) == 600,
                       "the gemm call structure's size changed");
UNISERVE_STATIC_ASSERT(offsetof(UniserveGemmCall, input) == 8,
                       "the gemm call structure's `input` moved");
UNISERVE_STATIC_ASSERT(offsetof(UniserveGemmCall, bias) == 24,
                       "the gemm call structure's `bias` moved");
UNISERVE_STATIC_ASSERT(offsetof(UniserveGemmCall, k) == 48,
                       "the gemm call structure's `k` moved");
UNISERVE_STATIC_ASSERT(offsetof(UniserveGemmCall, out_row_stride) == 72,
                       "the gemm call structure's strides moved");
UNISERVE_STATIC_ASSERT(offsetof(UniserveGemmCall, alpha) == 80,
                       "the gemm call structure's `alpha` moved");
UNISERVE_STATIC_ASSERT(offsetof(UniserveGemmCall, error) == 84,
                       "the gemm call structure's `error` moved");

#endif  // UNISERVE_GEMM_H
