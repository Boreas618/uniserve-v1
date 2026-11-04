// SPDX-License-Identifier: Apache-2.0

// The C entry of the nvfp4_gemm module: the operator's call (crates/ops/src/fp4.rs,
// `Nvfp4Gemm`'s argument order), field for field, and the entry the operator's
// Rust side calls.

#pragma once

#include <cstddef>
#include <cstdint>

// One dense product over packed E2M1 operands. The call struct is the operator's
// argument order; `stream` is the CUstream of the device the executor drives.
//
// `input` is [m, k / 2] packed bytes, `weight` [n, k / 2] packed bytes, both with
// unit stride on the contraction; `input_scales` the linear [m, k / 16] E4M3
// matrix, `weight_scales` the 128x4 swizzled [n, k / 16] one; `alpha` the [1] FP32
// product of the two tensor scales, which the epilogue reads on the device;
// `bias` the [n] addend a biasing epilogue takes, and `out` the [m, n] product.
struct uniserve_nvfp4_gemm_call {
  const void* input;
  const void* weight;
  const void* input_scales;
  const void* weight_scales;
  const void* alpha;
  const void* bias;
  void* out;
  int64_t m;
  int64_t n;
  int64_t k;
  char error[512];
};

// The operator crate's `DType::as_i64()` numbering.
enum uniserve_dtype : int64_t {
  UNISERVE_DTYPE_U8 = 0,
  UNISERVE_DTYPE_I32 = 1,
  UNISERVE_DTYPE_I64 = 2,
  UNISERVE_DTYPE_F16 = 3,
  UNISERVE_DTYPE_BF16 = 4,
  UNISERVE_DTYPE_F32 = 5,
  UNISERVE_DTYPE_F8E4M3 = 6,
  UNISERVE_DTYPE_F4E2M1 = 7,
};

static_assert(offsetof(uniserve_nvfp4_gemm_call, error) == 80,
              "the call's error message must sit where the operator's packing puts it");
static_assert(sizeof(uniserve_nvfp4_gemm_call) == 592,
              "the call must be the size of the operator's packing");

// Launch the module's product. Returns 0, or a nonzero status with a
// NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_nvfp4_gemm(const uniserve_nvfp4_gemm_call* call, void* stream);
