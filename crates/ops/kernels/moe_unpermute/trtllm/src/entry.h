// SPDX-License-Identifier: Apache-2.0

// The C entry of the moe_unpermute module: the operator's `moe::combine::Call`
// (crates/ops/src/moe/combine.rs, UNPERMUTE_CALL_LAYOUT), field for field, and
// the entry the operator's Rust side calls.

#pragma once

#include <cstddef>
#include <cstdint>

// One combine launch. The call struct is the operator's, byte for byte; `stream`
// is the CUstream of the device the executor drives.
//
// `routes` is [routes_rows, hidden] in permuted order, `expanded_to_permuted`
// [tokens, top] i32 mapping a route to its row of `routes` (-1 skips it), `scales`
// [tokens, top] the route weights, and `out` [tokens, hidden].
struct uniserve_unpermute_call {
  const void* routes;
  void* out;
  const int32_t* expanded_to_permuted;
  const void* scales;
  int64_t tokens;
  int64_t top;
  int64_t hidden;
  int64_t routes_rows;
  int64_t input_dtype;
  int64_t scales_dtype;
  int64_t output_dtype;
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

static_assert(offsetof(uniserve_unpermute_call, error) == 88,
              "the call's error message must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_unpermute_call) == 600,
              "the call must be the size of the operator's layout");

// Combine one route set into the tokens' rows. Returns 0, or a nonzero status
// with a NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_moe_unpermute(const uniserve_unpermute_call* call, void* stream);
