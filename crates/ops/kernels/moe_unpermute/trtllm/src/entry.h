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
// `routes` is [routes_rows, hidden] in the row convention `layout` names - permuted (found through
// `expanded_to_permuted`, [tokens, top] i32, -1 skipping a route) or expanded (`token * top + route`)
// - `scales` is [tokens, top] the route weights, and `out` [tokens, hidden].
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
  // // The input buffer's row convention (`ops::moe::combine::Layout`): 0 permuted, 1 expanded.
  int64_t layout;
  char error[512];
};

static_assert(offsetof(uniserve_unpermute_call, routes) == 0,
              "the call's routes must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, out) == 8,
              "the call's out must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, expanded_to_permuted) == 16,
              "the call's expanded_to_permuted must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, scales) == 24,
              "the call's scales must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, tokens) == 32,
              "the call's tokens must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, top) == 40,
              "the call's top must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, hidden) == 48,
              "the call's hidden must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, routes_rows) == 56,
              "the call's routes_rows must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, input_dtype) == 64,
              "the call's input_dtype must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, scales_dtype) == 72,
              "the call's scales_dtype must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, output_dtype) == 80,
              "the call's output_dtype must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, layout) == 88,
              "the call's layout must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_unpermute_call, error) == 96,
              "the call's error must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_unpermute_call) == 608,
              "the call must be the size of the operator's layout");

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


// Combine one route set into the tokens' rows. Returns 0, or a nonzero status
// with a NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_moe_unpermute(const uniserve_unpermute_call* call, void* stream);
