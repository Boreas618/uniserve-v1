// SPDX-License-Identifier: Apache-2.0

// The C entry of the gather_gemm module: the operator's `moe::grouped::GatherCall`
// (crates/ops/src/moe/grouped.rs, GATHER_CALL_LAYOUT), field for field, and the
// entry the operator's Rust side calls.

#pragma once

#include <cstddef>
#include <cstdint>

// One FC1 launch. The call struct is the operator's, byte for byte; `stream` is
// the CUstream of the device the executor drives.
//
// `hidden` is [tokens, hidden_size], `weights` [experts, 2*intermediate,
// hidden_size] with each expert's intermediate up rows then its intermediate
// gate rows, and `out` [permuted_rows, intermediate]. The sort metadata holds
// permuted_rows / tile entries in `tile_to_expert` and `tile_to_limit`, the
// permuted_rows row-to-expanded map in `permuted_to_expanded`, and one tile
// count in `num_non_exiting_tiles`.
struct uniserve_gather_call {
  const void* hidden;
  const void* weights;
  void* out;
  const int32_t* tile_to_expert;
  const int32_t* tile_to_limit;
  const int32_t* permuted_to_expanded;
  const int32_t* num_non_exiting_tiles;
  int64_t tokens;
  int64_t permuted_rows;
  int64_t width;
  int64_t hidden_size;
  int64_t experts;
  int64_t top;
  int64_t dtype;
  char error[512];
};

static_assert(offsetof(uniserve_gather_call, hidden) == 0,
              "the call's hidden must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, weights) == 8,
              "the call's weights must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, out) == 16,
              "the call's out must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, tile_to_expert) == 24,
              "the call's tile_to_expert must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, tile_to_limit) == 32,
              "the call's tile_to_limit must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, permuted_to_expanded) == 40,
              "the call's permuted_to_expanded must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, num_non_exiting_tiles) == 48,
              "the call's num_non_exiting_tiles must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, tokens) == 56,
              "the call's tokens must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, permuted_rows) == 64,
              "the call's permuted_rows must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, width) == 72,
              "the call's width must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, hidden_size) == 80,
              "the call's hidden_size must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, experts) == 88,
              "the call's experts must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, top) == 96,
              "the call's top must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, dtype) == 104,
              "the call's dtype must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_gather_call, error) == 112,
              "the call's error must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_gather_call) == 624,
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


// Launch the module's FC1 kernel. Returns 0, or a nonzero status with a
// NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_gather_gemm(const uniserve_gather_call* call, void* stream);
