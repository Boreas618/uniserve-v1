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

static_assert(offsetof(uniserve_gather_call, error) == 112,
              "the call's error message must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_gather_call) == 624,
              "the call must be the size of the operator's layout");

// Launch the module's FC1 kernel. Returns 0, or a nonzero status with a
// NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_gather_gemm(const uniserve_gather_call* call, void* stream);
