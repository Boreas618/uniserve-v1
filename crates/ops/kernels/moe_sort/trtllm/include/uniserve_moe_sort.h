// SPDX-License-Identifier: Apache-2.0
//
// Derived from TensorRT-LLM's routing pipeline,
// `cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingKernel.cuh`
// (Apache-2.0, NVIDIA CORPORATION). The tile and permutation semantics, the
// buffer layout and the ordering of the permutation follow it; the kernels
// here are written against this package's call structure.

// The C entry of the moe_sort module: the operator's `moe::sort::Call`
// (crates/ops/src/moe/sort.rs, SORT_CALL_LAYOUT), field for field, and the
// entry the operator's Rust side calls.

#pragma once

#include <cstddef>
#include <cstdint>

// One sort launch. The call structure is the operator's, byte for byte;
// `stream` is the CUstream of the device the executor drives.
//
// `ids` is [tokens, top] i32 route ids (the router's experts, in the whole
// vocabulary), `scales` is [tokens, top] route weights in `scales_dtype` —
// accepted and not read by the sort, which writes indices alone. The outputs
// are the tile metadata the grouped GEMMs read and the three permutation maps:
// `tile_to_expert`/`tile_to_limit` [max_tiles] — the expert of a tile and the
// permuted row its live rows end at, counted from the routing's first permuted
// row (TensorRT-LLM's `mPtrCtaIdxXyToMnLimit`), `expanded_to_permuted`
// [tokens, top] (-1 for a route this rank does not hold),
// `permuted_to_expanded` [max_permuted_rows], `permuted_to_token`
// [max_permuted_rows] (the token row a permuted row gathers, -1 on a tile's
// padding, TensorRT-LLM's `mPtrPermutedIdxToTokenIdx`), and the two scalars
// `total_padded_tokens` (padded permuted rows) and `num_non_exiting_tiles`
// (tiles that hold live rows).
struct uniserve_moe_sort_call {
  const int32_t* ids;
  const void* scales;
  int32_t* tile_to_expert;
  int32_t* tile_to_limit;
  int32_t* expanded_to_permuted;
  int32_t* permuted_to_expanded;
  int32_t* total_padded_tokens;
  int32_t* num_non_exiting_tiles;
  int64_t tokens;
  int64_t top;
  int64_t experts;
  int64_t local_experts;
  int64_t local_offset;
  int64_t tile_tokens;
  int64_t scales_dtype;
  char error[512];
  int32_t const* live_tokens;
  int32_t* permuted_to_token;
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

// Every field is asserted, not one per group: two fields of the same size can be swapped in a
// header alone, and a group-level assert is the check that cannot see it. The Rust side reads
// these literals (crates/ops/src/moe/sort.rs, `the_package_header_declares_this_call_layout`),
// so a layout that moves on one side fails on the other rather than at run time.
static_assert(offsetof(uniserve_moe_sort_call, ids) == 0,
              "the call's route ids must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, scales) == 8,
              "the call's route weights must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, tile_to_expert) == 16,
              "the call's tile metadata must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, tile_to_limit) == 24,
              "the call's tile metadata must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, expanded_to_permuted) == 32,
              "the call's the permutation map must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, permuted_to_expanded) == 40,
              "the call's the permutation map must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, total_padded_tokens) == 48,
              "the call's the two scalars must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, num_non_exiting_tiles) == 56,
              "the call's the two scalars must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, live_tokens) == 632,
              "the call's live count must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, tokens) == 64,
              "the call's the extents must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, top) == 72,
              "the call's the extents must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, experts) == 80,
              "the call's the extents must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, local_experts) == 88,
              "the call's the extents must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, local_offset) == 96,
              "the call's the extents must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, tile_tokens) == 104,
              "the call's the extents must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, scales_dtype) == 112,
              "the call's the scale type must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, error) == 120,
              "the call's the error message must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_moe_sort_call, permuted_to_token) == 640,
              "the call's the permutation map must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_moe_sort_call) == 648,
              "the call must be the size of the operator's layout");

// Group the routes by local expert: fills the tile metadata and the
// permutation maps. Returns 0, or a nonzero status with a NUL-terminated
// message in `call->error`.
extern "C" int32_t uniserve_moe_sort(const uniserve_moe_sort_call* call, void* stream);
