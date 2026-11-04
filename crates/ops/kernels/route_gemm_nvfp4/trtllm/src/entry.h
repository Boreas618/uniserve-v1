// SPDX-License-Identifier: Apache-2.0

// The C entry of the route_gemm_nvfp4 module's trtllm-gen backend: the operator's
// `RouteNvfp4Call` (crates/ops/src/moe/grouped.rs, whose offsets are asserted there)
// and the entry the operator's Rust side calls.
//
// The product is trtllm-gen's routed, block-scaled second matrix: `Gemm2::Runner` from
// the FlashInfer distribution, with this translation unit's entry in front of it. Both
// operands are encoded E2M1 -- two codes a byte along the contraction -- because FC1's
// epilogue is a BF16 row and `nvfp4_quantize` is the program that encoded it.

#pragma once

#include <cstddef>
#include <cstdint>

// One encoded FC2 launch. Every extent is in elements of the unpacked matrix; a packed
// operand's own width is half its contraction, and the entry checks that the operand it
// was handed is the encoding it declared.
//
// `hidden_size` is the output width and `intermediate` the contraction, as the
// descriptor's key and the runner's own arguments name them. `workspace` is the
// program's own scratch: the batched-GEMM tactic's requirement (which the entry asks
// the runner for, per call) plus the bias table and the padded-route counter the
// product reads. A workspace shorter than that requirement is refused by name, never
// grown: a step that allocates is a host allocation at an address the graph pool did
// not bake into the capture.
struct uniserve_route_gemm_nvfp4_call {
  const void* hidden;
  const void* hidden_scale;
  const void* hidden_token_scale;
  const void* weights;
  const void* scales;
  const void* global_scale;
  const int32_t* tile_to_expert;
  const int32_t* tile_to_limit;
  const int32_t* permuted_to_expanded;
  const int32_t* num_non_exiting_tiles;
  void* out;
  void* workspace;
  int64_t workspace_bytes;
  int64_t routes;
  int64_t hidden_size;
  int64_t intermediate;
  int64_t experts;
  int64_t top;
  int64_t tile_m;
  int64_t tile_n;
  int64_t cluster_m;
  int64_t cluster_n;
  int64_t dtype;
  char error[512];
};

static_assert(offsetof(uniserve_route_gemm_nvfp4_call, hidden_scale) == 8);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, hidden_token_scale) == 16);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, global_scale) == 40);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, num_non_exiting_tiles) == 72);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, workspace) == 88);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, workspace_bytes) == 96);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, top) == 136);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, dtype) == 176);
static_assert(offsetof(uniserve_route_gemm_nvfp4_call, error) == 184);
static_assert(sizeof(uniserve_route_gemm_nvfp4_call) == 696);

// The operator crate's `DType::as_i64()` numbering, for the types this entry names.
constexpr int64_t uniserve_dtype_bf16 = 3;
constexpr int64_t uniserve_dtype_e2m1 = 10;

// One launch. `stream` is the CUstream of the device the executor drives; the entry
// asks the device its own answers (its id, the tactic's scratch requirement) rather
// than being handed them, so a call is a description of the problem and nothing else.
extern "C" int uniserve_route_gemm_nvfp4(uniserve_route_gemm_nvfp4_call* call, void* stream);
