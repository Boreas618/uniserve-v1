// SPDX-License-Identifier: Apache-2.0

// The C entry of the route_gemm module. The operator's Rust side mirrors this
// struct field for field and calls the entry through the loaded library.

#pragma once

#include <cstdint>

// One FC2 launch. Device addresses come from the executor's arena; `stream` is
// the CUstream of the device the executor drives.
//
// `a` is `[m, k]` BF16 permuted rows, `b` `[experts, n, k]` BF16 down
// projections and `out` `[routes, n]` BF16, unweighted. The sort metadata holds
// `m / tile_size` tile entries (`tile_idx_to_expert_idx`, `tile_idx_to_mn_limit`),
// one tile count, and the `m` permuted-to-expanded index map.
struct uniserve_route_gemm_args {
  void* a;
  void* b;
  void* out;
  void* tile_idx_to_expert_idx;
  void* num_non_exiting_tiles;
  void* tile_idx_to_mn_limit;
  void* permuted_idx_to_expanded_idx;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t experts;
  int64_t routes;
  void* stream;
};

// Launch the module's FC2 kernel. Returns 0, or the host launcher's error code.
extern "C" int uniserve_route_gemm(const uniserve_route_gemm_args* args);
