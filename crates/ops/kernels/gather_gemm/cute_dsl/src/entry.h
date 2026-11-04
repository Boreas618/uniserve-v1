// SPDX-License-Identifier: Apache-2.0

// The C entry of the gather_gemm module. The operator's Rust side mirrors this
// struct field for field and calls the entry through the loaded library.

#pragma once

#include <cstdint>

// One FC1 launch. Device addresses come from the executor's arena; `stream` is
// the CUstream of the device the executor drives.
//
// `a` is `[tokens, k]` BF16 hidden states, `b` `[experts, n, k]` BF16 weights
// with each expert's `n / 2` up rows followed by its `n / 2` gate rows, `c`
// `[m, n / 2]` BF16 permuted output rows. The sort metadata holds `m /
// tile_size` tile entries (`tile_idx_to_expert_idx`, `tile_idx_to_mn_limit`),
// the `m` row-to-token map, and one tile count.
struct uniserve_gather_gemm_args {
  void* a;
  void* b;
  void* c;
  void* tile_idx_to_expert_idx;
  void* tile_idx_to_mn_limit;
  void* token_id_mapping;
  void* num_non_exiting_tiles;
  int64_t tokens;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t experts;
  void* stream;
};

// Launch the module's FC1 kernel. Returns 0, or the host launcher's error code.
extern "C" int uniserve_gather_gemm(const uniserve_gather_gemm_args* args);
