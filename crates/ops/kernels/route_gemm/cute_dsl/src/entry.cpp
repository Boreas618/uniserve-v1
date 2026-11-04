// SPDX-License-Identifier: Apache-2.0

// The route_gemm module's C entry: checks the operator's call against the
// compiled specialization, then launches the exported module. The module is
// loaded into this process's devices once; captured graphs keep referencing it.
//
// The generated header this file is compiled with (`route_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; `entry.h` is the stable ABI the operator's Rust side
// calls.

#include "route_gemm.h"

#include "entry.h"

#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

// The exported module holds the kernel's cubin and its host launcher.
route_gemm_Kernel_Module_t& module() {
  static route_gemm_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { route_gemm_Kernel_Module_Load(&handle); });
  return handle;
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(uniserve_route_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_route_gemm(const uniserve_route_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  // The specialization is a BF16 kernel: a dtype it does not serve is refused,
  // not reinterpreted.
  if (call->dtype != UNISERVE_DTYPE_BF16) {
    return refuse(const_cast<uniserve_route_call*>(call), "route_gemm serves bf16 only");
  }
  if (call->rows == nullptr || call->weights == nullptr || call->out == nullptr) {
    return refuse(const_cast<uniserve_route_call*>(call), "route_gemm needs rows, weights and out");
  }
  // The wrapper's `n` is the output width (the model's hidden size, which the
  // call names `width`) and its `k` the contraction length (the intermediate
  // width); the routes are the expanded rows.
  return cute_dsl_route_gemm_wrapper(
      &module(),
      const_cast<void*>(call->rows),
      const_cast<void*>(call->weights),
      call->out,
      const_cast<void*>(static_cast<const void*>(call->tile_to_expert)),
      const_cast<void*>(static_cast<const void*>(call->num_non_exiting_tiles)),
      const_cast<void*>(static_cast<const void*>(call->tile_to_limit)),
      const_cast<void*>(static_cast<const void*>(call->permuted_to_expanded)),
      call->permuted_rows,
      call->width,
      call->intermediate,
      call->experts,
      call->expanded_rows,
      reinterpret_cast<cudaStream_t>(stream));
}
