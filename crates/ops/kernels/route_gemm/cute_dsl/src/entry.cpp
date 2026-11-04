// SPDX-License-Identifier: Apache-2.0

// The route_gemm module's C entry: loads the exported module into this
// process's devices once and then launches through the generated wrapper.
//
// The generated header this file is compiled with (`route_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; `entry.h` is the stable ABI the operator's Rust
// side calls.

#include "route_gemm.h"

#include "entry.h"

#include <mutex>

namespace {

// The exported module holds the kernel's cubin and its host launcher. It is
// loaded into every device when the first launch arrives, and stays loaded for
// the process's life: captured graphs keep referencing it.
route_gemm_Kernel_Module_t& module() {
  static route_gemm_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { route_gemm_Kernel_Module_Load(&handle); });
  return handle;
}

}  // namespace

extern "C" int uniserve_route_gemm(const uniserve_route_gemm_args* args) {
  if (args == nullptr) {
    return 1;
  }
  return cute_dsl_route_gemm_wrapper(
      &module(),
      args->a,
      args->b,
      args->out,
      args->tile_idx_to_expert_idx,
      args->num_non_exiting_tiles,
      args->tile_idx_to_mn_limit,
      args->permuted_idx_to_expanded_idx,
      args->m,
      args->n,
      args->k,
      args->experts,
      args->routes,
      reinterpret_cast<cudaStream_t>(args->stream));
}
