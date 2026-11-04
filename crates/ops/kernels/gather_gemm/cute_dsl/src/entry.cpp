// SPDX-License-Identifier: Apache-2.0

// The gather_gemm module's C entry: loads the exported module into this
// process's devices once and then launches through the generated wrapper.
//
// The generated header this file is compiled with (`gather_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; `entry.h` is the stable ABI the operator's Rust
// side calls.

// The build compiles this unit beside the header `export_to_c` wrote, whose
// name is the build script's module name (`gather_gemm`): it declares the
// module handle and the exported launcher this entry calls.
#include "gather_gemm.h"

#include "entry.h"

#include <mutex>

namespace {

// The exported module holds the kernel's cubin and its host launcher. It is
// loaded into every device when the first launch arrives, and stays loaded for
// the process's life: captured graphs keep referencing it.
gather_gemm_Kernel_Module_t& module() {
  static gather_gemm_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { gather_gemm_Kernel_Module_Load(&handle); });
  return handle;
}

}  // namespace

extern "C" int uniserve_gather_gemm(const uniserve_gather_gemm_args* args) {
  if (args == nullptr) {
    return 1;
  }
  return cute_dsl_gather_gemm_wrapper(
      &module(),
      args->a,
      args->b,
      args->c,
      args->tile_idx_to_expert_idx,
      args->tile_idx_to_mn_limit,
      args->token_id_mapping,
      args->num_non_exiting_tiles,
      args->tokens,
      args->m,
      args->n,
      args->k,
      args->experts,
      reinterpret_cast<cudaStream_t>(args->stream));
}
