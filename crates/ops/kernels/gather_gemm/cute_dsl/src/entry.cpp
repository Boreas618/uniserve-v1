// SPDX-License-Identifier: Apache-2.0

// The gather_gemm module's C entry: checks the operator's call against the
// compiled specialization, then launches the exported module. The module is
// loaded into this process's devices once; captured graphs keep referencing it.
//
// The generated header this file is compiled with (`gather_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; `entry.h` is the stable ABI the operator's Rust side
// calls.

#include "gather_gemm.h"

#include "entry.h"

#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

// The exported module holds the kernel's cubin and its host launcher.
gather_gemm_Kernel_Module_t& module() {
  static gather_gemm_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { gather_gemm_Kernel_Module_Load(&handle); });
  return handle;
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(uniserve_gather_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_gather_gemm(const uniserve_gather_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  // The specialization is a BF16 kernel: the trade the operator records for a
  // dtype it does not serve is a refusal here, not a reinterpreted buffer.
  if (call->dtype != UNISERVE_DTYPE_BF16) {
    return refuse(const_cast<uniserve_gather_call*>(call), "gather_gemm serves bf16 only");
  }
  if (call->hidden == nullptr || call->weights == nullptr || call->out == nullptr) {
    return refuse(const_cast<uniserve_gather_call*>(call), "gather_gemm needs hidden, weights and out");
  }
  // Each expert holds its `intermediate` up rows and then its `intermediate`
  // gate rows, so the weight's N extent is twice the output's.
  if (call->width % 2 != 0) {
    return refuse(const_cast<uniserve_gather_call*>(call), "gather_gemm needs an even 2*intermediate width");
  }
  return cute_dsl_gather_gemm_wrapper(
      &module(),
      const_cast<void*>(call->hidden),
      const_cast<void*>(call->weights),
      call->out,
      const_cast<void*>(static_cast<const void*>(call->tile_to_expert)),
      const_cast<void*>(static_cast<const void*>(call->tile_to_limit)),
      const_cast<void*>(static_cast<const void*>(call->permuted_to_expanded)),
      const_cast<void*>(static_cast<const void*>(call->num_non_exiting_tiles)),
      call->tokens,
      call->permuted_rows,
      call->width,
      call->hidden_size,
      call->experts,
      reinterpret_cast<cudaStream_t>(stream));
}
