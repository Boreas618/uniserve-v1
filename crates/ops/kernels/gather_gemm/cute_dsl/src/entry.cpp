// SPDX-License-Identifier: Apache-2.0

// The gather_gemm module's C entry: checks the operator's call against the
// compiled specialization, then launches the exported module.
//
// The generated header this file is compiled with (`gather_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; `entry.h` is the stable ABI the operator's Rust side
// calls, and `uniserve_cute_dsl.h` what every CuTe DSL entry shares.

#include "gather_gemm.h"

#include "entry.h"

#include <uniserve_cute_dsl.h>

int32_t launch(const uniserve_gather_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  // The specialization is a BF16 kernel: the trade the operator records for a
  // dtype it does not serve is a refusal here, not a reinterpreted buffer.
  if (call->dtype != UNISERVE_DTYPE_BF16) {
    return cute_dsl_refuse(call, "gather_gemm serves bf16 only");
  }
  if (call->hidden == nullptr || call->weights == nullptr || call->out == nullptr) {
    return cute_dsl_refuse(call, "gather_gemm needs hidden, weights and out");
  }
  // Each expert holds its `intermediate` up rows and then its `intermediate`
  // gate rows, so the weight's N extent is twice the output's.
  if (call->width % 2 != 0) {
    return cute_dsl_refuse(call, "gather_gemm needs an even 2*intermediate width");
  }
  // The persistent grid keeps one CTA on every multiprocessor of the device
  // the stream runs on, so its bound is that device's count.
  int32_t multiprocessors = 0;
  if (cute_dsl_multiprocessors(reinterpret_cast<cudaStream_t>(stream), &multiprocessors) !=
      cudaSuccess) {
    return cute_dsl_refuse(call, "gather_gemm cannot read its device's multiprocessor count");
  }
  return cute_dsl_gather_gemm_wrapper(
      &cute_dsl_module<gather_gemm_Kernel_Module_t, gather_gemm_Kernel_Module_Load>(),
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
      multiprocessors,
      reinterpret_cast<cudaStream_t>(stream));
}

extern "C" int32_t uniserve_gather_gemm(const uniserve_gather_call* call, void* stream) {
  return cute_dsl_entry<uniserve_gather_call, launch>(call, stream);
}
