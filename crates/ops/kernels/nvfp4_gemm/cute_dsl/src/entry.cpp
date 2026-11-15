// SPDX-License-Identifier: Apache-2.0

// The nvfp4_gemm module's C entry: checks the operator's call against the
// compiled specialization, then launches the exported module.
//
// The generated header this file is compiled with (`nvfp4_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; the kernel library's ABI header, generated from the
// operator crate, declares the call structure the operator's Rust side passes,
// and `uniserve_cute_dsl.h` is what every CuTe DSL entry shares.

#include "nvfp4_gemm.h"

#include <uniserve_nvfp4_gemm.h>

#include <uniserve_cute_dsl.h>

int32_t launch(const uniserve_nvfp4_gemm_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->input == nullptr || call->weight == nullptr || call->input_scales == nullptr ||
      call->weight_scales == nullptr || call->alpha == nullptr || call->out == nullptr) {
    return cute_dsl_refuse(call,
                           "nvfp4_gemm needs input, weight, both scale matrices, alpha and out");
  }
  // The problem's three extents are compile-time values of the module, so the
  // call must name the problem that was compiled: a call of other extents would
  // read and write outside the operands the tiles were built for.
  if (call->m != UNISERVE_NVFP4_M || call->n != UNISERVE_NVFP4_N ||
      call->k != UNISERVE_NVFP4_K) {
    return cute_dsl_refuse(call, "the module is compiled for another problem size");
  }
  return cute_dsl_nvfp4_gemm_wrapper(
      &cute_dsl_module<nvfp4_gemm_Kernel_Module_t, nvfp4_gemm_Kernel_Module_Load>(),
      const_cast<void*>(call->input),
      const_cast<void*>(call->weight),
      const_cast<void*>(call->input_scales),
      const_cast<void*>(call->weight_scales),
      const_cast<void*>(call->alpha),
      call->out,
      reinterpret_cast<cudaStream_t>(stream));
}

extern "C" int32_t uniserve_nvfp4_gemm(const uniserve_nvfp4_gemm_call* call, void* stream) {
  return cute_dsl_entry<uniserve_nvfp4_gemm_call, launch>(call, stream);
}
