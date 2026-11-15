// SPDX-License-Identifier: Apache-2.0

// The route_gemm module's C entry: checks the operator's call against the
// compiled specialization, then launches the exported module.
//
// The generated header this file is compiled with (`route_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; the kernel library's ABI header, generated from the
// operator crate, declares the call structure the operator's Rust side passes,
// and `uniserve_cute_dsl.h` is what every CuTe DSL entry shares.

#include "route_gemm.h"

#include <uniserve_route_gemm.h>

#include <uniserve_cute_dsl.h>

int32_t launch(const uniserve_route_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  // The specialization is a BF16 kernel: a dtype it does not serve is refused,
  // not reinterpreted.
  if (call->dtype != UNISERVE_DTYPE_BF16) {
    return cute_dsl_refuse(call, "route_gemm serves bf16 only");
  }
  if (call->rows == nullptr || call->weights == nullptr || call->out == nullptr) {
    return cute_dsl_refuse(call, "route_gemm needs rows, weights and out");
  }
  // The persistent grid keeps at most one CTA on each multiprocessor of the
  // device the stream runs on. The operator states that device's count in the
  // call, so the entry asks the runtime nothing while the stream may capture;
  // the launcher takes it as a 32-bit word.
  if (call->multiprocessors <= 0 || call->multiprocessors > INT32_MAX) {
    return cute_dsl_refuse(call, "route_gemm needs its device's multiprocessor count");
  }
  // The wrapper's `n` is the output width (the model's hidden size, which the
  // call names `width`) and its `k` the contraction length (the intermediate
  // width); the rows are permuted in and out, so the wrapper's `m` and its output
  // row count are both the permuted height.
  return cute_dsl_route_gemm_wrapper(
      &cute_dsl_module<route_gemm_Kernel_Module_t, route_gemm_Kernel_Module_Load>(),
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
      call->permuted_rows,
      static_cast<int32_t>(call->multiprocessors),
      reinterpret_cast<cudaStream_t>(stream));
}

extern "C" int32_t uniserve_route_gemm(const uniserve_route_call* call, void* stream) {
  return cute_dsl_entry<uniserve_route_call, launch>(call, stream);
}
