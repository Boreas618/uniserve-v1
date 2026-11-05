// SPDX-License-Identifier: Apache-2.0

// The nvfp4_gemm module's C entry: checks the operator's call against the
// compiled specialization, then launches the exported module. The module is
// loaded into this process's devices once; captured graphs keep referencing it.
//
// The generated header this file is compiled with (`nvfp4_gemm.h`, written by
// `export_to_c` during the package's build) declares the module handle and the
// wrapper the export emits; `entry.h` is the stable ABI the operator's Rust side
// calls.

#include "nvfp4_gemm.h"

#include "entry.h"

#include <cstdio>
#include <mutex>
#include <exception>

namespace {

// The exported module holds the kernel's cubin and its host launcher.
nvfp4_gemm_Kernel_Module_t& module() {
  static nvfp4_gemm_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { nvfp4_gemm_Kernel_Module_Load(&handle); });
  return handle;
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(uniserve_nvfp4_gemm_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

}  // namespace

int32_t launch(const uniserve_nvfp4_gemm_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->input == nullptr || call->weight == nullptr || call->input_scales == nullptr ||
      call->weight_scales == nullptr || call->alpha == nullptr || call->out == nullptr) {
    return refuse(const_cast<uniserve_nvfp4_gemm_call*>(call),
                  "nvfp4_gemm needs input, weight, both scale matrices, alpha and out");
  }
  // The problem's three extents are compile-time values of the module, so the
  // call must name the problem that was compiled: a call of other extents would
  // read and write outside the operands the tiles were built for.
  if (call->m != UNISERVE_NVFP4_M || call->n != UNISERVE_NVFP4_N ||
      call->k != UNISERVE_NVFP4_K) {
    return refuse(const_cast<uniserve_nvfp4_gemm_call*>(call),
                  "the module is compiled for another problem size");
  }
  return cute_dsl_nvfp4_gemm_wrapper(
      &module(),
      const_cast<void*>(call->input),
      const_cast<void*>(call->weight),
      const_cast<void*>(call->input_scales),
      const_cast<void*>(call->weight_scales),
      const_cast<void*>(call->alpha),
      call->out,
      reinterpret_cast<cudaStream_t>(stream));
}

/// The operator's entry: the launch, and an exception turned into a message.
///
/// An exception raised inside the kernel's own code would otherwise cross this C boundary,
/// where nothing may unwind: the process ends and the only message a log holds is the runtime's
/// own (`internal error: entered unreachable code`), which names neither the operator nor the
/// reason. A thrown exception states one, so it goes into the call's error buffer and the entry
/// answers a refusal like every other one it makes.
extern "C" int32_t uniserve_nvfp4_gemm(const uniserve_nvfp4_gemm_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  try {
    return launch(const_cast<uniserve_nvfp4_gemm_call*>(call), stream);
  } catch (std::exception const& thrown) {
    return refuse(const_cast<uniserve_nvfp4_gemm_call*>(call), thrown.what());
  } catch (...) {
    return refuse(const_cast<uniserve_nvfp4_gemm_call*>(call), "the kernel's launch failed without stating why");
  }
}
