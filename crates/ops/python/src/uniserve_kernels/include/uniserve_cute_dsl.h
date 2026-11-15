// SPDX-License-Identifier: Apache-2.0

// What every CuTe DSL module's C entry shares: the exported module loaded once,
// the refusal written into the call's error buffer, and the C boundary that
// turns an exception into a refusal.
//
// A CuTe DSL package's entry unit (`src/entry.cpp`) includes the header the
// export generates (`<name>.h`, which declares the module handle, its loader and
// the launcher), its operator's call structure and then this header, which the
// Build Server's CuTe DSL driver puts on the include path from the kernel
// packages' shared sources (`uniserve_kernels`). Everything here has internal
// linkage: each module is one translation unit and exports only its C entries.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>

namespace {

// The multiprocessors of the device `stream` runs on, into `count`: the bound
// of a persistent grid that keeps one CTA on every multiprocessor. Both
// queries are host calls that enqueue nothing on the stream.
cudaError_t cute_dsl_multiprocessors(cudaStream_t stream, int32_t* count) {
  int device = 0;
  const cudaError_t found = cudaStreamGetDevice(stream, &device);
  if (found != cudaSuccess) {
    return found;
  }
  int multiprocessors = 0;
  const cudaError_t read =
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
  *count = multiprocessors;
  return read;
}

// The exported module, holding the kernel's cubin and its host launcher. The
// module is loaded into this process's devices once, by the export's `Load`;
// captured graphs keep referencing it.
template <typename Module, void (*Load)(Module*)>
Module& cute_dsl_module() {
  static Module handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { Load(&handle); });
  return handle;
}

// The message an entry writes before it returns a nonzero status: the call
// structure's `error` buffer holds it, NUL-terminated.
template <typename Call>
int32_t cute_dsl_refuse(const Call* call, const char* message) {
  std::snprintf(const_cast<Call*>(call)->error, sizeof(call->error), "%s", message);
  return 1;
}

// The operator's entry: `Launch`, and an exception turned into a message.
//
// An exception raised inside the kernel's own code would otherwise cross the C
// boundary, where nothing may unwind: the process ends and the only message a
// log holds is the runtime's own (`internal error: entered unreachable code`),
// which names neither the operator nor the reason. A thrown exception states
// one, so it goes into the call's error buffer and the entry answers a refusal
// like every other one it makes.
template <typename Call, int32_t (*Launch)(const Call*, void*)>
int32_t cute_dsl_entry(const Call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  try {
    return Launch(call, stream);
  } catch (std::exception const& thrown) {
    return cute_dsl_refuse(call, thrown.what());
  } catch (...) {
    return cute_dsl_refuse(call, "the kernel's launch failed without stating why");
  }
}

}  // namespace
