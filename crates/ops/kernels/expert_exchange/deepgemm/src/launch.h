// SPDX-License-Identifier: Apache-2.0
//
// What the deepgemm module's entry (`entry.cu`) hands one kernel instance (`kernel.cu`) to launch:
// the kernel's arguments as DeepGEMM's `sm100_fp8_fp4_mega_moe` passes them, built by the entry.

#pragma once

#include <cuda.h>

#include <cstdint>

#include <deep_gemm/layout/sym_buffer.cuh>

namespace uniserve::mega {

// The kernel's tensor maps, in its parameter order: the routed experts' first product (its rows,
// their scales, the weights, their scales, the gated rows it writes), the second (rows, scales,
// weights, scales), then the shared experts' in the same order. A layer without shared experts
// states the routed maps in their place, as DeepGEMM does.
enum Map : int {
  kL1Acts,
  kL1ActsSf,
  kL1Weights,
  kL1WeightsSf,
  kL1Output,
  kL2Acts,
  kL2ActsSf,
  kL2Weights,
  kL2WeightsSf,
  kSharedL1Acts,
  kSharedL1ActsSf,
  kSharedL1Weights,
  kSharedL1WeightsSf,
  kSharedL1Output,
  kSharedL2Acts,
  kSharedL2ActsSf,
  kSharedL2Weights,
  kSharedL2WeightsSf,
  kMaps,
};

struct Launch {
  // `[tokens, hidden]` BF16: this rank's rows' sums.
  void* y;
  // This rank's rows in the launch.
  uint32_t tokens;
  // Every rank's window as this rank maps it, and this rank's index among them.
  deep_gemm::layout::SymBuffer<UNISERVE_RANKS> windows;
  CUtensorMap maps[kMaps];
  // `[multiprocessors]` U8: the locality domain each multiprocessor's blocks schedule for.
  const uint8_t* domains;
};

}  // namespace uniserve::mega
