// SPDX-License-Identifier: Apache-2.0
//
// One instance of DeepGEMM's MegaMoE over MXFP4 experts (`sm100_fp8_fp4_mega_moe_impl`, the
// `deepgemm` pin) at one tile height, and its launch. The build compiles this unit once for each
// BLOCK_M DeepGEMM's rule picks for the descriptor's steps (`src/build.py`), each under its own
// entry name (`UNISERVE_INSTANCE`), and the module's entry (`entry.cu`) launches the height of a
// step.
//
// The template arguments are DeepGEMM's for the descriptor and the tile (`config.cuh`): the
// SwiGLU with the gated activation clamped before the sigmoid (`swiglu`, not `swiglustep`), no
// scaling of its inputs, and DeepGEMM's fast exponential and reciprocal, as vLLM launches it.

#include <cuda_runtime.h>

#include <cstdint>

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

#include "config.cuh"
#include "launch.h"

#if !defined(UNISERVE_BLOCK_M) || !defined(UNISERVE_INSTANCE)
#error "the build defines the instance's tile height and entry name (src/build.py)"
#endif

namespace {

using namespace uniserve::mega;

constexpr Tile kTile = tile(UNISERVE_BLOCK_M);

static_assert(kTile.stages >= 2, "the pipeline holds at least two stages");
static_assert(kTile.shared_bytes <= kSharedMemory, "the tile fits a block's shared memory");
static_assert(kRingTokens % UNISERVE_BLOCK_M == 0, "the ring is whole blocks of the tile");

constexpr auto kKernel = &deep_gemm::sm100_fp8_fp4_mega_moe_impl<
    kMaxTokens, kHidden, kIntermediate, kExperts, kShared, kTop,
    UNISERVE_BLOCK_M, kBlockN, kBlockK,
    kTile.store_block_m_l1, kTile.store_block_m_l2,
    kTile.sf_block_m, kTile.sf_block_n,
    kRingTokens, kSfRingTokens,
    kTile.stages, kBytesPerPull,
    kDispatchThreads, kNonEpilogueThreads, kEpilogueThreads,
    kMultiprocessors, kRanks,
    kClamp, /*kClampAfterSilu=*/false,
    /*kActivationAlpha=*/1.0f, /*kActivationBeta=*/0.0f,
    /*kFastMath=*/true,
    cutlass::detail::float_e2m1_unpacksmem_t>;

}  // namespace

// Launches the instance with `launch`'s arguments on `stream`: one block per multiprocessor of the
// device the module was compiled for, in clusters of two, after the staging kernel the entry
// launched before it. Answers the runtime's error code, zero on success.
extern "C" int32_t UNISERVE_INSTANCE(const Launch* launch, void* stream) {
  // The attribute is the device's: a process whose ranks are on several devices sets it on each,
  // and every launch states it again rather than track which devices have it.
  cudaError_t status = cudaFuncSetAttribute(
      kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kTile.shared_bytes);
  if (status != cudaSuccess) {
    return static_cast<int32_t>(status);
  }

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(kMultiprocessors, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kTile.shared_bytes;
  config.stream = static_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attributes[2];
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = 2;
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;
  // A programmatic dependent of the staging kernel before it on the stream: the kernel sets up
  // its barriers and tensor memory meanwhile, and waits for the staging's completion
  // (`cudaGridDependencySynchronize`) before it reads the window.
  attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attributes;
  config.numAttrs = 2;

  // No count of the rows each local expert received is kept: the router's top-k counts the load.
  const CUtensorMap* maps = launch->maps;
  status = cudaLaunchKernelEx(
      &config, kKernel, launch->y, static_cast<int*>(nullptr), launch->tokens, launch->windows,
      maps[kL1Acts], maps[kL1ActsSf], maps[kL1Weights], maps[kL1WeightsSf], maps[kL1Output],
      maps[kL2Acts], maps[kL2ActsSf], maps[kL2Weights], maps[kL2WeightsSf], maps[kSharedL1Acts],
      maps[kSharedL1ActsSf], maps[kSharedL1Weights], maps[kSharedL1WeightsSf],
      maps[kSharedL1Output], maps[kSharedL2Acts], maps[kSharedL2ActsSf], maps[kSharedL2Weights],
      maps[kSharedL2WeightsSf], launch->domains);
  return static_cast<int32_t>(status);
}
