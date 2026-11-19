// SPDX-License-Identifier: Apache-2.0

// The mxfp8_gemm module's DeepGEMM backend: the operator's call launched as DeepGEMM's SM100
// block-scaled product over MXFP8 operands.
//
// The kernel is DeepGEMM's own (`sm100_fp8_fp4_gemm_1d1d_impl`, the `deepgemm` pin), with the
// package's patch that strides its persistent blocks by the launched grid
// (`patches/launched-grid.patch`). It owns the product: TMA loads of both operands and both
// operands' packed UE8M0 scales, the tcgen05 block-scaled MMA with the scales applied inside it,
// FP32 accumulation in tensor memory and a TMA store of the BF16 tile. This unit owns what
// DeepGEMM's Python extension did around it for a normal K-major product (its
// `sm100_fp8_fp4_gemm_1d1d`, `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`, and the tensor
// map helpers of `runtime_utils.hpp`):
//
// * the template arguments, which the build chose as DeepGEMM's run-time heuristic chooses them
//   (`src/build.py`) and states as definitions;
// * the five tensor maps: A `[m, k]` and B `[n, k]` E4M3 under the 128-byte swizzle, both scale
//   tables as MN-major `u32` words (`[k / 128, padded rows]`, padded to whole 16-byte rows), and
//   the `[m, n]` BF16 output under the epilogue's swizzle;
// * the launch: one block per multiprocessor of the device, in clusters of the tile's multicast,
//   with the tile's dynamic shared memory.
//
// Every launch reads its rows from the call; `n` and `k` are compiled in, and a call of another
// width belongs to another module.

#include <uniserve_mxfp8_gemm.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

#if !defined(UNISERVE_N) || !defined(UNISERVE_K) || !defined(UNISERVE_BLOCK_M) || \
    !defined(UNISERVE_STAGES) || !defined(UNISERVE_SHARED_BYTES)
#error "the build defines the descriptor's extents and the tile (src/build.py)"
#endif

namespace {

using deep_gemm::GemmType;

// The descriptor's extents.
constexpr uint32_t kN = UNISERVE_N;
constexpr uint32_t kK = UNISERVE_K;

// The tile, as the build chose it.
constexpr uint32_t kBlockM = UNISERVE_BLOCK_M;
constexpr uint32_t kBlockN = UNISERVE_BLOCK_N;
constexpr uint32_t kBlockK = UNISERVE_BLOCK_K;
constexpr bool kSwapAB = UNISERVE_SWAP_AB;
constexpr uint32_t kCluster = UNISERVE_CLUSTER;
constexpr bool kMulticastOnA = UNISERVE_MULTICAST_ON_A;
constexpr uint32_t kLoadBlockM = UNISERVE_LOAD_BLOCK_M;
constexpr uint32_t kLoadBlockN = UNISERVE_LOAD_BLOCK_N;
constexpr uint32_t kStoreBlockM = UNISERVE_STORE_BLOCK_M;
constexpr uint32_t kStoreBlockN = UNISERVE_STORE_BLOCK_N;
constexpr uint32_t kSwizzleCD = UNISERVE_SWIZZLE_CD;
constexpr uint32_t kStages = UNISERVE_STAGES;
constexpr uint32_t kStoreStages = UNISERVE_STORE_STAGES;
constexpr uint32_t kSharedBytes = UNISERVE_SHARED_BYTES;
constexpr uint32_t kNonEpilogueThreads = UNISERVE_NON_EPILOGUE_THREADS;
constexpr uint32_t kEpilogueThreads = UNISERVE_EPILOGUE_THREADS;

// The multiprocessor count the kernel's L2 grouping is sized by; the grid is the device's.
constexpr uint32_t kPlannedSMs = UNISERVE_PLANNED_MULTIPROCESSORS;

// Elements one scale covers, and the operands' 128-byte swizzle (one 128-element row of E4M3).
constexpr uint32_t kGranK = 32;
constexpr uint32_t kSwizzleAB = 128;

// Tensor-core operand scales are loaded in whole blocks of 128 rows, one packed word a row a
// 128-element step of the contraction.
constexpr uint32_t kScaleBlockM = (kBlockM + 127) / 128 * 128;
constexpr uint32_t kScaleBlockN = (kBlockN + 127) / 128 * 128;

// Packed scale words a 16-byte row of a tensor map holds: the padding of a scale column.
constexpr int64_t kWordRows = 4;

static_assert(kK % 128 == 0, "the contraction is whole 128-element steps");
static_assert(kSharedBytes <= 232448, "the tile fits a block's shared memory");

// The kernel, with every template argument the descriptor and the tile fix: K-major operands of
// 32-element scales, rows decided at launch (`SHAPE_M = 0`), one group, a normal product with no
// accumulation and the identity epilogue.
constexpr auto kKernel = &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
    cute::UMMA::Major::K, cute::UMMA::Major::K,
    /*kGranKA=*/kGranK, /*kGranKB=*/kGranK, /*kKAlignment=*/kGranK,
    /*SHAPE_M=*/0, /*SHAPE_N=*/kN, /*SHAPE_K=*/kK,
    kBlockM, kBlockN, kBlockK,
    /*kNumGroups=*/1,
    kSwizzleAB, kSwizzleAB, kSwizzleCD,
    kStages, kStoreStages,
    kNonEpilogueThreads, kEpilogueThreads,
    kCluster, kMulticastOnA,
    kPlannedSMs,
    kSwapAB, /*kEnsureZeroPadding=*/true,
    GemmType::Normal, /*kWithAccumulation=*/false,
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
    deep_gemm::epilogue::operators::Identity>;

int refuse(uniserve_mxfp8_gemm_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

bool aligned(const void* address) {
  return address != nullptr && reinterpret_cast<uintptr_t>(address) % 16 == 0;
}

// A two-dimensional tensor map: `inner` by `outer` elements of `bytes` each, rows `stride`
// elements apart, loaded in `box_inner` by `box_outer` boxes under `swizzle`. Out-of-bounds
// elements of a box load as zeros.
bool encode(CUtensorMap* map, CUtensorMapDataType type, uint32_t bytes, const void* base,
            uint64_t inner, uint64_t outer, uint64_t stride, uint32_t box_inner,
            uint32_t box_outer, CUtensorMapSwizzle swizzle) {
  const cuuint64_t extents[2] = {inner, outer};
  const cuuint64_t strides[1] = {stride * bytes};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t steps[2] = {1, 1};
  return cuTensorMapEncodeTiled(map, type, 2, const_cast<void*>(base), extents, strides, box,
                                steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// The swizzle a tensor map states for a shared-memory row of `bytes` bytes.
CUtensorMapSwizzle swizzle_of(uint32_t bytes) {
  switch (bytes) {
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    default:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

// The scale table of `rows` rows: `[k / 128, padded rows]` words, columns `stride` words apart,
// loaded one 128-element step of `box_rows` rows at a time (DeepGEMM's `make_tma_sf_desc`).
bool encode_scales(CUtensorMap* map, const void* base, int64_t rows, int64_t stride,
                   uint32_t box_rows) {
  const uint64_t padded = static_cast<uint64_t>((rows + kWordRows - 1) / kWordRows * kWordRows);
  return encode(map, CU_TENSOR_MAP_DATA_TYPE_INT32, sizeof(uint32_t), base, padded, kK / 128,
                static_cast<uint64_t>(stride), box_rows, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
}

int launch(uniserve_mxfp8_gemm_call* call, cudaStream_t stream) {
  const auto m = static_cast<uint64_t>(call->m);
  CUtensorMap a, b, scales_a, scales_b, d;

  // The operands are K-major E4M3, read as bytes: a box is one swizzled 128-byte row of the
  // contraction by the rows one block of the cluster loads.
  if (!encode(&a, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, call->input, kK, m, kK, kSwizzleAB,
              kLoadBlockM, CU_TENSOR_MAP_SWIZZLE_128B) ||
      !encode(&b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, call->weight, kK, kN, kK, kSwizzleAB,
              kLoadBlockN, CU_TENSOR_MAP_SWIZZLE_128B)) {
    return refuse(call, "the operands' tensor maps cannot be encoded");
  }
  if (!encode_scales(&scales_a, call->input_scales, call->m, call->input_scale_stride,
                     kScaleBlockM) ||
      !encode_scales(&scales_b, call->weight_scales, call->n, call->weight_scale_stride,
                     kScaleBlockN)) {
    return refuse(call, "the scale tables' tensor maps cannot be encoded");
  }
  // The output's box is one store block; under a swizzle its row is the swizzle's width.
  const uint32_t store_inner =
      kSwizzleCD != 0 ? kSwizzleCD / sizeof(cutlass::bfloat16_t) : kStoreBlockN;
  if (!encode(&d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, sizeof(cutlass::bfloat16_t), call->out, kN, m,
              kN, store_inner, kStoreBlockM, swizzle_of(kSwizzleCD))) {
    return refuse(call, "the output's tensor map cannot be encoded");
  }

  const cudaError_t limited = cudaFuncSetAttribute(
      kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
  if (limited != cudaSuccess) {
    return refuse(call, cudaGetErrorString(limited));
  }

  // One block per multiprocessor, whole clusters: the patched scheduler strides by the grid.
  const auto grid = static_cast<unsigned>(call->multiprocessors / kCluster * kCluster);
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid, 1, 1);
  config.blockDim = dim3(kNonEpilogueThreads + kEpilogueThreads, 1, 1);
  config.dynamicSmemBytes = kSharedBytes;
  config.stream = stream;
  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = kCluster;
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;
  config.attrs = attributes;
  config.numAttrs = 1;

  const cudaError_t launched = cudaLaunchKernelEx(
      &config, kKernel, /*grouped_layout=*/static_cast<int*>(nullptr),
      static_cast<uint32_t>(call->m), kN, kK, deep_gemm::epilogue::operators::Identity{}, a, b,
      scales_a, scales_b, d);
  if (launched != cudaSuccess) {
    return refuse(call, cudaGetErrorString(launched));
  }
  return 0;
}

}  // namespace

extern "C" int uniserve_mxfp8_gemm(uniserve_mxfp8_gemm_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  std::memset(call->error, 0, sizeof(call->error));

  if (call->n != kN || call->k != kK) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "this module is compiled for a [%u, %u] weight and was handed [%lld, %lld]", kN,
                  kK, static_cast<long long>(call->n), static_cast<long long>(call->k));
    return refuse(call, message);
  }
  if (call->m < 0 || call->m > std::numeric_limits<uint32_t>::max()) {
    return refuse(call, "the kernel counts rows in 32 bits");
  }
  if (call->m == 0) {
    return 0;
  }
  // A scale column holds every row's word, padded to a whole 16-byte row of the tensor map.
  const int64_t input_rows = (call->m + kWordRows - 1) / kWordRows * kWordRows;
  const int64_t weight_rows = (call->n + kWordRows - 1) / kWordRows * kWordRows;
  if (call->input_scale_stride < input_rows || call->input_scale_stride % kWordRows != 0 ||
      call->weight_scale_stride < weight_rows || call->weight_scale_stride % kWordRows != 0) {
    return refuse(call, "a scale column holds every row's word, padded to four words");
  }
  if (!aligned(call->input) || !aligned(call->weight) || !aligned(call->input_scales) ||
      !aligned(call->weight_scales) || !aligned(call->out)) {
    return refuse(call, "every operand of the product starts on a 16-byte boundary");
  }
  if (call->multiprocessors < static_cast<int64_t>(kCluster)) {
    return refuse(call, "the device has fewer multiprocessors than one cluster of the tile");
  }

  return launch(call, reinterpret_cast<cudaStream_t>(stream));
}
