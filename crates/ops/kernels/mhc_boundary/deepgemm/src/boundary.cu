// SPDX-License-Identifier: Apache-2.0

// The mhc_boundary operator's fused form: one boundary of a residual of four BF16 streams
// launched as DeepGEMM's SM100 mega mHC kernel in its shifted form.
//
// The kernel is DeepGEMM's own (`sm100_mega_mhc_impl`, the `deepgemm` pin), with the package's
// patch (`patches/launched-grid.patch`) that strides its persistent tasks by the launched grid
// and keeps each block's two arrival lines side by side. It owns the boundary: per block of 64
// tokens and split of the column steps, TMA loads of the streams, the pending output and the
// projection; the pending output written back (`h' = round(post_k y + sum_j comb[j, k] h_j)`,
// FP32 with fused multiply-adds, one BF16 rounding); the product of the written streams with the
// projection on the tensor cores (one TF32 pass: the 16-bit streams are exact in TF32, the
// projection's tensor map reads it as TF32); the collapsed input with the carried pre-mix; the
// split's partials and its arrival; and, once a block's splits arrived, its mixes (sigmoids and
// the Sinkhorn iterations) and the input's RMS normalization. This unit owns what DeepGEMM's
// Python extension did around it (`csrc/apis/mega_mhc.hpp` and
// `csrc/jit_kernels/impls/sm100_mega_mhc.hpp`):
//
// * the template arguments: the descriptor's width, and one instance per split count the
//   operator's rule states for the descriptor's token class (`src/build.py`), which the call
//   names (`BoundaryDescriptor::fused_splits`);
// * the eight tensor maps: the streams, the pending output, the projection (read as TF32), the
//   post-mix, the combination matrix, the carried pre-mix, the written streams and the input;
// * the mixes' and the normalization's arguments, with the method's post-mix factor of two;
// * the launch: one block per multiprocessor of the device, with the kernel's shared memory.
//
// The workspace is the operator's: the partials (`[blocks, splits, 64, mixes + 2]` FP32) and the
// blocks' arrival lines (`[blocks, 2]` lines of 128 bytes), which the kernel tags with its grid's
// identity, so any earlier launch's tags or zeros are a valid start.

#include <uniserve_mhc_boundary.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include <deep_gemm/impls/sm100_mega_mhc.cuh>

#if !defined(UNISERVE_HIDDEN) || !defined(UNISERVE_SPLITS_0) || !defined(UNISERVE_SPLITS_1) || \
    !defined(UNISERVE_SPLITS_2) || !defined(UNISERVE_SPLITS_3) ||                              \
    !defined(UNISERVE_PLANNED_MULTIPROCESSORS)
#error "the build defines the descriptor's width and the split counts of its class (src/build.py)"
#endif

namespace {

namespace layout = deep_gemm::layout::mega_mhc;

// The descriptor's width, and the split counts the operator's rule states for its token class.
constexpr uint32_t kHidden = UNISERVE_HIDDEN;
constexpr uint32_t kSplits[4] = {UNISERVE_SPLITS_0, UNISERVE_SPLITS_1, UNISERVE_SPLITS_2,
                                 UNISERVE_SPLITS_3};

// The kernel's template asks for a multiprocessor count; the patched kernel strides by the
// launched grid, so this plans nothing.
constexpr uint32_t kPlannedSMs = UNISERVE_PLANNED_MULTIPROCESSORS;

// The kernel's own geometry: four streams, 64-token blocks of 64-column steps, 128-byte swizzle.
constexpr uint32_t kStreams = layout::kNumRoutes;
constexpr uint32_t kMixes = layout::kNumHCOutputs;
constexpr uint32_t kBlockM = layout::BLOCK_M;
constexpr uint32_t kBlockK = layout::BLOCK_K;
constexpr uint32_t kSwizzle = layout::kSwizzleMode;
constexpr uint32_t kSharedBytes = sizeof(layout::SharedStorage);

// The method's post-mix factor (`post = 2 sigmoid(...)`) and the input's unit scale.
constexpr float kPostScale = 2.0f;
constexpr float kNormScale = 1.0f;

static_assert(kHidden % (layout::kDefaultNumSplits * kBlockK) == 0,
              "the streams are whole column steps of every split");
static_assert(kSharedBytes <= 232448, "the kernel fits a block's shared memory");

// The shifted form that writes the normalized input as BF16 and no FP8 copy of it.
template <uint32_t kNumSplits>
constexpr auto kKernel = &deep_gemm::sm100_mega_mhc_impl<kHidden, kNumSplits, kPlannedSMs,
                                                         /*kIsShifted=*/true,
                                                         /*kStoreBF16=*/true,
                                                         /*kStoreFP8=*/false,
                                                         /*SF_BLOCK_M=*/0>;

// Every instance has one signature: eight tensor maps, the mixes' and the normalization's
// arguments, the partials and the arrival lines.
using Kernel = std::remove_cv_t<decltype(kKernel<kSplits[0]>)>;

int refuse(uniserve_mhc_boundary_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

bool aligned(const void* address) {
  return address != nullptr && reinterpret_cast<uintptr_t>(address) % 16 == 0;
}

// The instance of `splits` column splits, or none when the build compiled none of that count.
Kernel kernel_of(int64_t splits) {
  if (splits == kSplits[0]) return kKernel<kSplits[0]>;
  if (splits == kSplits[1]) return kKernel<kSplits[1]>;
  if (splits == kSplits[2]) return kKernel<kSplits[2]>;
  if (splits == kSplits[3]) return kKernel<kSplits[3]>;
  return nullptr;
}

// A tensor map of `rank` dimensions over `bytes`-byte elements of `type` at `base`: `extents`
// elements, the outer dimensions `strides` elements apart, loaded in `box` boxes under
// `swizzle` (DeepGEMM's `make_tma_2d_desc` and `make_tma_3d_desc`: under a swizzle the box's
// inner extent is the swizzle's width). Out-of-bounds elements of a box load as zeros and are
// never stored.
bool encode(CUtensorMap* map, CUtensorMapDataType type, uint32_t bytes, const void* base,
            uint32_t rank, const uint64_t* extents, const uint64_t* strides,
            const uint32_t* box, uint32_t swizzle) {
  cuuint64_t dims[3];
  cuuint64_t steps[2];
  cuuint32_t boxes[3];
  const cuuint32_t elements[3] = {1, 1, 1};
  for (uint32_t axis = 0; axis < rank; ++axis) {
    dims[axis] = extents[axis];
    boxes[axis] = box[axis];
  }
  for (uint32_t axis = 0; axis + 1 < rank; ++axis) {
    steps[axis] = strides[axis] * bytes;
    if (steps[axis] % 16 != 0) {
      return false;
    }
  }
  if (swizzle != 0) {
    boxes[0] = swizzle / bytes;
  }
  const CUtensorMapSwizzle mode = swizzle == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                                  : swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                  : swizzle == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                                  : CU_TENSOR_MAP_SWIZZLE_NONE;
  return cuTensorMapEncodeTiled(map, type, rank, const_cast<void*>(base), dims, steps, boxes,
                                elements, CU_TENSOR_MAP_INTERLEAVE_NONE, mode,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

int launch(uniserve_mhc_boundary_call* call, Kernel kernel, cudaStream_t stream) {
  const uint64_t tokens = static_cast<uint64_t>(call->tokens);
  const uint64_t width = static_cast<uint64_t>(kStreams) * kHidden;
  CUtensorMap streams, output, weight, post, comb, pre, written, input;

  // The streams and the written streams: `[tokens, streams, hidden]` read as `hidden` by
  // `tokens` by `streams`, one 64 x 64 block of every stream a load.
  const uint64_t residual_extents[3] = {kHidden, tokens, kStreams};
  const uint64_t residual_strides[2] = {width, kHidden};
  const uint32_t residual_box[3] = {kBlockK, kBlockM, kStreams};
  // The pending output and the input: `[tokens, hidden]`.
  const uint64_t row_extents[2] = {kHidden, tokens};
  const uint64_t row_strides[1] = {kHidden};
  const uint32_t row_box[2] = {kBlockK, kBlockM};
  // The projection `[mixes, streams * hidden]` read as `hidden` by `mixes` by `streams`, as
  // TF32: one step of every mix of every stream a load.
  const uint64_t weight_extents[3] = {kHidden, kMixes, kStreams};
  const uint64_t weight_strides[2] = {width, kHidden};
  const uint32_t weight_box[3] = {kBlockK, kMixes, kStreams};
  // The mixes: `[tokens, streams]` and `[tokens, streams, streams]` FP32, a block's rows a load.
  const uint64_t mix_extents[2] = {kStreams, tokens};
  const uint64_t mix_strides[1] = {kStreams};
  const uint32_t mix_box[2] = {kStreams, kBlockM};
  const uint64_t comb_extents[2] = {kStreams * kStreams, tokens};
  const uint64_t comb_strides[1] = {kStreams * kStreams};
  const uint32_t comb_box[2] = {kStreams * kStreams, kBlockM};

  const CUtensorMapDataType bf16 = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  const CUtensorMapDataType fp32 = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  if (!encode(&streams, bf16, 2, call->streams, 3, residual_extents, residual_strides,
              residual_box, kSwizzle) ||
      !encode(&written, bf16, 2, call->written, 3, residual_extents, residual_strides,
              residual_box, kSwizzle) ||
      !encode(&output, bf16, 2, call->output, 2, row_extents, row_strides, row_box, kSwizzle) ||
      !encode(&input, bf16, 2, call->input, 2, row_extents, row_strides, row_box, kSwizzle) ||
      !encode(&weight, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 4, call->weight, 3, weight_extents,
              weight_strides, weight_box, kSwizzle) ||
      !encode(&post, fp32, 4, call->post, 2, mix_extents, mix_strides, mix_box, 0) ||
      !encode(&pre, fp32, 4, call->pre, 2, mix_extents, mix_strides, mix_box, 0) ||
      !encode(&comb, fp32, 4, call->comb, 2, comb_extents, comb_strides, comb_box,
              kSwizzle / 2)) {
    return refuse(call, "the operands' tensor maps cannot be encoded");
  }

  const layout::MixArgs mixes = {
      .scales = static_cast<const float*>(call->scale),
      .bases = static_cast<const float*>(call->base),
      .new_prev_mix = static_cast<float*>(call->next_pre),
      .new_post_mix = static_cast<float*>(call->next_post),
      .new_comb_res_mix = static_cast<float*>(call->next_comb),
      .hc_norm_eps = call->rms_eps,
      .hc_pre_eps = call->eps,
      .hc_post_scale = kPostScale,
      .sinkhorn_eps = call->eps,
      .num_sinkhorn_iters = static_cast<uint32_t>(call->iterations),
  };
  const layout::NormArgs norm = {
      .num_tokens = static_cast<uint32_t>(tokens),
      .weight = static_cast<const nv_bfloat16*>(call->norm),
      .new_residual = static_cast<const nv_bfloat16*>(call->written),
      .eps = call->norm_eps,
      .scale = kNormScale,
      .y_bf16 = static_cast<nv_bfloat16*>(call->input),
      .y_fp8 = nullptr,
      .y_primary_sf = nullptr,
      .y_primary_sf_stride_token = 0,
      .y_primary_sf_stride_word = 0,
      .y_shared_sf = nullptr,
      .y_shared_sf_stride_word = 0,
  };

  const cudaError_t limited = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
  if (limited != cudaSuccess) {
    return refuse(call, cudaGetErrorString(limited));
  }

  // One block per multiprocessor: the patched kernel strides its tasks by the grid.
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(call->multiprocessors), 1, 1);
  config.blockDim = dim3(layout::kNumThreads, 1, 1);
  config.dynamicSmemBytes = kSharedBytes;
  config.stream = stream;
  config.attrs = nullptr;
  config.numAttrs = 0;

  const cudaError_t launched =
      cudaLaunchKernelEx(&config, kernel, streams, output, weight, post, comb, pre, written, input,
                         mixes, norm, call->partials, static_cast<uint64_t*>(call->lines));
  if (launched != cudaSuccess) {
    return refuse(call, cudaGetErrorString(launched));
  }
  return 0;
}

}  // namespace

extern "C" int uniserve_mhc_boundary(uniserve_mhc_boundary_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  std::memset(call->error, 0, sizeof(call->error));

  if (call->stream_count != kStreams || call->hidden != kHidden) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "this module is compiled for %u streams of %u elements and was handed %lld of "
                  "%lld",
                  kStreams, kHidden, static_cast<long long>(call->stream_count),
                  static_cast<long long>(call->hidden));
    return refuse(call, message);
  }
  if (call->tokens < 0 || call->tokens > static_cast<int64_t>(layout::kNumMaxTokens)) {
    return refuse(call, "the kernel's blocks hold at most 2^20 tokens");
  }
  if (call->tokens == 0) {
    return 0;
  }
  if (call->iterations < 1) {
    return refuse(call, "a combination matrix of no Sinkhorn iteration");
  }
  const Kernel kernel = kernel_of(call->splits);
  if (kernel == nullptr) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "this module compiles %u, %u, %u and %u column splits and was handed %lld",
                  kSplits[0], kSplits[1], kSplits[2], kSplits[3],
                  static_cast<long long>(call->splits));
    return refuse(call, message);
  }
  // The tensor maps' bases, and the 16-byte vectors the normalization reads and writes.
  for (const void* address :
       {call->streams, call->output, call->post, call->comb, static_cast<const void*>(call->written),
        call->weight, call->pre, call->norm, static_cast<const void*>(call->input)}) {
    if (!aligned(address)) {
      return refuse(call, "every tensor the kernel maps starts on a 16-byte boundary");
    }
  }
  if (call->scale == nullptr || call->base == nullptr || call->next_pre == nullptr ||
      call->next_post == nullptr || call->next_comb == nullptr || call->partials == nullptr ||
      reinterpret_cast<uintptr_t>(call->lines) % 8 != 0 || call->lines == nullptr) {
    return refuse(call, "the mixes, the partials and 8-byte aligned arrival lines are given");
  }
  if (call->multiprocessors < 1) {
    return refuse(call, "a launch over no multiprocessor");
  }

  return launch(call, kernel, reinterpret_cast<cudaStream_t>(stream));
}
