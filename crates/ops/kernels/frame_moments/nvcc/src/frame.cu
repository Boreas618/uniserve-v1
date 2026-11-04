// The per-frame group statistics of a video convolution's input.
//
// A causal video convolution of a pre-activation block reads
// pad(silu(group_norm(frame))). PyTorch evaluates the group norm's moments in its
// own kernel (`aten/src/ATen/native/cuda/group_norm_kernel.cu`,
// RowwiseMomentsCUDAKernel); this module computes them straight from the values with
// the same Welford operator, the same block reduction and the same launch width, and
// visits each row's elements in the order of the frame-folded
// [frames, channels, pixels] tensor PyTorch normalizes. The result is therefore bit
// for bit PyTorch's, which is what the embedding that consumes it relies on.
//
// The device code is the reference extension's
// (`uniserve_kernels/src/uniserve_kernels/norm/csrc/frame.cu`) with the ATen
// operator, the ATen reduction and the ATen stream replaced by their vendored
// counterparts and the call's stream.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "aten_block_reduce.cuh"
#include "uniserve_frame_moments.h"

namespace {

using uniserve::block_reduce;
using uniserve::kCudaBlockReduceNumThreads;
using uniserve::kCudaWarpSize;
using uniserve::kCudaWarpSizeUpperBound;
using uniserve::warp_reduce;
using uniserve::variance_and_mean;
using uniserve::welford_data;
using uniserve::welford_ops;

// Elements each thread of the moments kernel loads ahead of reducing them.
constexpr int kMomentLoads = 8;

// Mean and reciprocal standard deviation of row i = frame * groups + group: the
// group's `channels_per_group * pixels` values of one frame, visited in the order of
// the frame-folded [frames, channels, pixels] tensor PyTorch normalizes. A frame's
// pixels of one channel are contiguous.
__global__ void frame_moments_kernel(const float* values, int64_t stride_batch,
                                     int64_t stride_channel, int64_t stride_frame, int64_t frames,
                                     int64_t pixels, int64_t channels_per_group, int64_t groups,
                                     float eps, float* mean, float* rstd) {
  using WelfordType = welford_data<float, int64_t>;
  using WelfordOp = welford_ops<float, float, int64_t, variance_and_mean>;

  const int64_t i = blockIdx.x;
  const int64_t folded = i / groups;
  const int64_t group = i - folded * groups;
  const int64_t sample = folded / frames;
  const int64_t frame = folded - sample * frames;
  const float* row = values + sample * stride_batch + frame * stride_frame +
                     group * channels_per_group * stride_channel;
  const int64_t N = channels_per_group * pixels;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  // Element j of the row is pixel j % pixels of channel j / pixels; both advance by a
  // fixed stride between a thread's elements, so they are stepped rather than divided
  // out. A thread loads `kMomentLoads` of its elements before reducing them in order,
  // keeping several loads in flight without changing the order of its Welford updates.
  const int64_t step = blockDim.x;
  int64_t channel = threadIdx.x / pixels;
  int64_t pixel = threadIdx.x - channel * pixels;
  const int64_t channel_step = step / pixels;
  const int64_t pixel_step = step - channel_step * pixels;
  for (int64_t j0 = threadIdx.x; j0 < N; j0 += step * kMomentLoads) {
    float loaded[kMomentLoads];
#pragma unroll
    for (int u = 0; u < kMomentLoads; ++u) {
      if (j0 + u * step < N) {
        loaded[u] = static_cast<float>(row[channel * stride_channel + pixel]);
      }
      channel += channel_step;
      pixel += pixel_step;
      if (pixel >= pixels) {
        pixel -= pixels;
        channel += 1;
      }
    }
#pragma unroll
    for (int u = 0; u < kMomentLoads; ++u) {
      const int64_t j = j0 + u * step;
      if (j < N) {
        val = welford_op.reduce(val, loaded[u], i * N + j);
      }
    }
  }
  if (blockDim.x <= kCudaWarpSize) {
    val = warp_reduce(val, welford_op);
  } else {
    alignas(WelfordType) __shared__ char val_shared[sizeof(WelfordType) * kCudaWarpSizeUpperBound];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = block_reduce(val, welford_op, WelfordType(0, 0, 0, 0), val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    const variance_and_mean projected = welford_op.project(val);
    mean[i] = projected.mean;
    rstd[i] = ::rsqrtf(projected.variance + eps);
  }
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_frame_moments_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_frame_moments_call*>(call)->error, sizeof(call->error), "%s",
                message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_frame_moments(const uniserve_frame_moments_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->values == 0 || call->mean == 0 || call->rstd == 0) {
    return refuse(call, "frame_moments needs the values and both statistic buffers");
  }
  if (call->frames < 0 || call->pixels <= 0 || call->channels_per_group <= 0 ||
      call->groups <= 0) {
    return refuse(call, "frame_moments needs positive pixels, channels per group and groups");
  }
  if (call->values_frame < call->channels_per_group * call->pixels) {
    return refuse(call, "frame_moments needs a frame's strided row to hold the group's values");
  }
  const int64_t channels_per_group = call->channels_per_group;
  const int64_t pixels = call->pixels;
  const int64_t row_length = channels_per_group * pixels;
  // PyTorch's launch width, which fixes the reduction tree: one warp for a row a
  // warp already spans, its 512-thread block otherwise.
  const int64_t threads =
      row_length < kCudaBlockReduceNumThreads ? kCudaWarpSize : kCudaBlockReduceNumThreads;
  // The call structure carries no sample count, so the grid is one sample's rows. The
  // operator's caller runs one video sample per call; a call that batched several would
  // need the count in the structure.
  const int64_t blocks = call->frames * call->groups;
  const auto mean = reinterpret_cast<float*>(call->mean);
  const auto rstd = reinterpret_cast<float*>(call->rstd);
  if (blocks > 0) {
    frame_moments_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads),
                           0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const float*>(call->values), call->values_batch, call->values_channel,
        call->values_frame, call->frames, pixels, channels_per_group, call->groups, call->epsilon,
        mean, rstd);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      return refuse(call, cudaGetErrorString(status));
    }
  }
  return 0;
}
