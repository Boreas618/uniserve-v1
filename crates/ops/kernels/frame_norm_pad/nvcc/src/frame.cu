// The padded, normalized and activated input of a video convolution.
//
// A causal video convolution of a pre-activation block reads
// pad(silu(group_norm(frame))) for every frame of [batch, channels, frames, height,
// width] values. PyTorch evaluates that as a frame-folding copy, its group-norm
// moments and apply kernels, a SiLU kernel and the padding, five passes over the
// activation; this module writes the whole composition in one pass, with the
// arithmetic of PyTorch 2.13's CUDA kernels: the fused parameters of
// ComputeFusedParamsCUDAKernel, its `a * x + b`, GroupNorm1dForward's
// single-pixel expressions, and SiLU's `x / (1 + exp(-x))` from
// ActivationSiluKernel.cu. The module compiles without fast math, as PyTorch does,
// so the composition is bit for bit PyTorch's.
//
// The device code is the reference extension's
// (`uniserve_kernels/src/uniserve_kernels/norm/csrc/frame.cu`), unchanged; the
// entry around it is this package's, and it launches the kernel from the call's
// stream.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>

#include "uniserve_frame_norm_pad.h"

namespace {

// One output row segment: `kPixels` output columns by `kChannels` channels.
constexpr int kPixels = 64;
constexpr int kChannels = 32;
constexpr int kRows = 8;
constexpr int kThreads = 32 * kRows;

__device__ __forceinline__ int64_t source_index(int64_t index, int64_t extent, bool reflect) {
  if (reflect) {
    index = index < 0 ? -index : index;
    return index >= extent ? 2 * extent - 2 - index : index;
  }
  return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
}

// How a frame's values normalize, following the branch PyTorch's
// GroupNormKernelImplInternal takes for the parameters and frame size.
enum Normalization : int {
  // ComputeFusedParamsCUDAKernel's a = rstd * gamma and b = -a * mean + beta,
  // then the apply's `a * x + b`; per channel (first, second) = (a, b).
  kFused = 0,
  // No affine parameters: `(x - mean) * rstd`; per channel (mean, rstd).
  kCentered = 1,
  // One-pixel frames (GroupNorm1dForward); per channel (mean, rstd).
  kSinglePixel = 2,
};

// GroupNorm1dForward's expressions for one-pixel frames.
__device__ __forceinline__ float normalize_single(
    float x, float mean, float rstd, const float* weight, const float* bias, int channel) {
  if (weight != nullptr && bias != nullptr) {
    return (x - mean) * rstd * weight[channel] + bias[channel];
  }
  if (weight != nullptr) {
    return (x - mean) * rstd * weight[channel];
  }
  if (bias != nullptr) {
    return (x - mean) * rstd + bias[channel];
  }
  return (x - mean) * rstd;
}

// Writes pad(silu(group_norm(values))) for one (sample, output frame, output
// row) and a run of `kPixels` output columns, every channel in blocks of
// `kChannels`. Output frames before `front` are zero; every other element
// normalizes and activates its reflected or replicated source pixel with
// its channel's parameters, which the block forms once for its frame.
// Channels-last output is transposed through shared memory so both the
// column-major reads and the channel-major writes coalesce.
template <bool kChannelsLast>
__global__ void __launch_bounds__(kThreads) frame_norm_pad_kernel(
    const float* values,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_frame,
    const float* mean,
    const float* rstd,
    const float* weight,
    const float* bias,
    float* out,
    int64_t out_batch,
    int64_t out_channel,
    int64_t out_frame_stride,
    int64_t out_row,
    int64_t out_column,
    int channels,
    int frames,
    int height,
    int width,
    int out_frames,
    int out_height,
    int out_width,
    int channels_per_group,
    int groups,
    int top,
    int left,
    int front,
    bool reflect,
    int normalization) {
  extern __shared__ float parameters[];  // [2, channels]
  __shared__ float tile[kChannels][kPixels + 1];

  const int row_id = blockIdx.x;
  const int row = row_id % out_height;
  const int out_frame = (row_id / out_height) % out_frames;
  const int sample = row_id / (out_height * out_frames);
  const int column0 = blockIdx.y * kPixels;
  const int frame = out_frame - front;
  const int thread = threadIdx.y * 32 + threadIdx.x;
  float* destination = out + sample * out_batch + out_frame * out_frame_stride + row * out_row;

  if (frame < 0) {
    // A leading zero frame of the causal padding.
    for (int index = thread; index < channels * kPixels; index += kThreads) {
      const int channel = kChannelsLast ? index % channels : index / kPixels;
      const int column = column0 + (kChannelsLast ? index / channels : index % kPixels);
      if (column < out_width) {
        destination[channel * out_channel + column * out_column] = 0.0f;
      }
    }
    return;
  }

  const int group_base = (sample * frames + frame) * groups;
  for (int channel = thread; channel < channels; channel += kThreads) {
    const int ng = group_base + channel / channels_per_group;
    if (normalization == kFused) {
      const float scale = weight == nullptr ? rstd[ng] : rstd[ng] * weight[channel];
      const float shift = -scale * mean[ng] + (bias == nullptr ? 0 : bias[channel]);
      parameters[channel] = scale;
      parameters[channels + channel] = shift;
    } else {
      parameters[channel] = mean[ng];
      parameters[channels + channel] = rstd[ng];
    }
  }
  __syncthreads();

  // Each thread reads columns threadIdx.x and threadIdx.x + 32 of the source
  // row; a channel's pixels are contiguous within a frame.
  const int source_row = static_cast<int>(source_index(row - top, height, reflect));
  const float* source = values + sample * stride_batch + frame * stride_frame +
      static_cast<int64_t>(source_row) * width;
  int source_columns[kPixels / 32];
  bool live[kPixels / 32];
#pragma unroll
  for (int k = 0; k < kPixels / 32; ++k) {
    const int column = column0 + threadIdx.x + 32 * k;
    live[k] = column < out_width;
    source_columns[k] = static_cast<int>(source_index(column - left, width, reflect));
  }

  for (int channel0 = 0; channel0 < channels; channel0 += kChannels) {
    // All of a thread's loads of this channel block are issued before any
    // is used, so they are in flight together.
    float loaded[kChannels / kRows][kPixels / 32];
#pragma unroll
    for (int r = 0; r < kChannels / kRows; ++r) {
      const int channel = channel0 + threadIdx.y + r * kRows;
#pragma unroll
      for (int k = 0; k < kPixels / 32; ++k) {
        if (channel < channels && live[k]) {
          loaded[r][k] = source[channel * stride_channel + source_columns[k]];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kChannels / kRows; ++r) {
      const int c = threadIdx.y + r * kRows;
      const int channel = channel0 + c;
      if (channel >= channels) {
        continue;
      }
      const float first = parameters[channel];
      const float second = parameters[channels + channel];
#pragma unroll
      for (int k = 0; k < kPixels / 32; ++k) {
        if (!live[k]) {
          continue;
        }
        const float x = loaded[r][k];
        float y;
        if (normalization == kFused) {
          y = first * x + second;
        } else if (normalization == kCentered) {
          y = (x - first) * second;
        } else {
          y = normalize_single(x, first, second, weight, bias, channel);
        }
        // SiLU (ActivationSiluKernel.cu).
        const float value = y / (1.0f + ::exp(-y));
        const int column_offset = threadIdx.x + 32 * k;
        if constexpr (kChannelsLast) {
          tile[c][column_offset] = value;
        } else {
          destination[channel * out_channel + (column0 + column_offset) * out_column] = value;
        }
      }
    }
    if constexpr (kChannelsLast) {
      __syncthreads();
      // Writes run along channels, the innermost storage axis.
      const int channel = channel0 + threadIdx.x;
      for (int p = threadIdx.y; p < kPixels; p += kRows) {
        const int column = column0 + p;
        if (column < out_width && channel < channels) {
          destination[column * out_column + channel * out_channel] = tile[threadIdx.x][p];
        }
      }
      __syncthreads();
    }
  }
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_frame_norm_pad_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_frame_norm_pad_call*>(call)->error, sizeof(call->error), "%s",
                message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_frame_norm_pad(const uniserve_frame_norm_pad_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->values == 0 || call->mean == 0 || call->rstd == 0 || call->out == 0) {
    return refuse(call, "frame_norm_pad needs the values, both statistics and the output");
  }
  if (call->channels <= 0 || call->frames < 0 || call->height <= 0 || call->width <= 0 ||
      call->out_frames <= 0 || call->out_rows <= 0 || call->out_columns <= 0) {
    return refuse(call, "frame_norm_pad needs positive source and output extents");
  }
  if (call->channels_per_group <= 0 || call->groups <= 0) {
    return refuse(call, "frame_norm_pad needs positive channels per group and groups");
  }
  if (call->normalization != kFused && call->normalization != kCentered &&
      call->normalization != kSinglePixel) {
    return refuse(call, "frame_norm_pad was given no normalization branch it knows");
  }
  const int64_t widest = std::numeric_limits<int>::max();
  for (int64_t extent : {call->channels, call->frames, call->height, call->width, call->out_frames,
                         call->out_rows, call->out_columns}) {
    if (extent > widest) {
      return refuse(call, "frame_norm_pad addresses frames, rows and columns with 32-bit indices");
    }
  }
  // The storage's flag is the descriptor's, and the operator's strides are its consequence:
  // channels-last stores the columns fastest, so a channel's stride is the column count, and
  // channel-first stores the columns of one channel contiguously, so the channel stride is the
  // frame's whole extent.
  const bool channels_last = call->out_channel == call->out_columns &&
                             call->out_height == call->channels * call->out_columns;
  // The call structure carries no sample count, so the grid covers one sample's rows.
  // The operator's caller runs one video sample per call; a call that batched several
  // would need the count in the structure.
  const dim3 grid(static_cast<unsigned int>(call->out_frames * call->out_rows),
                  static_cast<unsigned int>((call->out_columns + kPixels - 1) / kPixels));
  const dim3 block(32, kRows);
  if (grid.x == 0 || grid.y == 0) {
    return 0;
  }
  const size_t parameter_bytes = 2 * static_cast<size_t>(call->channels) * sizeof(float);
  const auto values = reinterpret_cast<const float*>(call->values);
  const auto mean = reinterpret_cast<const float*>(call->mean);
  const auto rstd = reinterpret_cast<const float*>(call->rstd);
  const auto weight = call->weight == 0 ? nullptr : reinterpret_cast<const float*>(call->weight);
  const auto bias = call->bias == 0 ? nullptr : reinterpret_cast<const float*>(call->bias);
  auto out = reinterpret_cast<float*>(call->out);
  auto launch = [&](auto kernel) {
    if (parameter_bytes > 48 * 1024) {
      const cudaError_t attribute = cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(parameter_bytes));
      if (attribute != cudaSuccess) {
        return attribute;
      }
    }
    kernel<<<grid, block, parameter_bytes, reinterpret_cast<cudaStream_t>(stream)>>>(
        values, call->values_batch, call->values_channel, call->values_frame, mean, rstd, weight,
        bias, out, call->out_batch, call->out_channel, call->out_frame, call->out_height,
        call->out_width, static_cast<int>(call->channels), static_cast<int>(call->frames),
        static_cast<int>(call->height), static_cast<int>(call->width),
        static_cast<int>(call->out_frames), static_cast<int>(call->out_rows),
        static_cast<int>(call->out_columns), static_cast<int>(call->channels_per_group),
        static_cast<int>(call->groups), static_cast<int>(call->top), static_cast<int>(call->left),
        static_cast<int>(call->front), call->reflect != 0, static_cast<int>(call->normalization));
    return cudaGetLastError();
  };
  const cudaError_t status = channels_last ? launch(frame_norm_pad_kernel<true>)
                                           : launch(frame_norm_pad_kernel<false>);
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }
  return 0;
}
