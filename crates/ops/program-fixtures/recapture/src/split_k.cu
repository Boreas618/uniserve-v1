// SPDX-License-Identifier: Apache-2.0
//
// A `gemm` program that launches two kernels where the serving `cublas_lt`
// program launches one: the product into an FP32 scratch, then the alpha, the
// bias and the rounding into the output. It is a fixture for the program-layer
// delivery test, not a serving path: installing it as a new revision of
// `gemm`/`cublas_lt` changes the module's bytes and its number of launches, so a
// deployment that swaps it must recapture the graphs its GEMMs are in rather
// than replay one built over the old module.
//
// The arithmetic is the operator's: the contraction accumulates in FP32, the
// scalar multiplies before the epilogue, the bias adds after it, and the row
// rounds once into the output type — inside `ops::Tolerance::dot(k)` for the
// weight order whether the weight is stored `[k, n]` or `[n, k]`.
//
// The cell types are the descriptor's, known when the module is compiled:
// `UNISERVE_GEMM_INPUT`, `_WEIGHT`, `_ACCUMULATE`, `_OUTPUT` carry the operator
// crate's `DType` numbers, `UNISERVE_GEMM_ORDER` is 0 for `[k, n]` and 1 for
// `[n, k]`, `UNISERVE_GEMM_EPILOGUE` is 1 when a bias is added, and
// `UNISERVE_GEMM_K` is the contraction length the module was compiled for.

#include "uniserve_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>

#ifndef UNISERVE_GEMM_INPUT
#error "the package's build script defines UNISERVE_GEMM_INPUT"
#endif
#ifndef UNISERVE_GEMM_WEIGHT
#error "the package's build script defines UNISERVE_GEMM_WEIGHT"
#endif
#ifndef UNISERVE_GEMM_ACCUMULATE
#error "the package's build script defines UNISERVE_GEMM_ACCUMULATE"
#endif
#ifndef UNISERVE_GEMM_OUTPUT
#error "the package's build script defines UNISERVE_GEMM_OUTPUT"
#endif
#ifndef UNISERVE_GEMM_ORDER
#error "the package's build script defines UNISERVE_GEMM_ORDER"
#endif
#ifndef UNISERVE_GEMM_EPILOGUE
#error "the package's build script defines UNISERVE_GEMM_EPILOGUE"
#endif
#ifndef UNISERVE_GEMM_K
#error "the package's build script defines UNISERVE_GEMM_K"
#endif
#ifndef UNISERVE_GEMM_ENTRY
#error "the package's build script defines UNISERVE_GEMM_ENTRY: the package's launch entry"
#endif

// This fixture accumulates in FP32; a descriptor that says otherwise is one it
// does not implement, and the build script refuses it before this file sees it.
#if UNISERVE_GEMM_ACCUMULATE != 5
#error "this fixture accumulates in FP32 only: the descriptor's accumulate type must be f32"
#endif

namespace {

/// The output tile one block computes: `kTile` by `kTile` elements.
constexpr int kTile = 32;
/// The threads of one block, so each thread owns four elements of the tile.
constexpr int kThreads = 256;

/// One element of a buffer of the operator's element types, as FP32.
template <typename Element>
__device__ __forceinline__ float load(const void* base, int64_t index) {
  return static_cast<float>(static_cast<const Element*>(base)[index]);
}

/// One FP32 value into a buffer of the output's element type.
template <typename Element>
__device__ __forceinline__ void store(void* base, int64_t index, float value) {
  static_cast<Element*>(base)[index] = static_cast<Element>(value);
}

/// The product of one output tile into the FP32 scratch: the whole contraction,
/// in FP32, with the weight read in the order the checkpoint stores it.
template <typename Input, typename Weight>
__global__ void product_kernel(const UniserveGemmCall call, float* scratch) {
  const int row_base = static_cast<int>(blockIdx.y) * kTile;
  const int column_base = static_cast<int>(blockIdx.x) * kTile;
  for (int at = threadIdx.x; at < kTile * kTile; at += blockDim.x) {
    const int row = row_base + at / kTile;
    const int column = column_base + at % kTile;
    if (row >= call.m || column >= call.n) {
      continue;
    }
    float sum = 0.0f;
    for (int64_t index = 0; index < call.k; ++index) {
      const float x = load<Input>(call.input, static_cast<int64_t>(row) * call.input_row_stride +
                                                  index);
#if UNISERVE_GEMM_ORDER == 0
      // `[k, n]`: the contraction axis first, so a column's elements are one
      // row apart.
      const float w = load<Weight>(
          call.weight, static_cast<int64_t>(index) * call.weight_row_stride + column);
#else
      // `[n, k]`: the output axis first, so a column's row is contiguous.
      const float w = load<Weight>(
          call.weight, static_cast<int64_t>(column) * call.weight_row_stride + index);
#endif
      sum = fmaf(x, w, sum);
    }
    scratch[static_cast<int64_t>(row) * call.n + column] = sum;
  }
}

/// The epilogue: the scalar, the bias, and one rounding into the output type.
template <typename Output>
__global__ void epilogue_kernel(const UniserveGemmCall call, const float* scratch) {
  const int64_t elements = call.m * call.n;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; at < elements;
       at += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = at / call.n;
    const int64_t column = at % call.n;
    float value = scratch[at] * call.alpha;
#if UNISERVE_GEMM_EPILOGUE == 1
    value += load<Output>(call.bias, column);
#endif
    store<Output>(call.out, row * call.out_row_stride + column, value);
  }
}

/// Writes `message` into the call's error buffer and answers its status.
int32_t refuse(UniserveGemmCall* call, const char* message) {
  std::snprintf(call->error, 512, "%s", message);
  return 1;
}

template <typename Output>
int32_t epilogue(const UniserveGemmCall& call, const float* scratch, cudaStream_t stream) {
  const int64_t elements = call.m * call.n;
  const int64_t threads = 256;
  const int64_t blocks = (elements + threads - 1) / threads;
  epilogue_kernel<Output><<<static_cast<unsigned int>(blocks), static_cast<int>(threads), 0,
                             stream>>>(call, scratch);
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

template <typename Input, typename Weight>
int32_t product(const UniserveGemmCall& call, float* scratch, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((call.n + kTile - 1) / kTile),
                  static_cast<unsigned int>((call.m + kTile - 1) / kTile));
  product_kernel<Input, Weight><<<grid, kThreads, 0, stream>>>(call, scratch);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

}  // namespace

extern "C" int32_t UNISERVE_GEMM_ENTRY(UniserveGemmCall* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->input == nullptr || call->weight == nullptr || call->out == nullptr) {
    return refuse(call, "a product needs the addresses of its input, weight and output");
  }
  if (call->m <= 0 || call->n <= 0 || call->k <= 0) {
    return refuse(call, "a product needs positive rows, columns and contraction length");
  }
  if (call->k != UNISERVE_GEMM_K) {
    return refuse(call, "this module was compiled for another contraction length");
  }
  if (UNISERVE_GEMM_EPILOGUE == 1 && call->bias == nullptr) {
    return refuse(call, "this descriptor adds a bias and the call passed none");
  }
  if (UNISERVE_GEMM_EPILOGUE == 0 && call->bias != nullptr) {
    return refuse(call, "the call passed a bias this descriptor does not add");
  }

  const cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
  const int64_t elements = call->m * call->n;
  float* scratch = nullptr;
  if (cudaMallocAsync(&scratch, static_cast<size_t>(elements) * sizeof(float), cs) !=
      cudaSuccess) {
    return refuse(call, "the product's scratch could not be allocated");
  }

  int32_t status = 0;
  // The element types are the descriptor's; each pair the operator pairs is a
  // compile-time branch here.
  if (UNISERVE_GEMM_INPUT == 4 && UNISERVE_GEMM_WEIGHT == 4) {
    status = product<__nv_bfloat16, __nv_bfloat16>(*call, scratch, cs);
  } else if (UNISERVE_GEMM_INPUT == 3 && UNISERVE_GEMM_WEIGHT == 3) {
    status = product<__half, __half>(*call, scratch, cs);
  } else if (UNISERVE_GEMM_INPUT == 5 && UNISERVE_GEMM_WEIGHT == 5) {
    status = product<float, float>(*call, scratch, cs);
  } else {
    cudaFreeAsync(scratch, cs);
    return refuse(call, "this fixture's module pairs one element type for input and weight");
  }
  if (status == 0) {
    if (UNISERVE_GEMM_OUTPUT == 4) {
      status = epilogue<__nv_bfloat16>(*call, scratch, cs);
    } else if (UNISERVE_GEMM_OUTPUT == 3) {
      status = epilogue<__half>(*call, scratch, cs);
    } else if (UNISERVE_GEMM_OUTPUT == 5) {
      status = epilogue<float>(*call, scratch, cs);
    } else {
      status = 5;
    }
    if (status != 0) {
      status = refuse(call, "the product's epilogue could not be launched");
    }
  } else {
    status = refuse(call, "the product could not be launched");
  }
  cudaFreeAsync(scratch, cs);
  return status;
}
