// SPDX-License-Identifier: Apache-2.0
//
// The recapture fixture's unit: a `gemm` module that launches two kernels where
// the serving `cublas_lt` package launches one: the product into an FP32
// scratch, then the alpha, the bias and the rounding into the output. It is a
// fixture for the generation delivery test, not a serving path: installing the
// package it is compiled in as a new revision of `gemm`/`cublas_lt` changes the
// module's bytes and its number of launches, so a deployment that swaps it must
// recapture the graphs its GEMMs are in rather than replay one built over the old
// module.
//
// The arithmetic is the operator's: the contraction accumulates in FP32, the
// scalar multiplies before the epilogue, the bias adds after it, and the row
// rounds once into the output type — inside `ops::Tolerance::dot(k)` for the
// weight order whether the weight is stored `[k, n]` or `[n, k]`.
//
// It replaces the serving package's unit and reads the serving manifest's
// definitions: the cuBLASLt data types of the input, the weight and the output,
// the compute type of the accumulation, the weight order (a
// `uniserve_weight_order` enumerator: `[k, n]` or `[n, k]`) and the epilogue (a
// `uniserve_epilogue` enumerator: none, or add the bias).

#include <uniserve_gemm.h>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>

#if !defined(UNISERVE_INPUT_TYPE) || !defined(UNISERVE_WEIGHT_TYPE) ||       \
    !defined(UNISERVE_ACCUMULATE_TYPE) || !defined(UNISERVE_OUTPUT_TYPE) || \
    !defined(UNISERVE_WEIGHT_ORDER) || !defined(UNISERVE_EPILOGUE)
#error "the package's manifest defines the descriptor's compile-time choices"
#endif

namespace {

/// The element type a cuBLASLt data type names.
template <cudaDataType_t>
struct Element;
template <>
struct Element<CUDA_R_32F> {
  using Type = float;
};
template <>
struct Element<CUDA_R_16F> {
  using Type = __half;
};
template <>
struct Element<CUDA_R_16BF> {
  using Type = __nv_bfloat16;
};

using Input = Element<UNISERVE_INPUT_TYPE>::Type;
using Weight = Element<UNISERVE_WEIGHT_TYPE>::Type;
using Output = Element<UNISERVE_OUTPUT_TYPE>::Type;

// This fixture accumulates in FP32 and pairs one element type for its input and
// its weight; a descriptor that says otherwise is one it does not implement.
static_assert(UNISERVE_ACCUMULATE_TYPE == CUBLAS_COMPUTE_32F,
              "this fixture accumulates in FP32 only: the descriptor's accumulate type must be f32");
static_assert(UNISERVE_INPUT_TYPE == UNISERVE_WEIGHT_TYPE,
              "this fixture takes one element type for its input and its weight");

/// The output tile one block computes: `kTile` by `kTile` elements.
constexpr int kTile = 32;
/// The threads of one block, so each thread owns four elements of the tile.
constexpr int kThreads = 256;

/// One element of a buffer of the operator's element types, as FP32.
template <typename Value>
__device__ __forceinline__ float load(const void* base, int64_t index) {
  return static_cast<float>(static_cast<const Value*>(base)[index]);
}

/// One FP32 value into a buffer of the output's element type.
template <typename Value>
__device__ __forceinline__ void store(void* base, int64_t index, float value) {
  static_cast<Value*>(base)[index] = static_cast<Value>(value);
}

/// The product of one output tile into the FP32 scratch: the whole contraction,
/// in FP32, with the weight read in the order the checkpoint stores it.
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
      // `[k, n]`: the contraction axis first, so a column's elements are one row
      // apart. `[n, k]`: the output axis first, so a column's row is contiguous.
      const int64_t at = UNISERVE_WEIGHT_ORDER == UNISERVE_WEIGHT_ORDER_KXN
                             ? static_cast<int64_t>(index) * call.weight_row_stride + column
                             : static_cast<int64_t>(column) * call.weight_row_stride + index;
      const float w = load<Weight>(call.weight, at);
      sum = fmaf(x, w, sum);
    }
    scratch[static_cast<int64_t>(row) * call.n + column] = sum;
  }
}

/// The epilogue: the scalar, the bias, and one rounding into the output type.
__global__ void epilogue_kernel(const UniserveGemmCall call, const float* scratch) {
  const int64_t elements = call.m * call.n;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; at < elements;
       at += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = at / call.n;
    const int64_t column = at % call.n;
    float value = scratch[at] * call.alpha;
    if (UNISERVE_EPILOGUE == UNISERVE_EPILOGUE_BIAS) {
      value += load<Output>(call.bias, column);
    }
    store<Output>(call.out, row * call.out_row_stride + column, value);
  }
}

/// Writes `message` into the call's error buffer and answers its status.
int32_t refuse(UniserveGemmCall* call, const char* message) {
  std::snprintf(call->error, UNISERVE_MESSAGE, "%s", message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_gemm(UniserveGemmCall* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->input == nullptr || call->weight == nullptr || call->out == nullptr) {
    return refuse(call, "a product needs the addresses of its input, weight and output");
  }
  if (call->m <= 0 || call->n <= 0 || call->k <= 0) {
    return refuse(call, "a product needs positive rows, columns and contraction length");
  }
  if (UNISERVE_EPILOGUE == UNISERVE_EPILOGUE_BIAS && call->bias == nullptr) {
    return refuse(call, "this descriptor adds a bias and the call passed none");
  }
  if (UNISERVE_EPILOGUE == UNISERVE_EPILOGUE_NONE && call->bias != nullptr) {
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
  const dim3 grid(static_cast<unsigned int>((call->n + kTile - 1) / kTile),
                  static_cast<unsigned int>((call->m + kTile - 1) / kTile));
  product_kernel<<<grid, kThreads, 0, cs>>>(*call, scratch);
  if (cudaGetLastError() != cudaSuccess) {
    status = refuse(call, "the product could not be launched");
  } else {
    const int64_t threads = 256;
    const int64_t blocks = (elements + threads - 1) / threads;
    epilogue_kernel<<<static_cast<unsigned int>(blocks), static_cast<int>(threads), 0, cs>>>(
        *call, scratch);
    if (cudaGetLastError() != cudaSuccess) {
      status = refuse(call, "the product's epilogue could not be launched");
    }
  }
  cudaFreeAsync(scratch, cs);
  return status;
}
