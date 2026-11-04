// SPDX-License-Identifier: Apache-2.0
//
// The `dense_attention` operator's `f32` backend: the FP32 attention of the audio
// encoder and the VSA compression branch (design 9, "FP32 attention: cuBLASLt
// SGEMM and a softmax kernel program"). A native module whose entry
// `uniserve_dense_attention` runs one call — packed sequences, no cache, no page
// table.
//
// One query tile of `kTileQ` rows is three steps per head:
//
//   1. cuBLASLt computes the tile's scores  S = scale * Q_tile · K^T   into FP32
//      scratch (`kTileQ` rows by the call's key extent);
//   2. a kernel masks the scores the tile's rows may not read, and writes the
//      row-normalized weights  p_j = exp(s_j - max_j s_j) / sum_j p_j  back into
//      the same scratch, zeroing everything outside each row's own sequence;
//   3. cuBLASLt computes the tile's  out = P · V  into an FP32 tile, and a kernel
//      copies the tile's real rows — the rows the call's offsets name — into the
//      caller's output.
//
// The arithmetic is the operator's: every score is one FP32 product accumulated in
// FP32, the exponentials and their sum are FP32, and the only rounding after them
// is the output's own. That is the accumulation `dense_attention`'s tolerance is
// derived from (`crates/ops/src/attention/dense.rs`, Higham's convex-combination
// bound over the descriptor's `max_kv_len`).
//
// # Why the tile's key extent is the call's packed extent
//
// A tile's rows may belong to any sequence of the call, and this module reads the
// two offset tables on the device only: it never copies them to the host, so it
// never synchronizes the caller's stream and a captured call is captured as one
// graph. A per-sequence schedule would need each sequence's row count on the host
// (`cudaMemcpyAsync` plus a stream synchronize), which is what that constraint
// forbids. The cost of the choice is bounded and is paid only by calls that pack
// several sequences: the tile's product reads every key of the call rather than
// its own sequence's keys, so a call packing `b` sequences of equal length does up
// to `b` times the keys a per-sequence schedule would. A call that packs one
// sequence — the audio encoder's and the compression branch's usual shape — reads
// exactly its own keys. The masked weights are exactly zero outside each row's
// sequence, so the extra columns change no result.
//
// # The descriptor
//
// The mask is compiled in (`UNISERVE_CAUSAL`, from the descriptor's `mask` field):
// a causal module refuses a dense call and the other way round. The history window
// is not compiled in — it is a bound the call carries (`window_left`), applied by
// the masking kernel — so one module serves a windowed and an unwindowed call of
// the same descriptor. Everything else the descriptor names (heads, head width,
// the length cap) arrives in the call and is checked here.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "uniserve_dense_attention.h"

// The export name and the mask, defined by `src/build.py` from the package's
// manifest and the descriptor. The defaults keep this file readable by itself and
// are never what a module the Program Server builds carries.
#ifndef UNISERVE_ENTRY
#define UNISERVE_ENTRY uniserve_dense_attention
#endif
// 0: the dense mask, every key of the sequence visible. 1: the causal mask.
#ifndef UNISERVE_CAUSAL
#define UNISERVE_CAUSAL 0
#endif

namespace {

/// Query rows one work item covers. A tile is the unit the score product is
/// computed for, so it is also the scratch's row count: `kTileQ * keys` floats.
constexpr int64_t kTileQ = 64;

/// Threads of the masking and copy kernels.
constexpr int kThreads = 256;

/// `DType::as_i64()`'s number for the element type this backend serves.
constexpr int64_t kFloat32 = 5;

/// The largest workspace one cuBLASLt plan may ask for; a cap keeps a host that
/// serves many shapes from holding unbounded plan scratch.
constexpr size_t kWorkspaceCap = 64u * 1024u * 1024u;

/// Writes this entry's message into the call structure and returns the failure
/// code the operator reports.
int32_t fail(UniserveDenseAttentionCall* call, const char* format, ...) {
  va_list arguments;
  va_start(arguments, format);
  std::vsnprintf(reinterpret_cast<char*>(call->error), sizeof(call->error), format,
                 arguments);
  va_end(arguments);
  return 1;
}

const char* cublas_status(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cuBLAS is not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cuBLAS allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "cuBLAS refused an argument";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "cuBLAS does not support this architecture";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "cuBLAS does not support this combination";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "cuBLAS execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "cuBLAS internal error";
    default:
      return "cuBLAS error";
  }
}

/// One matrix as cuBLASLt reads it: stored column-major, `rows` by `columns`,
/// with `leading` elements between consecutive columns.
///
/// A row-major buffer read as column-major is its transpose with the row stride as
/// the leading dimension, which is why no copy or transpose kernel is needed: the
/// query, key, value and output buffers keep their caller's strides all the way
/// into cuBLASLt.
struct Matrix {
  int64_t rows;
  int64_t columns;
  int64_t leading;
};

/// The plan of one product: the layouts, the algorithm cuBLASLt chose for them and
/// the workspace that algorithm asks for.
struct Plan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  size_t workspace_bytes = 0;

  ~Plan() {
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
    }
    for (cublasLtMatrixLayout_t* layout : {&a, &b, &c, &d}) {
      if (*layout != nullptr) {
        cublasLtMatrixLayoutDestroy(*layout);
      }
    }
  }
};

/// The key of a plan: everything about a product that changes cuBLASLt's choice.
/// The rows are part of it because the heuristic's answer depends on them; a
/// deployment asks for a bounded set of (tile, key extent, head width) triples.
struct PlanKey {
  int device;
  int transpose_a;
  int64_t k;
  Matrix a;
  Matrix b;
  Matrix d;

  bool operator==(const PlanKey& other) const {
    return device == other.device && transpose_a == other.transpose_a &&
           k == other.k && a.rows == other.a.rows && a.columns == other.a.columns &&
           a.leading == other.a.leading && b.rows == other.b.rows &&
           b.columns == other.b.columns && b.leading == other.b.leading &&
           d.rows == other.d.rows && d.columns == other.d.columns &&
           d.leading == other.d.leading;
  }
};

struct PlanKeyHash {
  size_t operator()(const PlanKey& key) const {
    size_t hash = static_cast<size_t>(key.device);
    const int64_t values[] = {key.transpose_a,   key.k,
                              key.a.rows,        key.a.columns,
                              key.a.leading,     key.b.rows,
                              key.b.columns,     key.b.leading,
                              key.d.rows,        key.d.columns,
                              key.d.leading};
    for (int64_t value : values) {
      hash ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ull + (hash << 6) +
              (hash >> 2);
    }
    return hash;
  }
};

/// One cuBLASLt handle per device, created when the device is first used.
class Handles {
 public:
  cublasLtHandle_t of(int device, cublasStatus_t* status) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = handles_.find(device);
    if (found != handles_.end()) {
      *status = CUBLAS_STATUS_SUCCESS;
      return found->second;
    }
    cublasLtHandle_t handle = nullptr;
    *status = cublasLtCreate(&handle);
    if (*status == CUBLAS_STATUS_SUCCESS) {
      handles_[device] = handle;
    }
    return handle;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<int, cublasLtHandle_t> handles_;
};

Handles& handles() {
  static Handles instance;
  return instance;
}

/// The plan cache: one plan per shape and device, reused by every launch of it.
class Plans {
 public:
  Plan* of(const PlanKey& key, cublasLtHandle_t handle, std::string* error) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = plans_.find(key);
    if (found != plans_.end()) {
      return found->second.get();
    }
    auto plan = std::make_unique<Plan>();
    if (!build(key, handle, plan.get(), error)) {
      return nullptr;
    }
    Plan* built = plan.get();
    plans_[key] = std::move(plan);
    return built;
  }

 private:
  static bool layout(cublasLtMatrixLayout_t* into, const Matrix& matrix,
                     std::string* error) {
    const cublasStatus_t status =
        cublasLtMatrixLayoutCreate(into, CUDA_R_32F,
                                   static_cast<uint64_t>(matrix.rows),
                                   static_cast<uint64_t>(matrix.columns),
                                   matrix.leading);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("describing a matrix: ") + cublas_status(status);
      return false;
    }
    return true;
  }

  static bool build(const PlanKey& key, cublasLtHandle_t handle, Plan* plan,
                    std::string* error) {
    cublasStatus_t status = cublasLtMatmulDescCreate(
        &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("creating the matmul description: ") +
               cublas_status(status);
      return false;
    }
    const cublasOperation_t operation_a =
        key.transpose_a == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_b = CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(plan->operation,
                                            CUBLASLT_MATMUL_DESC_TRANSA,
                                            &operation_a, sizeof(operation_a));
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatmulDescSetAttribute(plan->operation,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &operation_b, sizeof(operation_b));
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("describing the transpose: ") + cublas_status(status);
      return false;
    }
    if (!layout(&plan->a, key.a, error) || !layout(&plan->b, key.b, error) ||
        !layout(&plan->c, key.d, error) || !layout(&plan->d, key.d, error)) {
      return false;
    }

    cublasLtMatmulPreference_t preference = nullptr;
    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("creating the preference: ") + cublas_status(status);
      return false;
    }
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceCap,
        sizeof(kWorkspaceCap));
    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatmulAlgoGetHeuristic(handle, plan->operation, plan->a,
                                              plan->b, plan->c, plan->d,
                                              preference, 1, &result, &returned);
    }
    cublasLtMatmulPreferenceDestroy(preference);
    if (status != CUBLAS_STATUS_SUCCESS || returned < 1) {
      *error = std::string("asking cuBLASLt for an algorithm: ") +
               (status == CUBLAS_STATUS_SUCCESS ? "no algorithm fits this shape"
                                                : cublas_status(status));
      return false;
    }
    plan->algorithm = result.algo;
    plan->workspace_bytes = result.workspaceSize;
    return true;
  }

  std::mutex mutex_;
  std::unordered_map<PlanKey, std::unique_ptr<Plan>, PlanKeyHash> plans_;
};

Plans& plans() {
  static Plans instance;
  return instance;
}

/// The sequence a packed row belongs to: the greatest `b` with
/// `cu_seq_lens_q[b] <= row`, which the caller's validation has shown to be a real
/// sequence.
__device__ int sequence_of(const int32_t* offsets, int batch, int64_t row) {
  int low = 0;
  int high = batch - 1;
  while (low < high) {
    const int middle = (low + high + 1) / 2;
    if (static_cast<int64_t>(offsets[middle]) <= row) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return low;
}

/// The reduction of one row's maximum or sum over the block's threads.
__device__ float reduce(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  return shared[0];
}

/// The reduction of one row's sum over the block's threads.
__device__ float reduce_sum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return shared[0];
}

/// Masks one tile's scores and turns them into the row-normalized weights.
///
/// `blockIdx.x` is the tile row; the row's own sequence comes from the offsets, and
/// every column outside that sequence's visible keys is zeroed, so the following
/// product over all of the call's keys reads nothing of another sequence. A tile
/// row past the packed token count is no row of the tensor: it is zeroed and its
/// output is never written.
__global__ void mask_softmax_kernel(float* tile, int64_t columns,
                                   const int32_t* cu_q, const int32_t* cu_k,
                                   int batch, int64_t first_row, int64_t q_tokens,
                                   int64_t window_left, int causal) {
  __shared__ float shared[kThreads];
  const int64_t row = first_row + blockIdx.x;
  float* line = tile + static_cast<int64_t>(blockIdx.x) * columns;
  if (row >= q_tokens || columns <= 0) {
    for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
      line[column] = 0.0f;
    }
    return;
  }

  const int sequence = sequence_of(cu_q, batch, row);
  const int64_t first_key = cu_k[sequence];
  const int64_t keys = static_cast<int64_t>(cu_k[sequence + 1]) - first_key;
  const int64_t local = row - cu_q[sequence];
  // The visible keys of the row, as offsets into its sequence's keys: the causal
  // mask ends at the row's own position, and a history window begins the same
  // distance to the left of it.
  int64_t last = causal != 0 ? (local < keys ? local : keys - 1) : keys - 1;
  int64_t first = 0;
  if (causal != 0 && window_left >= 0) {
    first = local - window_left;
    if (first < 0) {
      first = 0;
    }
  }
  const bool visible = first <= last && keys > 0;
  const int64_t begin = first_key + first;
  const int64_t end = first_key + last + 1;

  float largest = -INFINITY;
  if (visible) {
    for (int64_t column = begin + threadIdx.x; column < end; column += blockDim.x) {
      largest = fmaxf(largest, line[column]);
    }
  }
  largest = reduce(largest, shared);

  // The exponentials are written over the scores, so the third pass rescales them
  // in place. A row with nothing visible contributes nothing: its weights are all
  // zero and its output row is zero, which is what an unused layout row is.
  float total = 0.0f;
  if (visible) {
    for (int64_t column = begin + threadIdx.x; column < end; column += blockDim.x) {
      const float weight = expf(line[column] - largest);
      line[column] = weight;
      total += weight;
    }
  }
  total = reduce_sum(total, shared);
  const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const bool counted = column >= begin && column < end;
    line[column] = counted ? line[column] * inverse : 0.0f;
  }
}

/// Copies one tile's real rows of one head into the caller's output.
///
/// The tile's rows are `first_row` onwards; a row past the packed token count
/// belongs to no sequence and is not written, so the tile's padding cannot reach
/// the rows of the sequence that follows it.
__global__ void copy_rows_kernel(const float* tile, float* out,
                                 int64_t o_stride_tokens, int64_t first_row,
                                 int64_t q_tokens, int64_t head_dim) {
  const int64_t row = first_row + blockIdx.x;
  if (row >= q_tokens) {
    return;
  }
  const float* source = tile + static_cast<int64_t>(blockIdx.x) * head_dim;
  float* target = out + row * o_stride_tokens;
  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    target[column] = source[column];
  }
}

}  // namespace

/// One dense attention call. Returns 0 when the attention ran, and 1 with this
/// entry's message in `call->error` otherwise (design 4.2: a package reports
/// through the call structure it was given).
extern "C" int32_t UNISERVE_ENTRY(UniserveDenseAttentionCall* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  call->error[0] = '\0';
  const cudaStream_t queue = reinterpret_cast<cudaStream_t>(stream);

  if (call->out == nullptr || call->query == nullptr || call->key == nullptr ||
      call->value == nullptr || call->cu_seq_lens_q == nullptr ||
      call->cu_seq_lens_k == nullptr) {
    return fail(call, "the call leaves one of its buffers null");
  }
  if (call->dtype_query != kFloat32 || call->dtype_key_value != kFloat32 ||
      call->dtype_output != kFloat32) {
    return fail(call,
                "this backend serves FP32 activations only; the call holds "
                "dtypes %lld, %lld and %lld",
                static_cast<long long>(call->dtype_query),
                static_cast<long long>(call->dtype_key_value),
                static_cast<long long>(call->dtype_output));
  }
  if (call->causal != UNISERVE_CAUSAL) {
    return fail(call,
                "this module compiles the %s mask; the call launches causal = %lld",
                UNISERVE_CAUSAL == 1 ? "causal" : "dense",
                static_cast<long long>(call->causal));
  }
  if (call->batch <= 0 || call->q_tokens <= 0 || call->kv_tokens <= 0) {
    return fail(call,
                "a call attends from %lld query rows to %lld key rows in %lld "
                "sequences",
                static_cast<long long>(call->q_tokens),
                static_cast<long long>(call->kv_tokens),
                static_cast<long long>(call->batch));
  }
  if (call->head_dim <= 0 || call->q_heads <= 0 || call->kv_heads <= 0 ||
      call->q_heads % call->kv_heads != 0) {
    return fail(call,
                "%lld query heads and %lld key heads of %lld elements do not "
                "divide",
                static_cast<long long>(call->q_heads),
                static_cast<long long>(call->kv_heads),
                static_cast<long long>(call->head_dim));
  }
  if (call->q_stride_tokens < call->head_dim ||
      call->kv_stride_tokens < call->head_dim) {
    return fail(call,
                "the query and key row strides (%lld and %lld) are shorter than "
                "one head of %lld elements",
                static_cast<long long>(call->q_stride_tokens),
                static_cast<long long>(call->kv_stride_tokens),
                static_cast<long long>(call->head_dim));
  }
  if (call->q_stride_heads < 0 || call->kv_stride_heads < 0 ||
      call->o_stride_heads < 0 || call->o_stride_tokens < call->head_dim) {
    return fail(call,
                "the call's head strides (%lld, %lld, %lld) and output row stride "
                "(%lld) do not address its heads of %lld elements",
                static_cast<long long>(call->q_stride_heads),
                static_cast<long long>(call->kv_stride_heads),
                static_cast<long long>(call->o_stride_heads),
                static_cast<long long>(call->o_stride_tokens),
                static_cast<long long>(call->head_dim));
  }
  if (call->kv_tokens > call->max_kv_len) {
    return fail(call,
                "the module serves %lld keys, the call packs %lld",
                static_cast<long long>(call->max_kv_len),
                static_cast<long long>(call->kv_tokens));
  }
  if (!(call->scale > 0.0) || !std::isfinite(call->scale)) {
    return fail(call, "expected a finite nonzero scale, found %f", call->scale);
  }
  // A history window with the dense mask is not expressible; the operator refuses
  // it, and a module that reached the device with it would read every key anyway.
  if (call->window_left >= 0 && call->causal == 0) {
    return fail(call, "a history window of %lld keys needs the causal mask",
                static_cast<long long>(call->window_left));
  }

  int device = 0;
  cudaError_t current = cudaGetDevice(&device);
  if (current != cudaSuccess) {
    return fail(call, "reading the current device: %s", cudaGetErrorString(current));
  }
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  cublasLtHandle_t handle = handles().of(device, &status);
  if (handle == nullptr || status != CUBLAS_STATUS_SUCCESS) {
    return fail(call, "creating the cuBLASLt handle: %s", cublas_status(status));
  }

  // The two products of one work item, as cuBLASLt reads them. Both compute the
  // transpose of what the call names, because a row-major buffer is its transpose
  // read column-major and neither cuBLASLt nor this module transposes anything:
  //
  //   scores:  D = S^T = [keys, tile]      A = K stored [head_dim, keys], T
  //                                        B = Q tile stored [head_dim, tile]
  //   product: D = out^T = [head_dim, tile] A = V stored [head_dim, keys]
  //                                        B = weights stored [keys, tile]
  const Matrix score_a{call->head_dim, call->kv_tokens, call->kv_stride_tokens};
  const Matrix score_b{call->head_dim, kTileQ, call->q_stride_tokens};
  const Matrix score_d{call->kv_tokens, kTileQ, call->kv_tokens};
  const Matrix product_a{call->head_dim, call->kv_tokens, call->kv_stride_tokens};
  const Matrix product_b{call->kv_tokens, kTileQ, call->kv_tokens};
  const Matrix product_d{call->head_dim, kTileQ, call->head_dim};

  std::string error;
  Plan* score = plans().of(PlanKey{device, 1, call->head_dim, score_a, score_b, score_d},
                           handle, &error);
  if (score == nullptr) {
    return fail(call, "the score product: %s", error.c_str());
  }
  Plan* product =
      plans().of(PlanKey{device, 0, call->kv_tokens, product_a, product_b, product_d},
                 handle, &error);
  if (product == nullptr) {
    return fail(call, "the value product: %s", error.c_str());
  }

  // The scratch of one work item: the weights tile every product reads and writes,
  // and the tile the output product lands in before its real rows are copied out.
  // Both are stream-ordered, so tiles reused in enqueue order cannot overlap, and
  // two calls on different streams own their own.
  const size_t weights_bytes =
      static_cast<size_t>(kTileQ) * static_cast<size_t>(call->kv_tokens) * sizeof(float);
  const size_t output_bytes =
      static_cast<size_t>(kTileQ) * static_cast<size_t>(call->head_dim) * sizeof(float);
  void* raw = nullptr;
  cudaError_t allocated =
      cudaMallocAsync(&raw, weights_bytes + output_bytes, queue);
  if (allocated != cudaSuccess) {
    return fail(call, "allocating %zu bytes of scratch: %s",
                weights_bytes + output_bytes, cudaGetErrorString(allocated));
  }
  auto* scratch = static_cast<float*>(raw);
  float* weights = scratch;
  float* tile = scratch + kTileQ * call->kv_tokens;

  const size_t workspace_bytes =
      score->workspace_bytes > product->workspace_bytes ? score->workspace_bytes
                                                        : product->workspace_bytes;
  void* workspace = nullptr;
  if (workspace_bytes > 0) {
    allocated = cudaMallocAsync(&workspace, workspace_bytes, queue);
    if (allocated != cudaSuccess) {
      cudaFreeAsync(scratch, queue);
      return fail(call, "allocating %zu bytes of cuBLASLt workspace: %s",
                  workspace_bytes, cudaGetErrorString(allocated));
    }
  }

  const auto* queries = static_cast<const float*>(call->query);
  const auto* keys = static_cast<const float*>(call->key);
  const auto* values = static_cast<const float*>(call->value);
  auto* out = static_cast<float*>(call->out);
  const int64_t group = call->q_heads / call->kv_heads;
  const float scale = static_cast<float>(call->scale);
  const float one = 1.0f;
  const float zero = 0.0f;
  const int64_t tiles = (call->q_tokens + kTileQ - 1) / kTileQ;

  for (int64_t at = 0; at < tiles; ++at) {
    const int64_t first_row = at * kTileQ;
    for (int64_t head = 0; head < call->q_heads; ++head) {
      const int64_t group_head = head / group;
      const float* query =
          queries + head * call->q_stride_heads + first_row * call->q_stride_tokens;
      const float* key = keys + group_head * call->kv_stride_heads;
      const float* value = values + group_head * call->kv_stride_heads;
      float* target = out + head * call->o_stride_heads;

      status = cublasLtMatmul(handle, score->operation, &scale, key, score->a, query,
                              score->b, &zero, weights, score->c, weights,
                              score->d, &score->algorithm, workspace,
                              workspace_bytes, queue);
      if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(scratch, queue);
        if (workspace != nullptr) {
          cudaFreeAsync(workspace, queue);
        }
        return fail(call, "the score product of tile %lld, head %lld: %s",
                    static_cast<long long>(at), static_cast<long long>(head),
                    cublas_status(status));
      }
      mask_softmax_kernel<<<kTileQ, kThreads, 0, queue>>>(
          weights, call->kv_tokens, call->cu_seq_lens_q, call->cu_seq_lens_k,
          static_cast<int>(call->batch), first_row, call->q_tokens,
          call->window_left, UNISERVE_CAUSAL);
      status = cublasLtMatmul(handle, product->operation, &one, value, product->a,
                              weights, product->b, &zero, tile, product->c, tile,
                              product->d, &product->algorithm, workspace,
                              workspace_bytes, queue);
      if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(scratch, queue);
        if (workspace != nullptr) {
          cudaFreeAsync(workspace, queue);
        }
        return fail(call, "the value product of tile %lld, head %lld: %s",
                    static_cast<long long>(at), static_cast<long long>(head),
                    cublas_status(status));
      }
      copy_rows_kernel<<<kTileQ, kThreads, 0, queue>>>(
          tile, target, call->o_stride_tokens, first_row, call->q_tokens,
          call->head_dim);
    }
  }

  cudaFreeAsync(scratch, queue);
  if (workspace != nullptr) {
    cudaFreeAsync(workspace, queue);
  }
  current = cudaGetLastError();
  if (current != cudaSuccess) {
    return fail(call, "the attention's kernels: %s", cudaGetErrorString(current));
  }
  return 0;
}
