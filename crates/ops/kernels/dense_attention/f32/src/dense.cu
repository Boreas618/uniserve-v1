// SPDX-License-Identifier: Apache-2.0
//
// The `dense_attention` operator's `f32` backend: the FP32 attention of the audio
// encoder and the VSA compression branch (design 9, "FP32 attention: cuBLASLt
// SGEMM and a softmax kernel program"). A native module whose entry
// `uniserve_dense_attention` runs one call — packed sequences, no cache, no page
// table.
//
// One query tile is three launches for every head at once:
//
//   1. a batched cuBLASLt product computes the tile's scores
//      S = scale * Q_tile · K^T  into FP32 scratch, one matrix per head;
//   2. a kernel masks the scores the tile's rows may not read, and writes the
//      row-normalized weights  p_j = exp(s_j - max_j s_j) / sum_j p_j  back into
//      the same scratch, zeroing everything outside each row's own sequence;
//   3. a batched cuBLASLt product computes  out = P · V  straight into the caller's
//      output, through its strides.
//
// The launches are what a captured graph holds, so their count is the graph's size:
// a tile is as tall as `kScoreBytes` of scratch allows, and a head is a batch entry
// rather than a loop. A decoder's clip makes hundreds of these calls, and a schedule
// of a small tile per head was millions of nodes in one graph, which the driver
// could not launch.
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
// Two of the descriptor's fields are compiled in. The mask (`UNISERVE_CAUSAL`): a
// causal module refuses a dense call and the other way round. And the sequence
// layout (`UNISERVE_SEQUENCES`): `ops` prints `cu_seq_lens` only for the form whose
// launch carries no `[batch + 1]` offsets, and that form attends one sequence —
// `[0, q_tokens)` against `[0, kv_tokens)` — so a module built for it reads no
// table and never branches on a null pointer, and the scheduled batch form never
// sees a call without one. The history window is not compiled in: it is a bound the
// call carries (`window_left`), applied by the masking kernel, so one module serves
// a windowed and an unwindowed call. Nothing else is compiled from the descriptor —
// the heads and the head width arrive in the call and are checked here, and a
// launch's extents are the launch's, not the program's.

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
// 1: the launch carries `[batch + 1]` sequence offsets. 0: it carries none, and the
// module attends one sequence — `[0, q_tokens)` against `[0, kv_tokens)`. The two are
// separate compilations, so this module never branches on a missing table and never
// reads a null pointer.
#ifndef UNISERVE_SEQUENCES
#define UNISERVE_SEQUENCES 1
#endif

namespace {

/// The granularity of a query tile's rows. A tile is the unit the score product is
/// computed for, so its rows are also the scratch's: `rows * keys` floats per head.
constexpr int64_t kTileQ = 64;

/// The bytes one tile's scores may take, over all the heads of one product. It sets
/// how tall a tile is, and so how many launches a call makes.
constexpr int64_t kScoreBytes = 1024ll * 1024ll * 1024ll;

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
  /// The matrices one product covers, and the elements between two of them.
  int64_t batch;
  int64_t batch_stride;
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

  static bool same(const Matrix& left, const Matrix& right) {
    return left.rows == right.rows && left.columns == right.columns &&
           left.leading == right.leading && left.batch == right.batch &&
           left.batch_stride == right.batch_stride;
  }

  bool operator==(const PlanKey& other) const {
    return device == other.device && transpose_a == other.transpose_a &&
           k == other.k && same(a, other.a) && same(b, other.b) && same(d, other.d);
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
                              key.d.leading,     key.a.batch,
                              key.a.batch_stride, key.b.batch_stride,
                              key.d.batch_stride};
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
    cublasStatus_t status =
        cublasLtMatrixLayoutCreate(into, CUDA_R_32F,
                                   static_cast<uint64_t>(matrix.rows),
                                   static_cast<uint64_t>(matrix.columns),
                                   matrix.leading);
    if (status == CUBLAS_STATUS_SUCCESS) {
      const int32_t batch = static_cast<int32_t>(matrix.batch);
      status = cublasLtMatrixLayoutSetAttribute(*into, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                &batch, sizeof(batch));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutSetAttribute(
          *into, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &matrix.batch_stride,
          sizeof(matrix.batch_stride));
    }
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
/// `blockIdx.x` is the tile row and `blockIdx.y` the head; the row's own sequence
/// comes from the offsets, and every column outside that sequence's visible keys is
/// zeroed, so the following product over all of the call's keys reads nothing of
/// another sequence.
///
/// `kOffsets` is the descriptor's `cu_seq_lens`: a module compiled for the launch
/// that carries no table reads its one row range from the call's extents instead.
template <bool kOffsets>
__global__ void mask_softmax_kernel(float* tile, int64_t columns, int64_t head_stride,
                                   const int32_t* cu_q, const int32_t* cu_k,
                                   int batch, int64_t first_row,
                                   int64_t window_left, int causal) {
  __shared__ float shared[kThreads];
  const int64_t row = first_row + blockIdx.x;
  float* line = tile + static_cast<int64_t>(blockIdx.y) * head_stride +
                static_cast<int64_t>(blockIdx.x) * columns;

  // One sequence without offsets: every query row attends the whole key extent.
  int64_t first_key = 0;
  int64_t keys = columns;
  int64_t local = row;
  if constexpr (kOffsets) {
    const int sequence = sequence_of(cu_q, batch, row);
    first_key = cu_k[sequence];
    keys = static_cast<int64_t>(cu_k[sequence + 1]) - first_key;
    local = row - cu_q[sequence];
  }
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
      call->value == nullptr) {
    return fail(call, "the call leaves one of its buffers null");
  }
#if UNISERVE_SEQUENCES == 1
  if (call->cu_seq_lens_q == nullptr || call->cu_seq_lens_k == nullptr) {
    return fail(call, "this module reads the call's sequence offsets, which are null");
  }
#else
  // The no-offsets form's program has no ports for the tables (model's
  // `dense.rs::sequence_layout`): a launch that carries one is not this program's.
  if (call->cu_seq_lens_q != nullptr || call->cu_seq_lens_k != nullptr) {
    return fail(call, "this module attends one sequence and reads no offsets");
  }
  if (call->batch != 1) {
    return fail(call, "this module attends one sequence; the call names %lld",
                static_cast<long long>(call->batch));
  }
#endif
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

  // A product covers every head that reads its own keys, as one batch. Heads that share keys
  // are a batch per key head instead, whose key stride is zero.
  const int64_t group = call->q_heads / call->kv_heads;
  const int64_t heads = group == 1 ? call->q_heads : group;
  const int64_t passes = call->q_heads / heads;
  const int64_t key_batch_stride = group == 1 ? call->kv_stride_heads : 0;

  // A tile as tall as the scratch allows, and the call's rows spread evenly over the tiles.
  const int64_t row_bytes = heads * call->kv_tokens * static_cast<int64_t>(sizeof(float));
  int64_t tallest = kScoreBytes / row_bytes / kTileQ * kTileQ;
  if (tallest < kTileQ) {
    tallest = kTileQ;
  }
  const int64_t tiles = (call->q_tokens + tallest - 1) / tallest;
  const int64_t even = (call->q_tokens + tiles - 1) / tiles;
  const int64_t tile_rows = tiles == 1 ? call->q_tokens : (even + kTileQ - 1) / kTileQ * kTileQ;

  // The two products of one tile, as cuBLASLt reads them. Both compute the transpose of what
  // the call names, because a row-major buffer is its transpose read column-major and neither
  // cuBLASLt nor this module transposes anything:
  //
  //   scores:  D = S^T = [keys, rows]        A = K stored [head_dim, keys], T
  //                                          B = Q tile stored [head_dim, rows]
  //   product: D = out^T = [head_dim, rows]  A = V stored [head_dim, keys]
  //                                          B = weights stored [keys, rows]
  struct Products {
    Plan* score;
    Plan* value;
  };
  std::string error;
  auto products_of = [&](int64_t rows, Products* into) -> int32_t {
    const Matrix keys_or_values{call->head_dim, call->kv_tokens, call->kv_stride_tokens,
                                heads, key_batch_stride};
    const Matrix queries_of{call->head_dim, rows, call->q_stride_tokens, heads,
                            call->q_stride_heads};
    const Matrix weights_of{call->kv_tokens, rows, call->kv_tokens, heads,
                            rows * call->kv_tokens};
    const Matrix out_of{call->head_dim, rows, call->o_stride_tokens, heads,
                        call->o_stride_heads};
    into->score = plans().of(
        PlanKey{device, 1, call->head_dim, keys_or_values, queries_of, weights_of}, handle,
        &error);
    if (into->score == nullptr) {
      return fail(call, "the score product: %s", error.c_str());
    }
    into->value = plans().of(
        PlanKey{device, 0, call->kv_tokens, keys_or_values, weights_of, out_of}, handle,
        &error);
    if (into->value == nullptr) {
      return fail(call, "the value product: %s", error.c_str());
    }
    return 0;
  };
  const int64_t last_rows = call->q_tokens - (tiles - 1) * tile_rows;
  Products whole{};
  Products last{};
  if (const int32_t refused = products_of(tile_rows, &whole); refused != 0) {
    return refused;
  }
  if (const int32_t refused = products_of(last_rows, &last); refused != 0) {
    return refused;
  }

  // The scratch of one tile: the weights every product reads and writes, stream-ordered, so
  // tiles reused in enqueue order cannot overlap, and two calls on different streams own their
  // own. The output is the caller's: the value product writes it through its strides.
  const size_t weights_bytes = static_cast<size_t>(heads) * static_cast<size_t>(tile_rows) *
                               static_cast<size_t>(call->kv_tokens) * sizeof(float);
  void* raw = nullptr;
  cudaError_t allocated = cudaMallocAsync(&raw, weights_bytes, queue);
  if (allocated != cudaSuccess) {
    return fail(call, "allocating %zu bytes of scratch: %s", weights_bytes,
                cudaGetErrorString(allocated));
  }
  auto* scratch = static_cast<float*>(raw);
  float* weights = scratch;

  size_t workspace_bytes = 0;
  for (const Plan* plan : {whole.score, whole.value, last.score, last.value}) {
    workspace_bytes = plan->workspace_bytes > workspace_bytes ? plan->workspace_bytes
                                                              : workspace_bytes;
  }
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
  const float scale = static_cast<float>(call->scale);
  const float one = 1.0f;
  const float zero = 0.0f;

  for (int64_t at = 0; at < tiles; ++at) {
    const int64_t first_row = at * tile_rows;
    const bool final_tile = at + 1 == tiles;
    const int64_t rows = final_tile ? last_rows : tile_rows;
    const Products& products = final_tile ? last : whole;
    for (int64_t pass = 0; pass < passes; ++pass) {
      const int64_t first_head = pass * heads;
      const int64_t key_head = group == 1 ? 0 : pass;
      const float* query =
          queries + first_head * call->q_stride_heads + first_row * call->q_stride_tokens;
      const float* key = keys + key_head * call->kv_stride_heads;
      const float* value = values + key_head * call->kv_stride_heads;
      float* target =
          out + first_head * call->o_stride_heads + first_row * call->o_stride_tokens;
      const Plan* score = products.score;
      const Plan* product = products.value;

      status = cublasLtMatmul(handle, score->operation, &scale, key, score->a, query,
                              score->b, &zero, weights, score->c, weights,
                              score->d, &score->algorithm, workspace,
                              workspace_bytes, queue);
      if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(scratch, queue);
        if (workspace != nullptr) {
          cudaFreeAsync(workspace, queue);
        }
        return fail(call, "the score product of tile %lld: %s",
                    static_cast<long long>(at), cublas_status(status));
      }
      mask_softmax_kernel<UNISERVE_SEQUENCES == 1>
          <<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(heads)), kThreads, 0,
             queue>>>(weights, call->kv_tokens, rows * call->kv_tokens, call->cu_seq_lens_q,
                      call->cu_seq_lens_k, static_cast<int>(call->batch), first_row,
                      call->window_left, UNISERVE_CAUSAL);
      status = cublasLtMatmul(handle, product->operation, &one, value, product->a,
                              weights, product->b, &zero, target, product->c, target,
                              product->d, &product->algorithm, workspace,
                              workspace_bytes, queue);
      if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(scratch, queue);
        if (workspace != nullptr) {
          cudaFreeAsync(workspace, queue);
        }
        return fail(call, "the value product of tile %lld: %s",
                    static_cast<long long>(at), cublas_status(status));
      }
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
