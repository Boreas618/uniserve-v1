// The `gemm` operator's cuBLASLt backend: a native module whose entry
// `uniserve_gemm` runs one dense product through cuBLASLt (design 9, the dense
// layers' CUTLASS/cuBLASLt GEMM).
//
// The module is compiled once per descriptor: the element types, the weight
// order and the epilogue are macros the package's manifest binds to the
// descriptor's fields, so one module is one point of the operator's matrix of
// descriptors, as design 5.2 asks of a package.
//
// The product is row-major in `ops`' terms and cuBLASLt computes column-major,
// so the launch computes `out^T = alpha * weight^T * input^T` (D = A · B, D is
// [n, m]). Each row-major buffer read as column-major is its transpose with the
// row stride as the leading dimension, which is why no copy or transpose
// kernel is needed:
//
//   A = the weight buffer:  [n, k] with ld = weight_row_stride for kxn,
//                           [k, n] with ld = weight_row_stride and OP_T for nxk;
//   B = the input buffer:   [k, m] with ld = input_row_stride;
//   C = D = the out buffer: [n, m] with ld = out_row_stride.
//
// Algorithm choice, workspace and the matmul descriptor belong to the
// package's tuning data (design 5.4): each shape is asked from cuBLASLt's
// heuristic once, and its matmul is then reused for every launch of that
// shape. Serving never times
// anything; an offline tuner records its choices by publishing a new revision
// of the package, and the recorded algorithm arrives as that revision's code.
//
// cuBLASLt is loaded when the module is loaded (the operator's device work
// runs where CUDA exists); the CUDA runtime provides the workspace allocator.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <uniserve_gemm.h>

// The descriptor's compile-time choices, which the package's manifest defines
// (`[compile.definitions]`): the cuBLASLt data types of the input, the weight and
// the output, the compute type of the accumulation, the weight order (a
// `uniserve_weight_order` enumerator: `[k, n]`, or `[n, k]`, the checkpoint's own
// layout) and the epilogue (a `uniserve_epilogue` enumerator: none, or add the
// bias).
#if !defined(UNISERVE_INPUT_TYPE) || !defined(UNISERVE_WEIGHT_TYPE) ||       \
    !defined(UNISERVE_ACCUMULATE_TYPE) || !defined(UNISERVE_OUTPUT_TYPE) || \
    !defined(UNISERVE_WEIGHT_ORDER) || !defined(UNISERVE_EPILOGUE)
#error "the package's manifest defines the descriptor's compile-time choices"
#endif

namespace {

/// The largest workspace a matmul may ask for. cuBLASLt's heuristic never
/// needs more than a few tens of megabytes for these shapes; the cap keeps one
/// program's matmul workspaces from growing without bound on a host that runs
/// many.
constexpr size_t kWorkspaceCap = 64u * 1024u * 1024u;

/// Writes this entry's message into the call structure and returns the
/// failure code the operator reports.
int fail(UniserveGemmCall* call, const char* format, ...) {
  va_list arguments;
  va_start(arguments, format);
  std::vsnprintf(call->error, sizeof(call->error), format, arguments);
  va_end(arguments);
  return 1;
}

const char* cublas_status(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cuBLASLt is not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cuBLASLt allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "cuBLASLt refused an argument";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "cuBLASLt does not support this architecture";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "cuBLASLt does not support this combination";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "cuBLASLt execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "cuBLASLt internal error";
    default:
      return "cuBLASLt error";
  }
}

/// The matmul of one shape: the operation descriptor, the layouts, the
/// algorithm cuBLASLt chose for it and the workspace that algorithm needs.
struct Matmul {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  size_t workspace_bytes = 0;
  void* workspace = nullptr;

  ~Matmul() {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
    }
    if (a != nullptr) {
      cublasLtMatrixLayoutDestroy(a);
    }
    if (b != nullptr) {
      cublasLtMatrixLayoutDestroy(b);
    }
    if (c != nullptr) {
      cublasLtMatrixLayoutDestroy(c);
    }
    if (d != nullptr) {
      cublasLtMatrixLayoutDestroy(d);
    }
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

/// The key of a matmul: everything about a launch that changes cuBLASLt's
/// choice. The rows `m` is part of it because the heuristic's answer depends
/// on it; a deployment asks for a bounded set of buckets (design 4.5).
struct MatmulKey {
  int device;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t input_row_stride;
  int64_t weight_row_stride;
  int64_t out_row_stride;

  bool operator==(const MatmulKey& other) const {
    return device == other.device && m == other.m && n == other.n && k == other.k &&
           input_row_stride == other.input_row_stride &&
           weight_row_stride == other.weight_row_stride &&
           out_row_stride == other.out_row_stride;
  }
};

struct MatmulKeyHash {
  size_t operator()(const MatmulKey& key) const {
    size_t hash = static_cast<size_t>(key.device);
    const int64_t values[] = {key.m,          key.n,   key.k, key.input_row_stride,
                              key.weight_row_stride, key.out_row_stride};
    for (int64_t value : values) {
      hash ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

/// The matmul cache. A matmul is built once per shape and device and then
/// reused; `bias` is the only part that changes per launch, so it is set on
/// the operation each call.
class MatmulCache {
 public:
  /// Whether a matmul for this shape exists, without building one.
  bool contains(const MatmulKey& key) {
    std::lock_guard<std::mutex> guard(mutex_);
    return matmuls_.find(key) != matmuls_.end();
  }

  /// The matmul for one shape, built when it is first asked for.
  Matmul* of(const MatmulKey& key, cublasLtHandle_t handle, std::string* error) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = matmuls_.find(key);
    if (found != matmuls_.end()) {
      return found->second.get();
    }
    auto matmul = std::make_unique<Matmul>();
    if (!build(key, handle, matmul.get(), error)) {
      return nullptr;
    }
    Matmul* built = matmul.get();
    matmuls_[key] = std::move(matmul);
    return built;
  }

 private:
  /// Builds one matmul: the operation, the four layouts and the heuristic's
  /// algorithm with the workspace it asks for.
  static bool build(const MatmulKey& key, cublasLtHandle_t handle, Matmul* matmul,
                    std::string* error) {
    const bool kxn = UNISERVE_WEIGHT_ORDER == UNISERVE_WEIGHT_ORDER_KXN;
    // cuBLASLt's view: A is [n, k] (kxn, transposed by the layout) or [k, n]
    // (nxk, transposed by the operation), B is [k, m] and C, D are [n, m].
    const uint64_t a_rows = kxn ? static_cast<uint64_t>(key.n) : static_cast<uint64_t>(key.k);
    const uint64_t a_columns = kxn ? static_cast<uint64_t>(key.k) : static_cast<uint64_t>(key.n);
    const cublasOperation_t operation_a = kxn ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasStatus_t status = cublasLtMatmulDescCreate(&matmul->operation, UNISERVE_ACCUMULATE_TYPE,
                                                     CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("creating the matmul description: ") + cublas_status(status);
      return false;
    }
    status = cublasLtMatmulDescSetAttribute(matmul->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &operation_a, sizeof(operation_a));
    if (status == CUBLAS_STATUS_SUCCESS) {
      const cublasOperation_t operation_b = CUBLAS_OP_N;
      status = cublasLtMatmulDescSetAttribute(matmul->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                              &operation_b, sizeof(operation_b));
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("describing the transpose: ") + cublas_status(status);
      return false;
    }
    if (UNISERVE_EPILOGUE == UNISERVE_EPILOGUE_BIAS) {
      const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      const cudaDataType_t bias_type = UNISERVE_OUTPUT_TYPE;
      status = cublasLtMatmulDescSetAttribute(matmul->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                              &epilogue, sizeof(epilogue));
      if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(matmul->operation,
                                                CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                sizeof(bias_type));
      }
      if (status != CUBLAS_STATUS_SUCCESS) {
        *error = std::string("describing the bias epilogue: ") + cublas_status(status);
        return false;
      }
    }

    status = cublasLtMatrixLayoutCreate(&matmul->a, UNISERVE_WEIGHT_TYPE, a_rows, a_columns,
                                        key.weight_row_stride);
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutCreate(&matmul->b, UNISERVE_INPUT_TYPE,
                                          static_cast<uint64_t>(key.k),
                                          static_cast<uint64_t>(key.m), key.input_row_stride);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutCreate(&matmul->c, UNISERVE_OUTPUT_TYPE,
                                          static_cast<uint64_t>(key.n),
                                          static_cast<uint64_t>(key.m), key.out_row_stride);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutCreate(&matmul->d, UNISERVE_OUTPUT_TYPE,
                                          static_cast<uint64_t>(key.n),
                                          static_cast<uint64_t>(key.m), key.out_row_stride);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("creating a matrix layout: ") + cublas_status(status);
      return false;
    }

    cublasLtMatmulPreference_t preference = nullptr;
    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *error = std::string("creating the preference: ") + cublas_status(status);
      return false;
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceCap,
          sizeof(kWorkspaceCap));
    }
    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatmulAlgoGetHeuristic(handle, matmul->operation, matmul->a, matmul->b,
                                              matmul->c, matmul->d, preference, 1, &result,
                                              &returned);
    }
    cublasLtMatmulPreferenceDestroy(preference);
    if (status != CUBLAS_STATUS_SUCCESS || returned < 1) {
      *error = std::string("asking cuBLASLt for an algorithm: ") +
               (status == CUBLAS_STATUS_SUCCESS ? "no algorithm fits this shape"
                                                : cublas_status(status));
      return false;
    }
    matmul->algorithm = result.algo;
    matmul->workspace_bytes = result.workspaceSize;
    if (matmul->workspace_bytes > 0) {
      cudaError_t allocated = cudaMalloc(&matmul->workspace, matmul->workspace_bytes);
      if (allocated != cudaSuccess) {
        matmul->workspace = nullptr;
        matmul->workspace_bytes = 0;
        *error = std::string("allocating the matmul's workspace: ") + cudaGetErrorString(allocated);
        return false;
      }
    }
    return true;
  }

  std::mutex mutex_;
  std::unordered_map<MatmulKey, std::unique_ptr<Matmul>, MatmulKeyHash> matmuls_;
};

MatmulCache& matmuls() {
  static MatmulCache instance;
  return instance;
}

}  // namespace

/// One dense product. Returns 0 when the product ran, 1 with this entry's
/// message in `call->error` otherwise (design 4.2: a package reports through
/// the call structure it was given).
extern "C" int uniserve_gemm(UniserveGemmCall* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  call->error[0] = '\0';
  if (call->m <= 0 || call->n <= 0 || call->k <= 0) {
    return fail(call, "the product is [%lld, %lld] over %lld, a shape with no work",
                static_cast<long long>(call->m), static_cast<long long>(call->n),
                static_cast<long long>(call->k));
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

  MatmulKey key{device,      call->m,
                call->n,     call->k,
                call->input_row_stride, call->weight_row_stride,
                call->out_row_stride};
  // A matmul is built on its first call and that build allocates the algorithm's workspace with
  // `cudaMalloc`, which a capture cannot carry: the allocation is not part of the graph, and the
  // kernel then reads a pointer the graph never recorded. The refusal names the shape and the
  // remedy rather than letting the fault surface as an illegal address at the kernel's first read.
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream), &capture) == cudaSuccess &&
      capture != cudaStreamCaptureStatusNone && !matmuls().contains(key)) {
    return fail(call,
                "the matmul for [%lld, %lld] over %lld has not been built yet and building it "
                "allocates its workspace, which a capture cannot carry; this shape has to run "
                "once outside the bucket that captures it",
                static_cast<long long>(call->m), static_cast<long long>(call->n),
                static_cast<long long>(call->k));
  }

  std::string error;
  Matmul* matmul = matmuls().of(key, handle, &error);
  if (matmul == nullptr) {
    return fail(call, "%s", error.c_str());
  }
  if (UNISERVE_EPILOGUE == UNISERVE_EPILOGUE_BIAS) {
    status = cublasLtMatmulDescSetAttribute(matmul->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &call->bias, sizeof(call->bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return fail(call, "passing the bias to cuBLASLt: %s", cublas_status(status));
    }
  }

  const float beta = 0.0f;
  status = cublasLtMatmul(handle, matmul->operation, &call->alpha, call->weight, matmul->a,
                          call->input, matmul->b, &beta, call->out, matmul->c, call->out, matmul->d,
                          &matmul->algorithm, matmul->workspace, matmul->workspace_bytes,
                          reinterpret_cast<cudaStream_t>(stream));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return fail(call, "cuBLASLt refused [%lld, %lld] over %lld from [%lld, %lld] and [%lld, %lld]: %s",
                static_cast<long long>(call->m), static_cast<long long>(call->n),
                static_cast<long long>(call->k), static_cast<long long>(call->m),
                static_cast<long long>(call->k),
                UNISERVE_WEIGHT_ORDER == UNISERVE_WEIGHT_ORDER_KXN
                    ? static_cast<long long>(call->k)
                    : static_cast<long long>(call->n),
                UNISERVE_WEIGHT_ORDER == UNISERVE_WEIGHT_ORDER_KXN
                    ? static_cast<long long>(call->n)
                    : static_cast<long long>(call->k),
                cublas_status(status));
  }
  return 0;
}
