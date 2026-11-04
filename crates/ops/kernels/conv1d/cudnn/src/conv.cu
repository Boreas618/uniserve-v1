// SPDX-License-Identifier: Apache-2.0
//
// A convolution module: cuDNN 9's backend API, asked once per shape for an
// engine and then executed on the caller's stream.
//
// The engine is the heuristic's first proposal for the shape, which is stable
// for the cuDNN version a package's toolchain pins; a package that carries
// tuning data for its shape compiles the chosen engine's global index in as
// `UNISERVE_CONV_ENGINE`, and the module then takes that engine among the
// proposals and fails naming the index when this cuDNN proposes none — design
// section 5.4's tuning rule: never measure in service.
//
// The plan is built once per shape and device and reused, so service calls
// `cudnnBackendExecute` and nothing else. The workspace comes from the stream's
// pool, so a captured graph records it like any other work.
//
// `ops::conv::Call` carries three spatial axes for all three operators, so this
// module always describes three spatial dimensions: a 1-d convolution passes
// `[l, 1, 1]`, whose filters and strides are trivial along the other two, and
// cuDNN's arithmetic with them is the 1-d convolution's.
//
// A transposed convolution is cuDNN's backward-data pass of the convolution the
// layer transposes: `dx` is the layer's output, `dy` its input, and the filter is
// the layer's weight read with the output channels first, which the filter
// descriptor's strides express without copying the weight (the layer's weight is
// `[c, k, …]` in memory and `[k, c, …]` to cuDNN). The layer's output padding is
// zero in `ops`, so both paddings are the layer's own. A grouped transposed
// convolution is not described by that pass in this layout and is refused.
//
// The element types are the descriptor's, known when the descriptor's module is
// compiled: `UNISERVE_CONV_INPUT_DTYPE`, `UNISERVE_CONV_WEIGHT_DTYPE` and
// `UNISERVE_CONV_OUTPUT_DTYPE` carry the operator crate's `DType` numbers. cuDNN
// computes every convolution here in FP32, which keeps the video VAE's precision
// and lets the engine use its tensor cores for it.

#include "../include/uniserve_conv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#ifndef UNISERVE_CONV_ENTRY
#error "the package's build script defines UNISERVE_CONV_ENTRY: the module's launch entry"
#endif
#ifndef UNISERVE_CONV_INPUT_DTYPE
#error "the package's build script defines UNISERVE_CONV_INPUT_DTYPE: the descriptor's input type"
#endif
#ifndef UNISERVE_CONV_WEIGHT_DTYPE
#error "the package's build script defines UNISERVE_CONV_WEIGHT_DTYPE: the descriptor's weight type"
#endif
#ifndef UNISERVE_CONV_OUTPUT_DTYPE
#error "the package's build script defines UNISERVE_CONV_OUTPUT_DTYPE: the descriptor's output type"
#endif
#ifndef UNISERVE_CONV_ENGINE
#error "the package's build script defines UNISERVE_CONV_ENGINE: -1, or the tuned engine's index"
#endif

namespace {

/// The engine choice of a module built without tuning data: the heuristic's
/// first proposal.
constexpr int64_t kFirstEngine = -1;

// cuDNN 9's backend API, loaded at run time: the module carries no cuDNN headers
// and no link-time dependency, so it runs against the library the host ships.
constexpr const char* kLibrary = "libcudnn.so.9";
using Status = int32_t;
constexpr Status kSuccess = 0;

// `cudnnBackendDescriptorType_t`.
constexpr int32_t kDescriptorConvolution = 1;
constexpr int32_t kDescriptorEngineConfig = 3;
constexpr int32_t kDescriptorEngineHeuristic = 4;
constexpr int32_t kDescriptorExecutionPlan = 5;
constexpr int32_t kDescriptorOperationForward = 10;
constexpr int32_t kDescriptorOperationBackwardData = 12;
constexpr int32_t kDescriptorOperationGraph = 15;
constexpr int32_t kDescriptorVariantPack = 16;
constexpr int32_t kDescriptorTensor = 17;

// `cudnnBackendAttributeName_t`.
constexpr int32_t kAttrConvolutionCompType = 100;
constexpr int32_t kAttrConvolutionConvMode = 101;
constexpr int32_t kAttrConvolutionDilations = 102;
constexpr int32_t kAttrConvolutionFilterStrides = 103;
constexpr int32_t kAttrConvolutionPostPaddings = 104;
constexpr int32_t kAttrConvolutionPrePaddings = 105;
constexpr int32_t kAttrConvolutionSpatialDims = 106;
constexpr int32_t kAttrEngineHeurMode = 200;
constexpr int32_t kAttrEngineHeurOperationGraph = 201;
constexpr int32_t kAttrEngineHeurResults = 202;
constexpr int32_t kAttrEngineCfgEngine = 300;
constexpr int32_t kAttrExecutionPlanEngineConfig = 401;
constexpr int32_t kAttrExecutionPlanWorkspaceSize = 402;
constexpr int32_t kAttrOperationForwardAlpha = 700;
constexpr int32_t kAttrOperationForwardBeta = 701;
constexpr int32_t kAttrOperationForwardConvDesc = 702;
constexpr int32_t kAttrOperationForwardW = 703;
constexpr int32_t kAttrOperationForwardX = 704;
constexpr int32_t kAttrOperationForwardY = 705;
constexpr int32_t kAttrOperationBwdDataAlpha = 706;
constexpr int32_t kAttrOperationBwdDataBeta = 707;
constexpr int32_t kAttrOperationBwdDataConvDesc = 708;
constexpr int32_t kAttrOperationBwdDataW = 709;
constexpr int32_t kAttrOperationBwdDataDx = 710;
constexpr int32_t kAttrOperationBwdDataDy = 711;
constexpr int32_t kAttrOperationGraphOps = 801;
constexpr int32_t kAttrTensorDataType = 901;
constexpr int32_t kAttrTensorDimensions = 902;
constexpr int32_t kAttrTensorStrides = 903;
constexpr int32_t kAttrTensorUniqueId = 906;
constexpr int32_t kAttrVariantPackUniqueIds = 1000;
constexpr int32_t kAttrVariantPackDataPointers = 1001;
constexpr int32_t kAttrVariantPackWorkspace = 1003;
constexpr int32_t kAttrEngineGlobalIndex = 1301;

// `cudnnBackendAttributeType_t`.
constexpr int32_t kTypeDataType = 1;
constexpr int32_t kTypeInt64 = 3;
constexpr int32_t kTypeFloat = 4;
constexpr int32_t kTypeVoidPtr = 6;
constexpr int32_t kTypeConvolutionMode = 7;
constexpr int32_t kTypeHeurMode = 8;
constexpr int32_t kTypeBackendDescriptor = 15;

// `CUDNN_DATA_FLOAT`: the compute type of every convolution here.
constexpr int32_t kComputeFloat = 0;
// `CUDNN_HEUR_MODE_A`: cuDNN's own heuristic, which measures nothing.
constexpr int32_t kHeuristicModeA = 3;
// `CUDNN_CROSS_CORRELATION`: the convolution PyTorch's `Conv*d` computes.
constexpr int32_t kCrossCorrelation = 1;
// The number of engine configurations one heuristic query returns.
constexpr int64_t kMaxEngineConfigs = 32;
// The unique ids tying the operation graph's tensors to a variant pack.
constexpr int64_t kUidX = 0;
constexpr int64_t kUidW = 1;
constexpr int64_t kUidY = 2;
// The spatial axes `ops`'s call structure carries for every operator.
constexpr int32_t kSpatial = 3;

/// The cuDNN entry points this module calls.
struct Api {
  bool loaded = false;
  const char* why = "libcudnn.so.9 is not on this host";
  Status (*create)(void**) = nullptr;
  Status (*set_stream)(void*, cudaStream_t) = nullptr;
  Status (*backend_create)(int32_t, void**) = nullptr;
  Status (*backend_destroy)(void*) = nullptr;
  Status (*backend_set)(void*, int32_t, int32_t, int64_t, const void*) = nullptr;
  Status (*backend_get)(const void*, int32_t, int32_t, int64_t, int64_t*, void*) = nullptr;
  Status (*backend_finalize)(void*) = nullptr;
  Status (*backend_execute)(void*, void*, void*) = nullptr;
};

/// The host's cuDNN, loaded the first time a module of this process launches.
Api& api() {
  static Api loaded = [] {
    Api found;
    void* library = dlopen(kLibrary, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      return found;
    }
    found.create = reinterpret_cast<Status (*)(void**)>(dlsym(library, "cudnnCreate"));
    found.set_stream =
        reinterpret_cast<Status (*)(void*, cudaStream_t)>(dlsym(library, "cudnnSetStream"));
    found.backend_create =
        reinterpret_cast<Status (*)(int32_t, void**)>(dlsym(library, "cudnnBackendCreateDescriptor"));
    found.backend_destroy =
        reinterpret_cast<Status (*)(void*)>(dlsym(library, "cudnnBackendDestroyDescriptor"));
    found.backend_set =
        reinterpret_cast<Status (*)(void*, int32_t, int32_t, int64_t, const void*)>(
            dlsym(library, "cudnnBackendSetAttribute"));
    found.backend_get =
        reinterpret_cast<Status (*)(const void*, int32_t, int32_t, int64_t, int64_t*, void*)>(
            dlsym(library, "cudnnBackendGetAttribute"));
    found.backend_finalize =
        reinterpret_cast<Status (*)(void*)>(dlsym(library, "cudnnBackendFinalize"));
    found.backend_execute =
        reinterpret_cast<Status (*)(void*, void*, void*)>(dlsym(library, "cudnnBackendExecute"));
    found.loaded = found.create != nullptr && found.set_stream != nullptr &&
                   found.backend_create != nullptr && found.backend_destroy != nullptr &&
                   found.backend_set != nullptr && found.backend_get != nullptr &&
                   found.backend_finalize != nullptr && found.backend_execute != nullptr;
    if (!found.loaded) {
      found.why = "libcudnn.so.9 does not export the backend API this module calls";
    }
    return found;
  }();
  return loaded;
}

/// Writes `message` into an entry's error buffer and answers its status.
int32_t refuse(char* error, const char* message) {
  std::snprintf(error, 512, "%s", message);
  return 1;
}

/// The cuDNN element type of an operator element type, or -1.
int32_t cudnn_dtype(int64_t dtype) {
  switch (dtype) {
    case UNISERVE_CONV_DTYPE_F32:
      return 0;
    case UNISERVE_CONV_DTYPE_F16:
      return 2;
    case UNISERVE_CONV_DTYPE_BF16:
      return 9;
    default:
      return -1;
  }
}

/// One device's cuDNN handle, created once per process: the handle holds no
/// per-launch state and is not cheap to make.
void* handle_of(int32_t device, char* error, int32_t* status) {
  static std::mutex mutex;
  static std::map<int32_t, void*> handles;
  const std::lock_guard<std::mutex> guard(mutex);
  if (const auto found = handles.find(device); found != handles.end()) {
    return found->second;
  }
  const Api& library = api();
  if (!library.loaded) {
    *status = refuse(error, library.why);
    return nullptr;
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    *status = refuse(error, "this convolution's device is not available to the module");
    return nullptr;
  }
  void* handle = nullptr;
  if (library.create(&handle) != kSuccess) {
    *status = refuse(error, "cudnnCreate failed for this convolution's device");
    return nullptr;
  }
  handles[device] = handle;
  *status = 0;
  return handle;
}

/// One shape's execution plan and the workspace it asks for.
struct Plan {
  void* plan = nullptr;
  size_t workspace = 0;
};

/// The plan of a shape, from the process's cache or from cuDNN's heuristic.
Plan plan_of(const uniserve_conv_call& call, const std::string& key, char* error,
             int32_t* status) {
  static std::mutex mutex;
  static std::map<std::string, Plan> plans;
  const std::lock_guard<std::mutex> guard(mutex);
  if (const auto found = plans.find(key); found != plans.end()) {
    return found->second;
  }
  const Api& library = api();

  auto make = [&library, error, status](int32_t kind, void** descriptor) {
    if (library.backend_create(kind, descriptor) != kSuccess) {
      *status = refuse(error, "cudnnBackendCreateDescriptor failed");
      return false;
    }
    return true;
  };
  auto set = [&library, error, status](void* descriptor, int32_t attribute, int32_t type,
                                       int64_t count, const void* values) {
    if (library.backend_set(descriptor, attribute, type, count, values) != kSuccess) {
      *status = refuse(error, "cudnnBackendSetAttribute failed");
      return false;
    }
    return true;
  };
  auto finalize = [&library, error, status](void* descriptor) {
    if (library.backend_finalize(descriptor) != kSuccess) {
      *status = refuse(error, "cudnnBackendFinalize failed");
      return false;
    }
    return true;
  };
  auto tensor = [&](const int64_t* dims, const int64_t* strides, int32_t dtype, int64_t uid,
                    void** descriptor) {
    return make(kDescriptorTensor, descriptor) &&
           set(*descriptor, kAttrTensorDataType, kTypeDataType, 1, &dtype) &&
           set(*descriptor, kAttrTensorDimensions, kTypeInt64, kSpatial + 2, dims) &&
           set(*descriptor, kAttrTensorStrides, kTypeInt64, kSpatial + 2, strides) &&
           set(*descriptor, kAttrTensorUniqueId, kTypeInt64, 1, &uid) &&
           finalize(*descriptor);
  };
  /// Row-major strides of a contiguous tensor, the last axis running fastest.
  auto row_major = [](const int64_t* dims, int64_t* strides) {
    int64_t running = 1;
    for (int32_t axis = kSpatial + 1; axis >= 0; --axis) {
      strides[axis] = running;
      running *= dims[axis];
    }
  };

  const bool transposed = call.transposed != 0;
  const int32_t x_type = cudnn_dtype(UNISERVE_CONV_INPUT_DTYPE);
  const int32_t w_type = cudnn_dtype(UNISERVE_CONV_WEIGHT_DTYPE);
  const int32_t y_type = cudnn_dtype(UNISERVE_CONV_OUTPUT_DTYPE);

  int64_t x_dims[5];
  int64_t x_strides[5];
  int64_t w_dims[5];
  int64_t w_strides[5];
  int64_t y_dims[5];
  int64_t y_strides[5];
  std::memcpy(x_dims, call.input_shape, sizeof(x_dims));
  std::memcpy(w_dims, call.weight_shape, sizeof(w_dims));
  std::memcpy(y_dims, call.out_shape, sizeof(y_dims));
  row_major(x_dims, x_strides);
  row_major(y_dims, y_strides);
  row_major(w_dims, w_strides);
  if (transposed) {
    // The layer's weight is `[c, k, …]` in memory and `[k, c, …]` to the
    // backward-data pass: the same bytes with the two channel axes exchanged.
    std::swap(w_dims[0], w_dims[1]);
    std::swap(w_strides[0], w_strides[1]);
  }

  void* x = nullptr;
  void* w = nullptr;
  void* y = nullptr;
  bool built = tensor(x_dims, x_strides, x_type, kUidX, &x) &&
               tensor(w_dims, w_strides, w_type, kUidW, &w) &&
               tensor(y_dims, y_strides, y_type, kUidY, &y);

  // The geometry: the layer's padding before and after each spatial axis,
  // because the layer's output padding is zero.
  int64_t pre[kSpatial] = {call.padding[0], call.padding[1], call.padding[2]};
  int64_t post[kSpatial] = {call.padding[0], call.padding[1], call.padding[2]};
  int64_t strides[kSpatial] = {call.stride[0], call.stride[1], call.stride[2]};
  int64_t dilations[kSpatial] = {call.dilation[0], call.dilation[1], call.dilation[2]};
  int64_t spatial = kSpatial;
  const int32_t compute = kComputeFloat;
  const int32_t mode = kCrossCorrelation;
  void* convolution = nullptr;
  built = built && make(kDescriptorConvolution, &convolution) &&
          set(convolution, kAttrConvolutionCompType, kTypeDataType, 1, &compute) &&
          set(convolution, kAttrConvolutionConvMode, kTypeConvolutionMode, 1, &mode) &&
          set(convolution, kAttrConvolutionSpatialDims, kTypeInt64, 1, &spatial) &&
          set(convolution, kAttrConvolutionPrePaddings, kTypeInt64, kSpatial, pre) &&
          set(convolution, kAttrConvolutionPostPaddings, kTypeInt64, kSpatial, post) &&
          set(convolution, kAttrConvolutionFilterStrides, kTypeInt64, kSpatial, strides) &&
          set(convolution, kAttrConvolutionDilations, kTypeInt64, kSpatial, dilations) &&
          finalize(convolution);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  void* operation = nullptr;
  if (built) {
    built = make(transposed ? kDescriptorOperationBackwardData : kDescriptorOperationForward,
                 &operation) &&
            set(operation, transposed ? kAttrOperationBwdDataAlpha : kAttrOperationForwardAlpha,
                kTypeFloat, 1, &alpha) &&
            set(operation, transposed ? kAttrOperationBwdDataBeta : kAttrOperationForwardBeta,
                kTypeFloat, 1, &beta) &&
            set(operation,
                transposed ? kAttrOperationBwdDataConvDesc : kAttrOperationForwardConvDesc,
                kTypeBackendDescriptor, 1, &convolution) &&
            set(operation, transposed ? kAttrOperationBwdDataW : kAttrOperationForwardW,
                kTypeBackendDescriptor, 1, &w) &&
            set(operation, transposed ? kAttrOperationBwdDataDx : kAttrOperationForwardX,
                kTypeBackendDescriptor, 1, &y) &&
            set(operation, transposed ? kAttrOperationBwdDataDy : kAttrOperationForwardY,
                kTypeBackendDescriptor, 1, &x) &&
            finalize(operation);
  }
  void* graph = nullptr;
  built = built && make(kDescriptorOperationGraph, &graph) &&
          set(graph, kAttrOperationGraphOps, kTypeBackendDescriptor, 1, &operation) &&
          finalize(graph);

  // The heuristic, asked once for this shape: the engine the tuning data names,
  // or the first proposal.
  void* plan = nullptr;
  if (built) {
    void* heuristic = nullptr;
    built = make(kDescriptorEngineHeuristic, &heuristic) &&
            set(heuristic, kAttrEngineHeurOperationGraph, kTypeBackendDescriptor, 1, &graph) &&
            set(heuristic, kAttrEngineHeurMode, kTypeHeurMode, 1, &kHeuristicModeA) &&
            finalize(heuristic);
    void* configs[kMaxEngineConfigs] = {};
    int64_t count = 0;
    if (built &&
        (library.backend_get(heuristic, kAttrEngineHeurResults, kTypeBackendDescriptor,
                             kMaxEngineConfigs, &count, configs) != kSuccess ||
         count <= 0)) {
      *status = refuse(error, "cuDNN's heuristic proposes no engine for this convolution");
      built = false;
    }
    int64_t take = 0;
    if (built && UNISERVE_CONV_ENGINE != kFirstEngine) {
      bool hosted = false;
      for (int64_t at = 0; at < count && !hosted; ++at) {
        void* engine = nullptr;
        int64_t index = 0;
        if (library.backend_get(configs[at], kAttrEngineCfgEngine, kTypeBackendDescriptor, 1,
                                nullptr, &engine) == kSuccess &&
            engine != nullptr &&
            library.backend_get(engine, kAttrEngineGlobalIndex, kTypeInt64, 1, nullptr, &index) ==
                kSuccess) {
          hosted = index == UNISERVE_CONV_ENGINE;
          take = at;
        }
        library.backend_destroy(engine);
      }
      if (!hosted) {
        std::snprintf(error, 512,
                      "cuDNN proposes no engine %lld for this convolution, which the package's "
                      "tuning data names",
                      static_cast<long long>(UNISERVE_CONV_ENGINE));
        *status = 1;
        built = false;
      }
    }
    if (built) {
      built = make(kDescriptorExecutionPlan, &plan) &&
              set(plan, kAttrExecutionPlanEngineConfig, kTypeBackendDescriptor, 1,
                  &configs[take]) &&
              finalize(plan);
    }
    for (int64_t at = 0; at < count; ++at) {
      library.backend_destroy(configs[at]);
    }
    library.backend_destroy(heuristic);
  }
  library.backend_destroy(graph);
  library.backend_destroy(operation);
  library.backend_destroy(convolution);
  library.backend_destroy(y);
  library.backend_destroy(w);
  library.backend_destroy(x);
  if (!built) {
    return Plan{};
  }

  Plan made;
  made.plan = plan;
  int64_t workspace = 0;
  if (library.backend_get(plan, kAttrExecutionPlanWorkspaceSize, kTypeInt64, 1, nullptr,
                          &workspace) == kSuccess &&
      workspace > 0) {
    made.workspace = static_cast<size_t>(workspace);
  }
  plans.emplace(key, made);
  *status = 0;
  return made;
}

/// Adds a bias to a contiguous `[n, k, d, h, w]` output, one element per thread.
template <typename Element>
__global__ void bias_kernel(Element* y, const Element* bias, int64_t elements, int64_t channels,
                            int64_t spatial) {
  const int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (at >= elements) {
    return;
  }
  const int64_t channel = (at / spatial) % channels;
  const float bias_value = static_cast<float>(bias[channel]);
  y[at] = static_cast<Element>(static_cast<float>(y[at]) + bias_value);
}

/// The kernel that adds the bias of the module's element type.
void add_bias(void* y, const void* bias, int64_t elements, int64_t channels, int64_t spatial,
              int32_t dtype, cudaStream_t stream) {
  const int64_t threads = 256;
  const int64_t blocks = (elements + threads - 1) / threads;
  switch (dtype) {
    case UNISERVE_CONV_DTYPE_F32:
      bias_kernel<<<blocks, threads, 0, stream>>>(static_cast<float*>(y),
                                                  static_cast<const float*>(bias), elements,
                                                  channels, spatial);
      break;
    case UNISERVE_CONV_DTYPE_F16:
      bias_kernel<<<blocks, threads, 0, stream>>>(static_cast<__half*>(y),
                                                  static_cast<const __half*>(bias), elements,
                                                  channels, spatial);
      break;
    case UNISERVE_CONV_DTYPE_BF16:
      bias_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(bias), elements,
          channels, spatial);
      break;
    default:
      break;
  }
}

/// The key of one shape's plan: the tensors, the geometry, the element types,
/// the engine and the device, because a plan belongs to a device's cuDNN
/// context.
std::string plan_key(const uniserve_conv_call& call, int32_t device) {
  std::string key;
  auto append = [&key](const char* name, const int64_t* values, int32_t count) {
    key += name;
    for (int32_t at = 0; at < count; ++at) {
      key += '_';
      key += std::to_string(values[at]);
    }
  };
  append("input", call.input_shape, 5);
  append("weight", call.weight_shape, 5);
  append("out", call.out_shape, 5);
  for (int32_t axis = 0; axis < kSpatial; ++axis) {
    key += '_';
    key += std::to_string(call.kernel[axis]);
    key += 'x';
    key += std::to_string(call.stride[axis]);
    key += 'x';
    key += std::to_string(call.padding[axis]);
    key += 'x';
    key += std::to_string(call.dilation[axis]);
  }
  key += call.transposed != 0 ? "_transposed" : "_forward";
  key += '_';
  key += std::to_string(UNISERVE_CONV_INPUT_DTYPE);
  key += '_';
  key += std::to_string(UNISERVE_CONV_WEIGHT_DTYPE);
  key += '_';
  key += std::to_string(UNISERVE_CONV_OUTPUT_DTYPE);
  key += '_';
  key += std::to_string(UNISERVE_CONV_ENGINE);
  key += "_device_";
  key += std::to_string(device);
  return key;
}

}  // namespace

extern "C" int32_t UNISERVE_CONV_ENTRY(const uniserve_conv_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  char* error = const_cast<char*>(call->error);
  if (call->input == nullptr || call->weight == nullptr || call->out == nullptr) {
    return refuse(error, "a convolution needs the addresses of its input, filter and output");
  }
  for (int32_t axis = 0; axis < kSpatial; ++axis) {
    if (call->kernel[axis] <= 0 || call->stride[axis] <= 0 || call->dilation[axis] <= 0 ||
        call->padding[axis] < 0) {
      return refuse(error,
                    "a convolution's filter, stride and dilation are positive and its padding is "
                    "not negative");
    }
  }
  if (call->kernel[0] <= 0) {
    return refuse(error, "a convolution's filter is positive");
  }
  if (call->groups <= 0) {
    return refuse(error, "a convolution's group count is positive");
  }
  if (call->transposed != 0 && call->groups != 1) {
    return refuse(error,
                  "a grouped transposed convolution is not cuDNN's backward-data pass of the "
                  "convolution it transposes, so this module does not describe it");
  }
  if (cudnn_dtype(UNISERVE_CONV_INPUT_DTYPE) < 0 ||
      cudnn_dtype(UNISERVE_CONV_WEIGHT_DTYPE) < 0 ||
      cudnn_dtype(UNISERVE_CONV_OUTPUT_DTYPE) < 0) {
    return refuse(error, "a convolution's element type is one cuDNN computes: f32, f16 or bf16");
  }
  for (int64_t value : call->input_shape) {
    if (value <= 0) {
      return refuse(error, "a convolution's input extents are positive");
    }
  }
  for (int64_t value : call->weight_shape) {
    if (value <= 0) {
      return refuse(error, "a convolution's filter extents are positive");
    }
  }
  for (int64_t value : call->out_shape) {
    if (value <= 0) {
      return refuse(error, "a convolution's output extents are positive");
    }
  }
  // `tf32` is the descriptor's statement about the numerics of an FP32
  // convolution. cuDNN 9's backend API chooses the engine's arithmetic from the
  // engine it proposes and exposes no math-type knob, so an FP32 convolution here
  // runs whatever that engine does with FP32, and a half-precision one has no
  // such choice. A descriptor that says otherwise is not one this module runs.
  const bool fp32 = UNISERVE_CONV_INPUT_DTYPE == UNISERVE_CONV_DTYPE_F32;
  if (fp32 != (call->tf32 != 0)) {
    return refuse(error,
                  "a convolution's tf32 value describes FP32 inputs only: this module's input "
                  "element type and the descriptor's tf32 disagree");
  }

  int32_t device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return refuse(error, "a convolution needs the caller's device");
  }
  int32_t status = 0;
  void* handle = handle_of(device, error, &status);
  if (handle == nullptr) {
    return status;
  }
  const Api& library = api();
  if (library.set_stream(handle, reinterpret_cast<cudaStream_t>(stream)) != kSuccess) {
    return refuse(error, "cudnnSetStream failed");
  }

  const Plan plan = plan_of(*call, plan_key(*call, device), error, &status);
  if (plan.plan == nullptr) {
    return status != 0 ? status : refuse(error, "the convolution's plan could not be built");
  }
  void* workspace = nullptr;
  if (plan.workspace > 0 &&
      cudaMallocAsync(&workspace, plan.workspace, reinterpret_cast<cudaStream_t>(stream)) !=
          cudaSuccess) {
    return refuse(error, "a convolution's workspace could not be allocated");
  }

  // The variant pack names this call's buffers by the operation graph's unique
  // ids: the input is the operation's `x` (its `dy` when transposed), the output
  // its `y` (its `dx`).
  void* variant = nullptr;
  bool built = library.backend_create(kDescriptorVariantPack, &variant) == kSuccess;
  const int64_t uids[3] = {kUidX, kUidW, kUidY};
  void* pointers[3] = {const_cast<void*>(call->input), const_cast<void*>(call->weight), call->out};
  built = built && library.backend_set(variant, kAttrVariantPackDataPointers, kTypeVoidPtr, 3,
                                       pointers) == kSuccess &&
          library.backend_set(variant, kAttrVariantPackUniqueIds, kTypeInt64, 3, uids) ==
              kSuccess &&
          library.backend_set(variant, kAttrVariantPackWorkspace, kTypeVoidPtr, 1, &workspace) ==
              kSuccess &&
          library.backend_finalize(variant) == kSuccess;
  if (!built) {
    library.backend_destroy(variant);
    if (workspace != nullptr) {
      cudaFreeAsync(workspace, reinterpret_cast<cudaStream_t>(stream));
    }
    return refuse(error, "a convolution's variant pack could not be built");
  }
  const Status executed = library.backend_execute(handle, plan.plan, variant);
  library.backend_destroy(variant);
  if (executed != kSuccess) {
    if (workspace != nullptr) {
      cudaFreeAsync(workspace, reinterpret_cast<cudaStream_t>(stream));
    }
    return refuse(error, "cudnnBackendExecute failed");
  }

  if (call->bias != nullptr) {
    const int64_t channels = call->out_shape[1];
    const int64_t spatial = call->out_shape[2] * call->out_shape[3] * call->out_shape[4];
    const int64_t elements = call->out_shape[0] * channels * spatial;
    add_bias(call->out, call->bias, elements, channels, spatial, UNISERVE_CONV_OUTPUT_DTYPE,
             reinterpret_cast<cudaStream_t>(stream));
    if (cudaGetLastError() != cudaSuccess) {
      if (workspace != nullptr) {
        cudaFreeAsync(workspace, reinterpret_cast<cudaStream_t>(stream));
      }
      return refuse(error, "a convolution's bias could not be added");
    }
  }
  if (workspace != nullptr) {
    cudaFreeAsync(workspace, reinterpret_cast<cudaStream_t>(stream));
  }
  return 0;
}
