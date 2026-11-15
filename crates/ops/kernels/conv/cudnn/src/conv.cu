// SPDX-License-Identifier: Apache-2.0
//
// The convolution module: cuDNN 9's backend API, asked once per shape for an
// engine and then executed on the caller's stream.
//
// The module is compiled against the cuDNN headers and linked against the
// library of the distribution the Build Server's toolchain description names,
// which the serving process's loader finds by the same default search.
//
// The engine is chosen as PyTorch's cuDNN convolution chooses it with
// `torch.backends.cudnn.benchmark` off, the runtime the models' references compute
// in (`aten/src/ATen/native/cudnn/Conv_v8.cpp`: `run_single_conv` and
// `filterEngineConfigs`): the first proposal of cuDNN's heuristic
// (`CUDNN_HEUR_MODE_A`, PyTorch's `CUDNN_HEUR_MODE_INSTANT`) that keeps the call's
// numerics, and when the heuristic proposes none, the first such engine of cuDNN's
// fallback list (`CUDNN_HEUR_MODE_FALLBACK`). Neither list measures anything -
// design section 5.4's rule: never measure in service. An engine keeps the call's
// numerics when it does not convert its inputs to a narrower type and, for FP32
// inputs whose call withholds `tf32`, does not compute on tensor cores, which read
// FP32 operands as TF32. A shape for which neither list holds such an engine is
// refused, with the graph and the geometry check in the message.
//
// The executable (the finalized `CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR`) is
// built once per shape and device and reused, so service calls
// `cudnnBackendExecute` and nothing else. The workspace comes from the stream's
// pool, so a captured graph records it like any other work.
//
// `ops::conv::Call` carries three spatial axes for every convolution, so this
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
// `channels_last` is the activations' storage, and every convolution takes it: the shapes in the
// call stay logical, and the input and the output get channels-last strides. The filter does not,
// because a checkpoint stores it packed. The bias is the one thing outside cuDNN that reads the
// output's order, so it is told it.
//
// The element types are the descriptor's, known when the descriptor's module is
// compiled: the package's manifest defines `UNISERVE_CONV_INPUT`,
// `UNISERVE_CONV_WEIGHT` and `UNISERVE_CONV_OUTPUT` as cuDNN's data types of the
// descriptor's fields. cuDNN computes every convolution here in FP32, which keeps
// the video VAE's precision and lets the engine use its tensor cores for it.

#include <uniserve_conv.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if !defined(UNISERVE_CONV_INPUT) || !defined(UNISERVE_CONV_WEIGHT) || \
    !defined(UNISERVE_CONV_OUTPUT)
#error "the package's manifest defines UNISERVE_CONV_INPUT, _WEIGHT and _OUTPUT"
#endif

namespace {

/// Whether a cuDNN data type is one this module computes: an operand's storage
/// it reads and writes, around FP32 arithmetic.
constexpr bool computed(cudnnDataType_t type) {
  return type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_HALF || type == CUDNN_DATA_BFLOAT16;
}

static_assert(computed(UNISERVE_CONV_INPUT) && computed(UNISERVE_CONV_WEIGHT) &&
                  computed(UNISERVE_CONV_OUTPUT),
              "a convolution's element types are those cuDNN computes: f32, f16 or bf16");

// Whether this module's input element type is FP32, the only precision whose
// convolution has a TF32 choice to make.
constexpr bool kFp32Inputs = UNISERVE_CONV_INPUT == CUDNN_DATA_FLOAT;

/// The alignment every tensor of this program is handed to cuDNN with.
///
/// cuDNN 9 requires a tensor's byte alignment to be stated before its graph is finalized, and
/// answers a missing one with the attribute number rather than a sentence. The buffers are the
/// caller's: the exchange's and the arena's allocations are at least this aligned (the arena pads
/// every tensor to `ALIGNMENT`), so 16 is a statement about the allocation and not a request.
constexpr int64_t kTensorByteAlignment = 16;
// The number of engine configurations one heuristic query returns when it states no count.
constexpr int64_t kMaxEngineConfigs = 32;
// The notes one engine can carry, as `cudnnBackendGetAttribute` returns them.
constexpr int64_t kMaxNumericalNotes = 16;
// The unique ids tying the operation graph's tensors to a variant pack.
constexpr int64_t kUidX = 0;
constexpr int64_t kUidW = 1;
constexpr int64_t kUidY = 2;
// The spatial axes `ops`'s call structure carries for every convolution.
constexpr int32_t kSpatial = 3;

/// Writes `message` into an entry's error buffer and answers its status.
int32_t refuse(char* error, const char* message) {
  std::snprintf(error, UNISERVE_MESSAGE, "%s", message);
  return 1;
}

/// One device's cuDNN handle, created once per process: the handle holds no
/// per-launch state and is not cheap to make.
cudnnHandle_t handle_of(int32_t device, char* error, int32_t* status) {
  static std::mutex mutex;
  static std::map<int32_t, cudnnHandle_t> handles;
  const std::lock_guard<std::mutex> guard(mutex);
  if (const auto found = handles.find(device); found != handles.end()) {
    return found->second;
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    *status = refuse(error, "this convolution's device is not available to the module");
    return nullptr;
  }
  cudnnHandle_t handle = nullptr;
  if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
    *status = refuse(error, "cudnnCreate failed for this convolution's device");
    return nullptr;
  }
  handles[device] = handle;
  *status = 0;
  return handle;
}

/// One shape's executable: its finalized execution descriptor and the
/// workspace it asks for.
struct Executable {
  cudnnBackendDescriptor_t descriptor = nullptr;
  size_t workspace = 0;
};

/// The executable of a shape, from the process's cache or from cuDNN's heuristic.
Executable executable_of(const uniserve_conv_call& call, const std::string& key,
                         cudnnHandle_t handle, char* error, int32_t* status) {
  static std::mutex mutex;
  static std::map<std::string, Executable> executables;
  const std::lock_guard<std::mutex> guard(mutex);
  if (const auto found = executables.find(key); found != executables.end()) {
    return found->second;
  }

  auto make = [error, status](cudnnBackendDescriptorType_t kind,
                              cudnnBackendDescriptor_t* descriptor) {
    if (cudnnBackendCreateDescriptor(kind, descriptor) != CUDNN_STATUS_SUCCESS) {
      *status = refuse(error, "cudnnBackendCreateDescriptor failed");
      return false;
    }
    return true;
  };
  auto set = [error, status](cudnnBackendDescriptor_t descriptor,
                             cudnnBackendAttributeName_t attribute,
                             cudnnBackendAttributeType_t type, int64_t count, const void* values) {
    if (cudnnBackendSetAttribute(descriptor, attribute, type, count, values) !=
        CUDNN_STATUS_SUCCESS) {
      *status = refuse(error, "cudnnBackendSetAttribute failed");
      return false;
    }
    return true;
  };
  // Which descriptor could not be finalized, because cuDNN's own reason is one string for all of them:
  // a tensor with the wrong strides, a convolution with an impossible geometry and a graph whose
  // implied output shape is not the one given all read the same, and the caller has to know which.
  auto finalize = [error, status](cudnnBackendDescriptor_t descriptor, const char* which) {
    if (cudnnBackendFinalize(descriptor) != CUDNN_STATUS_SUCCESS) {
      char why[256];
      std::snprintf(why, sizeof(why), "cudnnBackendFinalize failed on %s", which);
      *status = refuse(error, why);
      return false;
    }
    return true;
  };
  // A tensor's refusal carries the tensor's own numbers. A caller that reads only the operand's name
  // cannot tell a zero extent from a layout cuDNN will not take from a dimension past its limit, and
  // those have different cures - so the dims, the strides and the element type travel with the fault.
  auto finalize_tensor = [error, status](cudnnBackendDescriptor_t descriptor, const char* which,
                                         const int64_t* dims, const int64_t* strides,
                                         cudnnDataType_t dtype) {
    if (cudnnBackendFinalize(descriptor) == CUDNN_STATUS_SUCCESS) {
      return true;
    }
    std::snprintf(error, UNISERVE_MESSAGE,
                  "cudnnBackendFinalize failed on %s: dims [%lld %lld %lld %lld %lld] strides "
                  "[%lld %lld %lld %lld %lld] dtype %d",
                  which, static_cast<long long>(dims[0]), static_cast<long long>(dims[1]),
                  static_cast<long long>(dims[2]), static_cast<long long>(dims[3]),
                  static_cast<long long>(dims[4]), static_cast<long long>(strides[0]),
                  static_cast<long long>(strides[1]), static_cast<long long>(strides[2]),
                  static_cast<long long>(strides[3]), static_cast<long long>(strides[4]),
                  static_cast<int>(dtype));
    *status = 1;
    return false;
  };
  auto tensor = [&](const int64_t* dims, const int64_t* strides, cudnnDataType_t dtype,
                    int64_t uid, const char* which, cudnnBackendDescriptor_t* descriptor) {
    return make(CUDNN_BACKEND_TENSOR_DESCRIPTOR, descriptor) &&
           set(*descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype) &&
           set(*descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, kSpatial + 2, dims) &&
           set(*descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, kSpatial + 2, strides) &&
           set(*descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid) &&
           set(*descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1,
               &kTensorByteAlignment) &&
           finalize_tensor(*descriptor, which, dims, strides, dtype);
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
  const cudnnDataType_t x_type = UNISERVE_CONV_INPUT;
  const cudnnDataType_t w_type = UNISERVE_CONV_WEIGHT;
  const cudnnDataType_t y_type = UNISERVE_CONV_OUTPUT;

  int64_t x_dims[5];
  int64_t x_strides[5];
  int64_t w_dims[5];
  int64_t w_strides[5];
  int64_t y_dims[5];
  int64_t y_strides[5];
  std::memcpy(x_dims, call.input_shape, sizeof(x_dims));
  std::memcpy(w_dims, call.weight_shape, sizeof(w_dims));
  std::memcpy(y_dims, call.out_shape, sizeof(y_dims));
  /// Channels-last strides of dimensions ordered `[n, c, spatial...]`: the channel axis runs fastest,
  /// then the `kSpatial` axes from the innermost out, then the batch. cuDNN reads the format from the
  /// strides. A 1-d convolution's `[l, 1, 1]` needs no case of its own: its two unit axes multiply
  /// by one, so `l` takes the channel count as its stride, which is what `[n, l, c]` storage is.
  auto channels_last_strides = [](const int64_t* dims, int64_t* strides) {
    strides[1] = 1;
    int64_t running = dims[1];
    for (int32_t axis = kSpatial + 1; axis >= 2; --axis) {
      strides[axis] = running;
      running *= dims[axis];
    }
    strides[0] = running;
  };

  const bool channels_last = call.channels_last != 0;
  row_major(x_dims, x_strides);
  row_major(y_dims, y_strides);
  row_major(w_dims, w_strides);
  if (channels_last) {
    // The activations and the output are the two operands a channels-last *memory format*
    // reorders. The filter is not one: a checkpoint stores every module's weight packed in
    // `[out, in, kd, kh, kw]`, which is the KCRS form the filter descriptor wants, and no
    // framework's channels-last format touches module weights. Reordering it here reads the
    // filter at the wrong offsets while the numbers still look like a convolution.
    channels_last_strides(x_dims, x_strides);
    channels_last_strides(y_dims, y_strides);
  }
  // A transposed layer's weight is `[in, out, …]`, and that is already the filter the backward-data
  // pass reads: its `dy` is the layer's input, so the filter's `k` is `in`, and its `dx` the layer's
  // output, so its `c` is `out`. The axes stay as they are.

  cudnnBackendDescriptor_t x = nullptr;
  cudnnBackendDescriptor_t w = nullptr;
  cudnnBackendDescriptor_t y = nullptr;
  bool built = tensor(x_dims, x_strides, x_type, kUidX, "the input tensor", &x) &&
               tensor(w_dims, w_strides, w_type, kUidW, "the weight tensor", &w) &&
               tensor(y_dims, y_strides, y_type, kUidY, "the output tensor", &y);

  // The geometry: the layer's padding before and after each spatial axis,
  // because the layer's output padding is zero.
  int64_t pre[kSpatial] = {call.padding[0], call.padding[1], call.padding[2]};
  int64_t post[kSpatial] = {call.padding[0], call.padding[1], call.padding[2]};
  int64_t strides[kSpatial] = {call.stride[0], call.stride[1], call.stride[2]};
  int64_t dilations[kSpatial] = {call.dilation[0], call.dilation[1], call.dilation[2]};
  int64_t spatial = kSpatial;
  // FP32 arithmetic for every convolution, and the cross-correlation PyTorch's `Conv*d` computes.
  const cudnnDataType_t compute = CUDNN_DATA_FLOAT;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  cudnnBackendDescriptor_t convolution = nullptr;
  built = built && make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convolution) &&
          set(convolution, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute) &&
          set(convolution, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1,
              &mode) &&
          set(convolution, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial) &&
          set(convolution, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, kSpatial, pre) &&
          set(convolution, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, kSpatial,
              post) &&
          set(convolution, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, kSpatial,
              strides) &&
          set(convolution, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, kSpatial,
              dilations) &&
          finalize(convolution, "the convolution");

  // **Which descriptor is the operation's input and which its output depends on the pass.** A forward
  // convolution is `Y = W * X` over the layer's own `input` and `out`; the backward-data pass of a
  // transposed one is `Dx = W * Dy`, where `dx` is the layer's `out` and `dy` its `input`. Handing `y`
  // to the forward's `X` finalizes a graph whose implied output shape is the input's, which cuDNN
  // refuses - so the descriptors follow the case, not only the attribute's name.
  // The descriptor the operation's `X`/`Dx` attribute takes, and the one its `Y`/`Dy` takes.
  cudnnBackendDescriptor_t dx_or_x = transposed ? y : x;
  cudnnBackendDescriptor_t dy_or_y = transposed ? x : y;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnBackendDescriptor_t operation = nullptr;
  if (built) {
    built = make(transposed ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
                            : CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                 &operation) &&
            set(operation,
                transposed ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA
                           : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                CUDNN_TYPE_FLOAT, 1, &alpha) &&
            set(operation,
                transposed ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA
                           : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                CUDNN_TYPE_FLOAT, 1, &beta) &&
            set(operation,
                transposed ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC
                           : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution) &&
            set(operation,
                transposed ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W
                           : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w) &&
            set(operation,
                transposed ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX
                           : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dx_or_x) &&
            set(operation,
                transposed ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY
                           : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dy_or_y) &&
            finalize(operation, "the operation");
  }
  // `CUDNN_ATTR_OPERATIONGRAPH_HANDLE`: the device the graph is built for. Without one the library
  // reads a blank device, every engine that checks its grid or architecture against it declines, and
  // a graph that a channels-last engine happens not to check is the only kind that finds an engine.
  cudnnBackendDescriptor_t graph = nullptr;
  built = built && make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph) &&
          set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle) &&
          set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
              &operation) &&
          finalize(graph, "the operation graph");

  // The device the engine is for, which the heuristic is told twice: its multiprocessor count
  // (`CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET`) and its properties (`..._DEVICEPROP`). cuDNN treats
  // both as optional, and **an engine that needs a device property the request does not name is
  // not proposed at all** - so a request that leaves them out can come back empty for a shape that
  // has an engine.
  int32_t ordinal = 0;
  int32_t multiprocessors = 0;
  cudnnBackendDescriptor_t properties = nullptr;
  if (built && (cudaGetDevice(&ordinal) != cudaSuccess ||
                cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                                       ordinal) != cudaSuccess)) {
    *status = refuse(error, "the convolution cannot read its device's multiprocessor count");
    built = false;
  }
  built = built && make(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR, &properties) &&
          set(properties, CUDNN_ATTR_DEVICEPROP_DEVICE_ID, CUDNN_TYPE_INT32, 1, &ordinal) &&
          finalize(properties, "the device's properties");

  // One list of engine configurations for the graph, asked of a heuristic descriptor of `mode`
  // made for it: an attribute cannot be reset once a descriptor is finalized, so each list has its
  // own. Fills `configs` with configurations the caller releases and `answered` with whether
  // cuDNN answered the query, and answers false when the heuristic descriptor cannot be built.
  //
  // **A list longer than the array is a query cuDNN refuses, and a refused query reads exactly like
  // an empty one.** The count is asked for separately - zero elements, no array - and the array is
  // sized from the answer, so the list is readable whatever its length. The cap keeps a library that
  // reports nonsense from taking the process with it. The results are written into descriptors the
  // caller made: an array of null pointers is refused entry by entry, and the list then reads as
  // empty. The configurations do not refer to the heuristic, which is released here.
  auto list_of = [&](cudnnBackendHeurMode_t mode, std::vector<cudnnBackendDescriptor_t>* configs,
                     bool* answered) {
    cudnnBackendDescriptor_t heuristic = nullptr;
    const bool asked =
        make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristic) &&
        set(heuristic, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
            &graph) &&
        set(heuristic, CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET, CUDNN_TYPE_INT32, 1,
            &multiprocessors) &&
        set(heuristic, CUDNN_ATTR_ENGINEHEUR_DEVICEPROP, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
            &properties) &&
        set(heuristic, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode) &&
        finalize(heuristic, "the heuristic");
    *answered = false;
    configs->clear();
    if (asked) {
      int64_t needed = 0;
      const bool sized =
          cudnnBackendGetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                   CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &needed,
                                   nullptr) == CUDNN_STATUS_SUCCESS;
      int64_t room = sized && needed > 0 ? needed : kMaxEngineConfigs;
      if (room > kMaxEngineConfigs * 64) {
        room = kMaxEngineConfigs * 64;
      }
      configs->assign(static_cast<size_t>(room), nullptr);
      for (cudnnBackendDescriptor_t& config : *configs) {
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &config) !=
            CUDNN_STATUS_SUCCESS) {
          config = nullptr;
        }
      }
      int64_t count = 0;
      *answered = cudnnBackendGetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR, room, &count,
                                           configs->data()) == CUDNN_STATUS_SUCCESS;
      // A refused query may have written some of its array, and an answered one fills only its
      // count: the rest are configurations only this function can release.
      const size_t kept =
          *answered && count > 0 ? static_cast<size_t>(count < room ? count : room) : 0;
      for (size_t at = kept; at < configs->size(); ++at) {
        if ((*configs)[at] != nullptr) {
          cudnnBackendDestroyDescriptor((*configs)[at]);
        }
      }
      configs->resize(kept);
    }
    if (heuristic != nullptr) {
      cudnnBackendDestroyDescriptor(heuristic);
    }
    return asked;
  };

  // Whether an engine keeps the call's numerics, read from the notes cuDNN publishes for it - the
  // rule of PyTorch's `filterEngineConfigs`. `CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS` marks an
  // engine that converts its inputs to a narrower type before computing, which no call admits: the
  // narrowest product an FP32 call may declare is TF32's, and a 16-bit call's products are exact.
  // `CUDNN_NUMERICAL_NOTE_TENSOR_CORE` marks a tensor-core engine, which reads FP32 operands as
  // TF32, so an FP32 call takes one only when it grants `tf32`. An engine whose notes cannot be
  // read states nothing the call can rely on and is not taken.
  const bool exact_fp32 = kFp32Inputs && call.tf32 == 0;
  auto keeps_numerics = [exact_fp32](cudnnBackendDescriptor_t config) {
    // As with the list, the engine is read into a descriptor made here.
    cudnnBackendDescriptor_t engine = nullptr;
    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine) !=
        CUDNN_STATUS_SUCCESS) {
      return false;
    }
    int64_t engines = 0;
    cudnnBackendNumericalNote_t notes[kMaxNumericalNotes] = {};
    int64_t notes_count = 0;
    const bool read =
        cudnnBackendGetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE,
                                 CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engines,
                                 &engine) == CUDNN_STATUS_SUCCESS &&
        cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                 CUDNN_TYPE_NUMERICAL_NOTE, kMaxNumericalNotes, &notes_count,
                                 notes) == CUDNN_STATUS_SUCCESS;
    cudnnBackendDestroyDescriptor(engine);
    if (!read) {
      return false;
    }

    for (int64_t at = 0; at < notes_count && at < kMaxNumericalNotes; ++at) {
      if (notes[at] == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
          (exact_fp32 && notes[at] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE)) {
        return false;
      }
    }
    return true;
  };

  // The engine: the first configuration that keeps the call's numerics, of the heuristic's list
  // and then of the fallback list. **The heuristic's list can hold no exact engine for a wide FP32
  // convolution**: on sm_100 it can be tensor-core configurations alone, all of which an exact call
  // forbids, while the fallback list holds engines that compute in FP32.
  //
  // What each list said when neither holds an engine the call may take: a query cuDNN refused, an
  // empty list and a list of engines the call's numerics forbid have different cures, so the
  // refusal names each list's answer.
  const cudnnBackendHeurMode_t modes[2] = {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK};
  const char* said[2] = {"unasked", "unasked"};
  std::vector<cudnnBackendDescriptor_t> configs;
  int64_t take = -1;
  for (int32_t list = 0; built && list < 2 && take < 0; ++list) {
    bool answered = false;
    built = list_of(modes[list], &configs, &answered);
    for (int64_t at = 0; at < static_cast<int64_t>(configs.size()) && take < 0; ++at) {
      if (keeps_numerics(configs[at])) {
        take = at;
      }
    }
    if (take < 0) {
      said[list] = !answered         ? "unanswered"
                   : configs.empty() ? "no engine"
                                     : "only engines whose numerics the call forbids";
      for (cudnnBackendDescriptor_t config : configs) {
        cudnnBackendDestroyDescriptor(config);
      }
      configs.clear();
    }
  }

  if (built && take < 0) {
    // **The geometry is checked here, before the library is blamed.** cuDNN answers an implied
    // output shape that disagrees with the stated one with no engines at all, not with a geometry
    // error, so a graph whose arithmetic is wrong reads exactly like a shape the library cannot
    // serve. The forward's axis is `(in + 2p - d(k-1) - 1)/s + 1` and the backward-data pass's is
    // `(dy - 1)s - 2p + d(k-1) + 1`; both are recomputed from the stated input, weight and
    // geometry, and the refusal prints them beside the stated output.
    int64_t implied[kSpatial] = {0, 0, 0};
    bool agrees = true;
    for (int32_t axis = 0; axis < kSpatial; ++axis) {
      const int64_t window =
          static_cast<int64_t>(call.dilation[axis]) * (static_cast<int64_t>(call.kernel[axis]) - 1) + 1;
      const int64_t spread = 2 * static_cast<int64_t>(call.padding[axis]);
      const int64_t step = static_cast<int64_t>(call.stride[axis]);
      const int64_t source = x_dims[axis + 2];
      implied[axis] = transposed ? (source - 1) * step - spread + window
                                 : (source + spread - window) / step + 1;
      agrees = agrees && implied[axis] == y_dims[axis + 2];
    }
    // **The graph travels with the refusal.** An empty engine list is an answer about one exact
    // request: these dims, these strides, this geometry, this element type and this permission.
    // A reader given only the sentence has to guess which of them cuDNN would not take - and the
    // caller of this operator cannot see them at all, because the descriptor carries no geometry.
    char why[512];
    std::snprintf(
        why, sizeof(why),
        // The things that decide the cure come first, so that a message truncated by its own room
        // still says what each list answered and whether the geometry agrees: `heuristic` and
        // `fallback` are the lists' answers, and `geometry` compares the stated output with the
        // one the input, weight and geometry imply. The graph follows, because a reader with only
        // the verdict has to guess which shape cuDNN would not take.
        "cuDNN proposes no engine for this convolution: %s, heuristic %s, fallback %s, geometry "
        "%s (imply [%lld %lld %lld]); x [%lld %lld %lld %lld %lld] strides [%lld %lld %lld %lld "
        "%lld] type %d, w [%lld %lld %lld %lld %lld] type %d, y [%lld %lld %lld %lld %lld] type "
        "%d, kernel [%lld %lld %lld] stride [%lld %lld %lld] padding [%lld %lld %lld] dilation "
        "[%lld %lld %lld], tf32 %d, channels_last %d",
        transposed ? "backward-data" : "forward", said[0], said[1],
        agrees ? "agrees" : "DISAGREES",
        static_cast<long long>(implied[0]), static_cast<long long>(implied[1]),
        static_cast<long long>(implied[2]), static_cast<long long>(x_dims[0]),
        static_cast<long long>(x_dims[1]), static_cast<long long>(x_dims[2]),
        static_cast<long long>(x_dims[3]), static_cast<long long>(x_dims[4]),
        static_cast<long long>(x_strides[0]), static_cast<long long>(x_strides[1]),
        static_cast<long long>(x_strides[2]), static_cast<long long>(x_strides[3]),
        static_cast<long long>(x_strides[4]), static_cast<int>(x_type),
        static_cast<long long>(w_dims[0]), static_cast<long long>(w_dims[1]),
        static_cast<long long>(w_dims[2]), static_cast<long long>(w_dims[3]),
        static_cast<long long>(w_dims[4]), static_cast<int>(w_type),
        static_cast<long long>(y_dims[0]), static_cast<long long>(y_dims[1]),
        static_cast<long long>(y_dims[2]), static_cast<long long>(y_dims[3]),
        static_cast<long long>(y_dims[4]), static_cast<int>(y_type),
        static_cast<long long>(call.kernel[0]), static_cast<long long>(call.kernel[1]),
        static_cast<long long>(call.kernel[2]), static_cast<long long>(call.stride[0]),
        static_cast<long long>(call.stride[1]), static_cast<long long>(call.stride[2]),
        static_cast<long long>(call.padding[0]), static_cast<long long>(call.padding[1]),
        static_cast<long long>(call.padding[2]), static_cast<long long>(call.dilation[0]),
        static_cast<long long>(call.dilation[1]), static_cast<long long>(call.dilation[2]),
        static_cast<int>(call.tf32), static_cast<int>(call.channels_last));
    *status = refuse(error, why);
    built = false;
  }

  cudnnBackendDescriptor_t executable = nullptr;
  if (built) {
    built = make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executable) &&
            set(executable, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configs[take]) &&
            finalize(executable, "the executable");
  }
  for (cudnnBackendDescriptor_t config : configs) {
    cudnnBackendDestroyDescriptor(config);
  }
  // The heuristics, which name the device-properties descriptor and the graph, are released where
  // their lists are read. A descriptor that was never made is null, which cuDNN refuses to destroy
  // and nothing here needs.
  for (cudnnBackendDescriptor_t made : {properties, graph, operation, convolution, y, w, x}) {
    if (made != nullptr) {
      cudnnBackendDestroyDescriptor(made);
    }
  }
  if (!built) {
    if (executable != nullptr) {
      cudnnBackendDestroyDescriptor(executable);
    }
    return Executable{};
  }

  Executable made;
  made.descriptor = executable;
  int64_t workspace = 0;
  if (cudnnBackendGetAttribute(executable, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                               CUDNN_TYPE_INT64, 1, nullptr, &workspace) == CUDNN_STATUS_SUCCESS &&
      workspace > 0) {
    made.workspace = static_cast<size_t>(workspace);
  }
  executables.emplace(key, made);
  *status = 0;
  return made;
}

/// Adds a bias to a dense output, one element per thread. `channel_stride` is the elements between
/// two channels of one position: the spatial extent of `[n, k, d, h, w]` storage, 1 of channels-last
/// storage, where the channel is the fastest axis.
template <typename Element>
__global__ void bias_kernel(Element* y, const Element* bias, int64_t elements, int64_t channels,
                            int64_t channel_stride) {
  const int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (at >= elements) {
    return;
  }
  const int64_t channel = (at / channel_stride) % channels;
  const float bias_value = static_cast<float>(bias[channel]);
  y[at] = static_cast<Element>(static_cast<float>(y[at]) + bias_value);
}

/// The kernel that adds the bias of the module's element type.
void add_bias(void* y, const void* bias, int64_t elements, int64_t channels,
              int64_t channel_stride, cudnnDataType_t dtype, cudaStream_t stream) {
  const int64_t threads = 256;
  const int64_t blocks = (elements + threads - 1) / threads;
  switch (dtype) {
    case CUDNN_DATA_FLOAT:
      bias_kernel<<<blocks, threads, 0, stream>>>(static_cast<float*>(y),
                                                  static_cast<const float*>(bias), elements,
                                                  channels, channel_stride);
      break;
    case CUDNN_DATA_HALF:
      bias_kernel<<<blocks, threads, 0, stream>>>(static_cast<__half*>(y),
                                                  static_cast<const __half*>(bias), elements,
                                                  channels, channel_stride);
      break;
    case CUDNN_DATA_BFLOAT16:
      bias_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(bias), elements,
          channels, channel_stride);
      break;
    default:
      break;
  }
}

/// The key of one shape's executable: the tensors, the geometry, the element
/// types and the device, because an executable belongs to a device's cuDNN
/// context.
std::string executable_key(const uniserve_conv_call& call, int32_t device) {
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
  key += std::to_string(UNISERVE_CONV_INPUT);
  key += '_';
  key += std::to_string(UNISERVE_CONV_WEIGHT);
  key += '_';
  key += std::to_string(UNISERVE_CONV_OUTPUT);
  // The engine the executable holds depends on the call's TF32 permission...
  key += call.tf32 != 0 ? "_tf32" : "_exact";
  // ...and on the iteration order, because the two orders' tensor descriptors make different
  // executables over one engine: one cached for one order would give the other the wrong
  // strides.
  key += call.channels_last != 0 ? "_ndhwc" : "_ncdhw";
  key += "_device_";
  key += std::to_string(device);
  return key;
}

}  // namespace

extern "C" int32_t uniserve_conv(const uniserve_conv_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  char* error = const_cast<char*>(call->error);
  // `tf32` is a permission, not a requirement: it says an FP32 convolution *may*
  // narrow its inputs to TF32. A convolution whose inputs are half precision has no
  // such choice, so asking for it there is not a call this module can honour. An FP32
  // convolution may both grant and withhold the permission, and the engine selection
  // above is what makes the withholding real: with `tf32` clear, only an engine that
  // does not narrow its inputs is used.
  if (call->tf32 != 0 && !kFp32Inputs) {
    return refuse(error,
                  "a convolution's tf32 value describes FP32 inputs only: this module's input "
                  "element type cannot use it");
  }

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
  if (call->groups <= 0) {
    return refuse(error, "a convolution's group count is positive");
  }
  if (call->transposed != 0 && call->groups != 1) {
    return refuse(error,
                  "a grouped transposed convolution is not cuDNN's backward-data pass of the "
                  "convolution it transposes, so this module does not describe it");
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
  int32_t device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return refuse(error, "a convolution needs the caller's device");
  }
  int32_t status = 0;
  cudnnHandle_t handle = handle_of(device, error, &status);
  if (handle == nullptr) {
    return status;
  }
  if (cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)) != CUDNN_STATUS_SUCCESS) {
    return refuse(error, "cudnnSetStream failed");
  }

  const Executable executable =
      executable_of(*call, executable_key(*call, device), handle, error, &status);
  if (executable.descriptor == nullptr) {
    return status != 0 ? status
                       : refuse(error, "the convolution's executable could not be built");
  }
  void* workspace = nullptr;
  if (executable.workspace > 0 &&
      cudaMallocAsync(&workspace, executable.workspace, reinterpret_cast<cudaStream_t>(stream)) !=
          cudaSuccess) {
    return refuse(error, "a convolution's workspace could not be allocated");
  }

  // The variant pack names this call's buffers by the operation graph's unique
  // ids: the input is the operation's `x` (its `dy` when transposed), the output
  // its `y` (its `dx`).
  cudnnBackendDescriptor_t variant = nullptr;
  bool built = cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant) ==
               CUDNN_STATUS_SUCCESS;
  const int64_t uids[3] = {kUidX, kUidW, kUidY};
  // **Each pointer is the tensor its descriptor describes.** The id `X` is the descriptor of the
  // layer's input and `Y` of its output, in both passes: a transposed pass reads `dy` from `X`'s
  // descriptor and writes `dx` through `Y`'s, and the graph above is what names them so.
  void* pointers[3] = {const_cast<void*>(call->input), const_cast<void*>(call->weight),
                       const_cast<void*>(call->out)};
  built = built &&
          cudnnBackendSetAttribute(variant, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                   CUDNN_TYPE_VOID_PTR, 3, pointers) == CUDNN_STATUS_SUCCESS &&
          cudnnBackendSetAttribute(variant, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64,
                                   3, uids) == CUDNN_STATUS_SUCCESS &&
          cudnnBackendSetAttribute(variant, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                   CUDNN_TYPE_VOID_PTR, 1, &workspace) == CUDNN_STATUS_SUCCESS &&
          cudnnBackendFinalize(variant) == CUDNN_STATUS_SUCCESS;
  if (!built) {
    if (variant != nullptr) {
      cudnnBackendDestroyDescriptor(variant);
    }
    if (workspace != nullptr) {
      cudaFreeAsync(workspace, reinterpret_cast<cudaStream_t>(stream));
    }
    return refuse(error, "a convolution's variant pack could not be built");
  }
  const cudnnStatus_t executed = cudnnBackendExecute(handle, executable.descriptor, variant);
  cudnnBackendDestroyDescriptor(variant);
  if (executed != CUDNN_STATUS_SUCCESS) {
    if (workspace != nullptr) {
      cudaFreeAsync(workspace, reinterpret_cast<cudaStream_t>(stream));
    }
    return refuse(error, "cudnnBackendExecute failed");
  }

  if (call->bias != nullptr) {
    const int64_t channels = call->out_shape[1];
    const int64_t spatial = call->out_shape[2] * call->out_shape[3] * call->out_shape[4];
    const int64_t elements = call->out_shape[0] * channels * spatial;
    add_bias(call->out, call->bias, elements, channels, call->channels_last != 0 ? 1 : spatial,
             UNISERVE_CONV_OUTPUT, reinterpret_cast<cudaStream_t>(stream));
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
