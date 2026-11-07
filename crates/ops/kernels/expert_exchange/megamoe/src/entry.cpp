// SPDX-License-Identifier: Apache-2.0
//
// The megamoe backend of expert_exchange: one launch per layer exchanges the tokens,
// runs the experts and combines.
//
// The kernels are FlashInfer's MegaMoE (`vendor/`), compiled here by the CuTe DSL. The
// Python deployment placed their buffers on an NVSHMEM symmetric heap, and NVSHMEM
// serves one rank per process, which a main process is not. The kernels ask for less
// than a heap: they reach a peer's buffer by adding that peer's offset to the local
// address, so this module maps one window per rank, each rank's buffers at the same
// offset of its window, and states the differences between the windows.
//
// The data flow of a layer, per rank:
//
// - the layer encodes its rows as NVFP4 straight into the window's `packed` views;
// - `uniserve_megamoe_stage_routes` writes the routes as 64-bit ids, masking the rows
//   past the live count;
// - `cute_dsl_mega_wrapper` launches the fused kernel and its reduction, which leave
//   this rank's combined rows in the caller's output.
//
// A rank with no tokens still launches over one masked row, because its experts serve
// the tokens its peers route to them.

#include "../include/uniserve_expert_exchange.h"
#include "mega.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

extern "C" int32_t uniserve_megamoe_stage_routes(int64_t* routes, float* weights,
                                                 const int32_t* ids, const float* route_weights,
                                                 const int32_t* live, int32_t tokens,
                                                 int32_t extent, int32_t top, int32_t experts,
                                                 void* stream);

namespace {

constexpr int kRanks = UNISERVE_RANKS;
constexpr int kHidden = UNISERVE_HIDDEN;
constexpr int kIntermediate = UNISERVE_INTERMEDIATE;
constexpr int kTop = UNISERVE_TOP;
constexpr int kExperts = UNISERVE_EXPERTS;
constexpr int kMaxTokens = UNISERVE_MAX_TOKENS;
constexpr int kLocalExperts = kExperts / kRanks;
constexpr int kScaleColumns = ((kHidden / 16 + 3) / 4) * 4;
constexpr size_t kLocalWorkspace = UNISERVE_LOCAL_WORKSPACE;
constexpr size_t kSharedWorkspace = UNISERVE_SHARED_WORKSPACE;

size_t aligned(size_t bytes, size_t to = 256) { return (bytes + to - 1) / to * to; }

/// Offsets of one rank's window, the same on every rank.
struct Layout {
  size_t shared;   // the kernels' own symmetric workspace
  size_t packed;   // [max_tokens, hidden / 2]
  size_t scales;   // [max_tokens, scale columns]
  size_t routes;   // [max_tokens, top] i64
  size_t weights;  // [max_tokens, top] f32
  size_t bytes;
};

Layout layout_of() {
  Layout layout{};
  size_t at = 0;
  auto take = [&at](size_t bytes) {
    const size_t here = at;
    at = aligned(at + bytes);
    return here;
  };
  layout.shared = take(kSharedWorkspace);
  layout.packed = take(static_cast<size_t>(kMaxTokens) * kHidden / 2);
  layout.scales = take(static_cast<size_t>(kMaxTokens) * kScaleColumns);
  layout.routes = take(static_cast<size_t>(kMaxTokens) * kTop * 8);
  layout.weights = take(static_cast<size_t>(kMaxTokens) * kTop * 4);
  layout.bytes = at;
  return layout;
}

struct Transport {
  int32_t rank;
  int64_t device;
  Layout layout;
  CUdeviceptr base;  // `ranks` windows of `window` bytes, this rank's at `rank * window`
  size_t window;
  CUmemGenericAllocationHandle own;
  CUmemGenericAllocationHandle imported[kRanks];
  bool mapped;
  void* local_workspace;
  void* idle_output;  // [1, hidden] the one masked row of a rank with no tokens
};

char* own_base(const Transport& transport) {
  return reinterpret_cast<char*>(transport.base) + transport.rank * transport.window;
}

int32_t refuse(char* error, const char* message) {
  std::snprintf(error, 512, "%s", message);
  return 1;
}

int32_t driver_refusal(char* error, const char* what, CUresult status) {
  const char* name = nullptr;
  cuGetErrorName(status, &name);
  std::snprintf(error, 512, "expert_exchange: %s failed: %s", what, name ? name : "unknown");
  return 1;
}

struct Shared {
  unsigned char bytes[64];
};

struct Record {
  Shared handle;
  int64_t kind;
};

/// Creates this rank's window and maps every rank's: a fabric handle where the driver
/// has one, and a file descriptor between the ranks of one process otherwise.
int32_t map_windows(Transport& transport, const UniserveGroup* group, char* error) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(transport.device);
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  size_t granularity = 0;
  CUresult status =
      cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuMemGetAllocationGranularity", status);
  }
  transport.window = aligned(transport.layout.bytes, granularity);
  CUmemAllocationHandleType kind = CU_MEM_HANDLE_TYPE_FABRIC;
  Shared mine{};
  status = cuMemCreate(&transport.own, transport.window, &prop, 0);
  if (status == CUDA_SUCCESS) {
    status = cuMemExportToShareableHandle(mine.bytes, transport.own, kind, 0);
    if (status != CUDA_SUCCESS) {
      cuMemRelease(transport.own);
    }
  }
  if (status != CUDA_SUCCESS) {
    kind = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.requestedHandleTypes = kind;
    status = cuMemCreate(&transport.own, transport.window, &prop, 0);
    if (status != CUDA_SUCCESS) {
      return driver_refusal(error, "cuMemCreate", status);
    }
    int descriptor = -1;
    status = cuMemExportToShareableHandle(&descriptor, transport.own, kind, 0);
    if (status != CUDA_SUCCESS) {
      return driver_refusal(error, "cuMemExportToShareableHandle", status);
    }
    std::memcpy(mine.bytes, &descriptor, sizeof(descriptor));
  }
  Record record{mine, static_cast<int64_t>(kind)};
  Record records[kRanks];
  if (group->all_gather(group->handle, &record, records, sizeof(Record)) != 0) {
    return refuse(error, "expert_exchange could not gather its peers' handles");
  }
  status = cuMemAddressReserve(&transport.base, transport.window * kRanks, granularity, 0, 0);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuMemAddressReserve", status);
  }
  for (int rank = 0; rank < kRanks; ++rank) {
    if (records[rank].kind != static_cast<int64_t>(kind)) {
      return refuse(error, "expert_exchange's ranks share no handle kind");
    }
    CUmemGenericAllocationHandle mapped = transport.own;
    if (rank != transport.rank) {
      void* shared = records[rank].handle.bytes;
      int descriptor = -1;
      if (kind != CU_MEM_HANDLE_TYPE_FABRIC) {
        std::memcpy(&descriptor, records[rank].handle.bytes, sizeof(descriptor));
        shared = reinterpret_cast<void*>(static_cast<intptr_t>(descriptor));
      }
      status = cuMemImportFromShareableHandle(&mapped, shared, kind);
      if (status != CUDA_SUCCESS) {
        return driver_refusal(error, "cuMemImportFromShareableHandle", status);
      }
      transport.imported[rank] = mapped;
    }
    status = cuMemMap(transport.base + rank * transport.window, transport.window, 0, mapped, 0);
    if (status != CUDA_SUCCESS) {
      return driver_refusal(error, "cuMemMap", status);
    }
  }
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(transport.device);
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  status = cuMemSetAccess(transport.base, transport.window * kRanks, &access, 1);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuMemSetAccess", status);
  }
  transport.mapped = true;
  status = cuMemsetD8(reinterpret_cast<CUdeviceptr>(own_base(transport)), 0, transport.window);
  if (status == CUDA_SUCCESS) {
    status = cuCtxSynchronize();
  }
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "clearing the window", status);
  }
  if (group->barrier(group->handle) != 0) {
    return refuse(error, "expert_exchange's ranks did not meet after clearing their windows");
  }
  return 0;
}

/// The exported module: its cubin is loaded into every device of the process once.
mega_Kernel_Module_t& module() {
  static mega_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { mega_Kernel_Module_Load(&handle); });
  return handle;
}

}  // namespace

extern "C" int32_t uniserve_expert_exchange_create(uniserve_exchange_lifecycle* call,
                                                   void* stream) {
  (void)stream;
  if (call == nullptr) {
    return 1;
  }
  call->transport = nullptr;
  if (call->group == nullptr) {
    return refuse(call->error, "expert_exchange's create needs the group's channel");
  }
  if (call->ranks != kRanks || call->attention_ranks != 0 || call->hidden != kHidden ||
      call->top != kTop || call->experts != kExperts || call->max_tokens != kMaxTokens ||
      call->intermediate != kIntermediate || call->dtype != UNISERVE_DTYPE_BF16) {
    return refuse(call->error,
                  "expert_exchange's megamoe module was compiled for another group or another "
                  "shape: its descriptor fixes them");
  }
  auto* transport = static_cast<Transport*>(std::calloc(1, sizeof(Transport)));
  if (transport == nullptr) {
    return refuse(call->error, "expert_exchange could not allocate its transport record");
  }
  transport->rank = static_cast<int32_t>(call->rank);
  transport->device = call->device;
  transport->layout = layout_of();
  int32_t status = map_windows(*transport, call->group, call->error);
  if (status == 0) {
    cudaError_t cuda = cudaMalloc(&transport->local_workspace, kLocalWorkspace);
    if (cuda == cudaSuccess) {
      cuda = cudaMemset(transport->local_workspace, 0, kLocalWorkspace);
    }
    if (cuda == cudaSuccess) {
      cuda = cudaMalloc(&transport->idle_output, static_cast<size_t>(kHidden) * 2);
    }
    if (cuda == cudaSuccess) {
      cuda = cudaDeviceSynchronize();
    }
    if (cuda != cudaSuccess) {
      status = refuse(call->error, "expert_exchange could not create its local workspace");
    }
  }
  if (status != 0) {
    std::free(transport);
    return status;
  }
  // A launch reads every route of its extent, so no row is live before the first one.
  char* own = own_base(*transport);
  cudaMemset(own + transport->layout.routes, 0xff, static_cast<size_t>(kMaxTokens) * kTop * 8);
  cudaDeviceSynchronize();
  // The exported module loads into every device of the process, which waits for those
  // devices to drain. No rank may be inside a launch that waits for its peers meanwhile,
  // so the load happens here, before the ranks leave their creation together.
  module();
  if (call->group->barrier(call->group->handle) != 0) {
    return refuse(call->error, "expert_exchange's ranks did not meet after loading the kernel");
  }
  call->transport = transport;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_step(uniserve_exchange_lifecycle* call, void* stream) {
  (void)stream;
  // The kernels size themselves from each rank's own extent, so no capacity reaches them.
  return call == nullptr || call->transport == nullptr ? 1 : 0;
}

extern "C" int32_t uniserve_expert_exchange_suspend(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  return call == nullptr ? 1
                         : refuse(call->error, "the megamoe exchange does not suspend yet");
}

extern "C" int32_t uniserve_expert_exchange_restore(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  return call == nullptr ? 1
                         : refuse(call->error, "the megamoe exchange does not suspend yet");
}

extern "C" int32_t uniserve_expert_exchange_release(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  cudaDeviceSynchronize();
  if (call->group != nullptr) {
    call->group->barrier(call->group->handle);
  }
  if (transport->mapped) {
    for (int rank = 0; rank < kRanks; ++rank) {
      cuMemUnmap(transport->base + rank * transport->window, transport->window);
      if (rank != transport->rank) {
        cuMemRelease(transport->imported[rank]);
      }
    }
    cuMemRelease(transport->own);
  }
  cuMemAddressFree(transport->base, transport->window * kRanks);
  cudaFree(transport->local_workspace);
  cudaFree(transport->idle_output);
  std::free(transport);
  call->transport = nullptr;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_info(uniserve_exchange_info* call, void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  const auto* transport = static_cast<const Transport*>(call->transport);
  char* own = own_base(*transport);
  call->packed = own + transport->layout.packed;
  call->packed_scales = own + transport->layout.scales;
  // A fused launch computes in its own storage, so the received views hold nothing a
  // layer reads; they name the window so that their extents can still be stated.
  call->states = own;
  call->ids = own;
  call->weights = own;
  call->combined = own;
  call->live = own;
  call->ranks = kRanks;
  call->rank = transport->rank;
  call->capacity = kMaxTokens;
  call->hidden = kHidden;
  call->top = kTop;
  call->dtype = UNISERVE_DTYPE_BF16;
  call->device = transport->device;
  call->fused = 1;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_launch(uniserve_exchange_call* call, void* stream) {
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  if (call->mode != UNISERVE_EXCHANGE_FUSED || call->layer == nullptr) {
    return refuse(call->error, "expert_exchange's megamoe backend runs only a fused launch");
  }
  auto* transport = static_cast<Transport*>(call->transport);
  const uniserve_fused_layer& layer = *call->layer;
  const auto tokens = static_cast<int32_t>(call->tokens);
  if (tokens > kMaxTokens) {
    return refuse(call->error, "expert_exchange's rows exceed the transport's capacity");
  }
  // A rank with no tokens launches over one masked row: its experts still serve.
  const int32_t extent = tokens > 0 ? tokens : 1;
  char* own = own_base(*transport);
  auto* routes = reinterpret_cast<int64_t*>(own + transport->layout.routes);
  auto* weights = reinterpret_cast<float*>(own + transport->layout.weights);
  if (uniserve_megamoe_stage_routes(routes, weights, call->ids, call->weights, call->live, tokens,
                                    extent, kTop, kExperts, stream) != 0) {
    return refuse(call->error, "expert_exchange could not stage its routes");
  }
  constexpr int64_t kUpRows = 2 * static_cast<int64_t>(kIntermediate);
  mega_Tensor_activation_t activation{own + transport->layout.packed, {extent, kHidden}, {kHidden}};
  mega_Tensor_activation_sf_t activation_sf{
      own + transport->layout.scales, {extent, kScaleColumns}, {kScaleColumns}};
  mega_Tensor_topk_idx_t topk_idx{routes, {extent, kTop}, {kTop}};
  mega_Tensor_topk_weights_t topk_weights{weights, {extent, kTop}, {kTop}};
  // K-major views of `[local, rows, columns / 2]`: the packed axis is the leading one,
  // counted in elements, so its stride is the row's element count.
  mega_Tensor_fc1_weight_t fc1{const_cast<void*>(layer.up_gate),
                               {kLocalExperts, kHidden, static_cast<int32_t>(kUpRows)},
                               {kUpRows * kHidden, kHidden}};
  mega_Tensor_fc1_weight_sf_t fc1_sf{const_cast<void*>(layer.up_gate_scales),
                                     {kLocalExperts, static_cast<int32_t>(layer.up_gate_scale_bytes)},
                                     {layer.up_gate_scale_bytes}};
  mega_Tensor_fc2_weight_t fc2{const_cast<void*>(layer.down),
                               {kLocalExperts, kIntermediate, kHidden},
                               {static_cast<int64_t>(kHidden) * kIntermediate, kIntermediate}};
  mega_Tensor_fc2_weight_sf_t fc2_sf{const_cast<void*>(layer.down_scales),
                                     {kLocalExperts, static_cast<int32_t>(layer.down_scale_bytes)},
                                     {layer.down_scale_bytes}};
  mega_Tensor_fc1_alpha_t fc1_alpha{const_cast<float*>(layer.up_gate_alpha), {kLocalExperts}};
  mega_Tensor_fc2_alpha_t fc2_alpha{const_cast<float*>(layer.down_alpha), {kLocalExperts}};
  mega_Tensor_fc1_norm_const_t fc1_norm{const_cast<float*>(layer.gated_scale), {kLocalExperts}};
  void* output = tokens > 0 ? call->out : transport->idle_output;
  mega_Tensor_output_activation_t out{output, {extent, kHidden}, {kHidden}};
  // Each rank's offset is the distance from this rank's window to that rank's: the
  // kernels add it to a local address to reach the same buffer on the peer.
  int64_t offsets[UNISERVE_OFFSETS] = {};
  for (int rank = 0; rank < kRanks; ++rank) {
    offsets[rank] = (static_cast<int64_t>(rank) - transport->rank) *
                    static_cast<int64_t>(transport->window);
  }
  const int32_t status = cute_dsl_mega_wrapper(
      &module(), &activation, &activation_sf, &topk_idx, &topk_weights, &fc1, &fc1_sf, &fc2,
      &fc2_sf, &fc1_alpha, &fc2_alpha, &fc1_norm, &out, transport->local_workspace,
      own + transport->layout.shared, reinterpret_cast<int64_t>(own + transport->layout.shared),
      UNISERVE_OFFSET_ARGUMENTS, transport->rank, static_cast<CUstream>(stream));
  return status == 0 ? 0 : refuse(call->error, "the fused kernel's launch failed");
}
