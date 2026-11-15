// SPDX-License-Identifier: Apache-2.0
//
// The deepep backend of expert_exchange: DeepEP's elastic dispatch and combine,
// without its Python extension.
//
// The kernels are DeepEP's (the build's pinned `deepep` source), unchanged. DeepEP specializes them at
// run time through a compiler it carries; here the descriptor fixes every template
// argument, so the build compiles them once and nothing compiles on a serving host.
// The rest of this file is what DeepEP's `ElasticBuffer` did around them for the one
// path a serving rank takes: its received rows live in the transport's own storage,
// every launch runs on the caller's stream, and no host reads a device count.
//
// The data flow of a step, per rank:
//
// - `route_kernel` turns this rank's expert ids into the 64-bit ids DeepEP reads,
//   shifted past the slots of the ranks that hold no experts, with -1 for the rows
//   past the live count, which DeepEP skips;
// - `dispatch_impl` sends each token once to each rank holding one of its experts,
//   through peer-mapped windows of an NCCL symmetric allocation;
// - `dispatch_copy_epilogue_impl` packs what arrived into `states`, `weights` and
//   the routes, ranks first, with no gaps;
// - `received_kernel` states the packed rows' count in `live` and rewrites the
//   routes as vocabulary ids the sort reads;
// - the rank's experts write each received row's sum into `combined`;
// - `combine_impl` pushes each row back to the rank it came from, and
//   `combine_reduce_epilogue_impl` sums a token's partial rows in FP32.
//
// A group is one NVLink domain: every rank's window is peer-mapped, so DeepEP's
// direct kernels run and its RDMA contexts are never requested.

#include <uniserve_expert_exchange.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <deep_ep/common/layout.cuh>
#include <deep_ep/impls/barrier.cuh>
#include <deep_ep/impls/combine.cuh>
#include <deep_ep/impls/combine_reduce_epilogue.cuh>
#include <deep_ep/impls/dispatch.cuh>
#include <deep_ep/impls/dispatch_copy_epilogue.cuh>

namespace {

using deep_ep::topk_idx_t;
namespace layout = deep_ep::elastic::layout;

// What the descriptor fixes, as the build states it.
constexpr int kRanks = UNISERVE_RANKS;
constexpr int kAttentionRanks = UNISERVE_ATTENTION_RANKS;
constexpr int kHidden = UNISERVE_HIDDEN;
constexpr int kTop = UNISERVE_TOP;
constexpr int kExperts = UNISERVE_EXPERTS;
constexpr int kMaxTokens = UNISERVE_MAX_TOKENS;

// The rows a rank receives at most, and the vocabulary DeepEP's kernels see: slots
// for the attention ranks too, which no route names.
constexpr int kReceived = kRanks * kMaxTokens;
constexpr int kLocalExperts = kExperts / (kRanks - kAttentionRanks);
constexpr int kWireExperts = kRanks * kLocalExperts;

// The launch shapes of the exchange, as DeepEP derives them from what the build states: the
// multiprocessors the dispatch and combine run on (the manifest's `exchange_multiprocessors`),
// every multiprocessor of the device for the epilogues (the descriptor's `multiprocessors`), the
// RDMA queue pairs the kernels address (the manifest's `queue_pairs`), and the shared memory a
// block of the target may opt into, in bytes.
constexpr int kSMs = UNISERVE_EXCHANGE_MULTIPROCESSORS;
constexpr int kFullSMs = UNISERVE_MULTIPROCESSORS;
constexpr int kQPs = UNISERVE_QUEUE_PAIRS;
constexpr int kSharedBytes = UNISERVE_SHARED_MEMORY;
static_assert(kSMs <= kFullSMs, "the exchange runs on the device's multiprocessors");

// DeepEP bounds each wait for a peer by a count of `clock64` cycles, after which it traps. A
// rank of this exchange waits without a bound of its own, as the alltoall backend's do: a peer
// that never arrives leaves the launch in flight, and the host detects it as it detects any job
// that stops progressing (the worker's watchdog reports the stalled job after half
// `worker_config.watchdog_timeout`, which fails the rank's group, and the group is replaced;
// `crates/engine/src/worker/watchdog.rs`). The largest count is one no wait reaches.
constexpr int64_t kTimeoutCycles = std::numeric_limits<int64_t>::max();
constexpr int kNotifyWarps = 4;
constexpr int kHiddenBytes = kHidden * 2;

constexpr int align_up(int value, int to) { return (value + to - 1) / to * to; }

/// `layout::TokenLayout::get_num_bytes`, which is no constant expression there.
constexpr int token_bytes(int metadata, bool barrier) {
  return align_up(kHiddenBytes, 32) + align_up(metadata, 32) + (barrier ? align_up(8, 32) : 0);
}
constexpr int kDispatchMetadata = kTop * 8 + (1 + kTop) * 4;
constexpr int kCombineMetadata = kTop * 8;
constexpr int kNotifySharedBytes = align_up(kRanks + kWireExperts, kNotifyWarps * 32) * 4;

constexpr int kDispatchWarps =
    std::min(std::min((kSharedBytes - kNotifySharedBytes) / token_bytes(kDispatchMetadata, true),
                      32 - kNotifyWarps),
             (512 + kSMs - 1) / kSMs);
constexpr int kDispatchThreads = (kNotifyWarps + kDispatchWarps) * 32;
constexpr int kCopyWarps = std::min(kSharedBytes / token_bytes(kDispatchMetadata, true), 32);
constexpr int kCombineWarps = std::min(kSharedBytes / token_bytes(kCombineMetadata, true), 32);
constexpr int kReduceWarps = std::min(kSharedBytes / token_bytes(0, false), 32);
constexpr int kBarrierThreads = 512;
constexpr int kCluster = 2 - (kSMs % 2);

#define UNISERVE_DISPATCH                                                                   \
  deep_ep::elastic::dispatch_impl<true, false, false, kSMs, kNotifyWarps, kDispatchWarps,  \
                                  kRanks, kHiddenBytes, 0, kMaxTokens, kWireExperts, kTop, \
                                  1, kQPs, kTimeoutCycles>
#define UNISERVE_COPY                                                                        \
  deep_ep::elastic::dispatch_copy_epilogue_impl<false, false, kFullSMs, 1, kCopyWarps, 1,   \
                                                kRanks, kHiddenBytes, 0, kMaxTokens,        \
                                                kWireExperts, kTop>
#define UNISERVE_COMBINE                                                                      \
  deep_ep::elastic::combine_impl<true, false, true, kSMs, kCombineWarps, kRanks, kHidden,    \
                                 kMaxTokens, kWireExperts, kTop, kQPs, kTimeoutCycles>
#define UNISERVE_REDUCE                                                                       \
  deep_ep::elastic::combine_reduce_epilogue_impl<false, true, kFullSMs, kReduceWarps, 1,     \
                                                 kRanks, kHidden, kMaxTokens, kWireExperts,  \
                                                 kTop>
#define UNISERVE_BARRIER \
  deep_ep::elastic::barrier_impl<true, 1, kBarrierThreads, 1, kRanks, kTimeoutCycles>

/// A created transport. A plain record: a launch reads it from another module of
/// this package, whose code is this file's.
struct Transport {
  int32_t rank;
  int64_t device;
  ncclComm_t comm;
  ncclDevComm_t dev_comm;
  ncclWindow_t window;
  void* raw_window;
  void* workspace;
  void* buffer;
  void* host_workspace;
  void* mapped_host_workspace;
  // This rank's scratch, sized once for the largest step.
  topk_idx_t* routes;        // [max_tokens, top] the routes it sends
  topk_idx_t* received;      // [received, top] the routes it received, as DeepEP states them
  int* slots;                // [max_tokens, top] DeepEP's destination slots
  int* rank_sums;            // [ranks] received rows, inclusive prefix sums
  int* expert_sums;          // [local + 1]
  int* metadata;             // [received, top + 2] where each received row came from
  // The views the layer reads.
  void* states;
  int32_t* ids;
  float* weights;
  void* combined;
  int32_t* live;
};

int32_t refuse(char* error, const char* message) {
  std::snprintf(error, 512, "%s", message);
  return 1;
}

int32_t nccl_refusal(char* error, const char* what, ncclResult_t status) {
  std::snprintf(error, 512, "expert_exchange: %s failed: %s", what, ncclGetErrorString(status));
  return 1;
}

int32_t cuda_refusal(char* error, const char* what, cudaError_t status) {
  std::snprintf(error, 512, "expert_exchange: %s failed: %s", what, cudaGetErrorString(status));
  return 1;
}

int64_t buffer_bytes() {
  const auto dispatch_token = layout::TokenLayout(kHiddenBytes, 0, kTop, true);
  const auto combine_token = layout::TokenLayout(kHiddenBytes, 0, kTop, false);
  // One guard slot beyond the largest step, as DeepEP's elastic metadata can use.
  const int rows = kMaxTokens + 1;
  const auto dispatch_recv = layout::BufferLayout<false>(dispatch_token, kRanks, rows);
  const int in_layout = std::min(kRanks, kTop);
  const auto combine_recv = layout::BufferLayout<false>(combine_token, in_layout, rows);
  return std::max<int64_t>(dispatch_recv.get_num_bytes(), combine_recv.get_num_bytes());
}

/// The bytes of this rank's window in the NCCL symmetric allocation: DeepEP's workspace, then
/// its buffer, in whole 16-byte vectors.
size_t window_bytes() {
  const size_t workspace_bytes = layout::WorkspaceLayout::get_num_bytes();
  return (workspace_bytes + static_cast<size_t>(buffer_bytes()) + 15) / 16 * 16;
}

/// One device allocation of a transport beside its window: the field it fills and its bytes.
struct Allocation {
  void** where;
  size_t bytes;
};

/// The device allocations of `transport` beside its window, sized once for the largest step.
std::array<Allocation, 11> allocations(Transport& transport) {
  constexpr size_t rows = kReceived;
  return {{
      {reinterpret_cast<void**>(&transport.routes), sizeof(topk_idx_t) * kMaxTokens * kTop},
      {reinterpret_cast<void**>(&transport.received), sizeof(topk_idx_t) * rows * kTop},
      {reinterpret_cast<void**>(&transport.slots), sizeof(int) * kMaxTokens * kTop},
      {reinterpret_cast<void**>(&transport.rank_sums), sizeof(int) * kRanks},
      {reinterpret_cast<void**>(&transport.expert_sums), sizeof(int) * (kLocalExperts + 1)},
      {reinterpret_cast<void**>(&transport.metadata), sizeof(int) * rows * (kTop + 2)},
      {&transport.states, rows * kHiddenBytes},
      {reinterpret_cast<void**>(&transport.ids), sizeof(int32_t) * rows * kTop},
      {reinterpret_cast<void**>(&transport.weights), sizeof(float) * rows * kTop},
      {&transport.combined, rows * kHiddenBytes},
      {reinterpret_cast<void**>(&transport.live), sizeof(int32_t)},
  }};
}

/// What creating the transport takes of the device beyond its window and its allocations: NCCL's
/// communicator and device communicator, and NCCL's rounding of the window to its allocation
/// granule. NCCL sizes them itself, from its configuration and the topology it finds, so this is
/// a measured bound: creating the transport over three ranks at 512 rows took 1.65 GB of the
/// device with NCCL 2.30.4, of which 32 MB were the window and the allocations.
constexpr int64_t kCommunicatorBytes = 2ll << 30;

/// Launches `kernel` with DeepEP's launch attributes.
template <typename Kernel, typename... Arguments>
cudaError_t launch(Kernel kernel, int blocks, int threads, int shared, int cluster,
                   bool cooperative, bool programmatic, cudaStream_t stream,
                   Arguments... arguments) {
  // The first launch of a kernel on a device raises its shared-memory limit. A warm
  // walk reaches every launch before a capture does, so a capture sets nothing.
  if (shared > 0) {
    const cudaError_t raised = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (raised != cudaSuccess) {
      return raised;
    }
  }
  cudaLaunchAttribute attributes[3];
  int count = 0;
  if (cluster > 1) {
    attributes[count].id = cudaLaunchAttributeClusterDimension;
    attributes[count].val.clusterDim = {static_cast<unsigned>(cluster), 1, 1};
    ++count;
  }
  if (cooperative) {
    attributes[count].id = cudaLaunchAttributeCooperative;
    attributes[count].val.cooperative = 1;
    ++count;
  }
  if (programmatic) {
    attributes[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[count].val.programmaticStreamSerializationAllowed = 1;
    ++count;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks, 1, 1);
  config.blockDim = dim3(threads, 1, 1);
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = count;
  void* parameters[] = {&arguments...};
  return cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), parameters);
}

/// This rank's routes as DeepEP reads them.
__global__ void route_kernel(topk_idx_t* routes, const int32_t* ids, const int32_t* live,
                             int32_t tokens, int32_t offset) {
  const int32_t count = live != nullptr ? min(*live, tokens) : tokens;
  const int64_t all = static_cast<int64_t>(tokens) * kTop;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; at < all;
       at += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    routes[at] = at / kTop < count ? static_cast<topk_idx_t>(ids[at]) + offset : -1;
  }
}

/// States how many rows arrived and rewrites their routes as vocabulary ids; a route
/// to another rank's expert names the vocabulary's end, which no sort window holds.
__global__ void received_kernel(int32_t* ids, const topk_idx_t* routes, const int* rank_sums,
                                int32_t* live, int32_t first, int32_t invalid) {
  const int count = rank_sums[kRanks - 1];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *live = count;
  }
  const int64_t all = static_cast<int64_t>(count) * kTop;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; at < all;
       at += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const auto route = routes[at];
    ids[at] = route >= 0 ? static_cast<int32_t>(route) + first : invalid;
  }
}

int32_t open(Transport& transport, const uniserve_exchange_lifecycle* call, char* error) {
  const UniserveGroup* group = call->group;
  transport.rank = static_cast<int32_t>(call->rank);
  transport.device = call->device;

  // Rank 0's id is the group's, and it travels over the host channel.
  ncclUniqueId mine{};
  if (call->rank == 0) {
    const ncclResult_t drawn = ncclGetUniqueId(&mine);
    if (drawn != ncclSuccess) {
      return nccl_refusal(error, "ncclGetUniqueId", drawn);
    }
  }
  ncclUniqueId ids[kRanks];
  if (group->all_gather(group->handle, &mine, ids, sizeof(ncclUniqueId)) != 0) {
    return refuse(error, "expert_exchange could not gather the communicator's id");
  }
  ncclResult_t status = ncclCommInitRank(&transport.comm, kRanks, ids[0], transport.rank);
  if (status != ncclSuccess) {
    return nccl_refusal(error, "ncclCommInitRank", status);
  }
  ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  status = ncclDevCommCreate(transport.comm, &requirements, &transport.dev_comm);
  if (status != ncclSuccess) {
    return nccl_refusal(error, "ncclDevCommCreate", status);
  }
  if (transport.dev_comm.lsaSize != kRanks || transport.dev_comm.lsaRank != transport.rank) {
    return refuse(error,
                  "expert_exchange's deepep backend serves one NVLink domain, and these ranks "
                  "span several");
  }
  const size_t workspace_bytes = layout::WorkspaceLayout::get_num_bytes();
  status = ncclMemAlloc(&transport.raw_window, window_bytes());
  if (status != ncclSuccess) {
    return nccl_refusal(error, "ncclMemAlloc", status);
  }
  status = ncclCommWindowRegister(transport.comm, transport.raw_window, window_bytes(),
                                  &transport.window, NCCL_WIN_DEFAULT);
  if (status != ncclSuccess) {
    return nccl_refusal(error, "ncclCommWindowRegister", status);
  }
  status = ncclGetLsaDevicePointer(transport.window, 0, transport.rank, &transport.workspace);
  if (status != ncclSuccess) {
    return nccl_refusal(error, "ncclGetLsaDevicePointer", status);
  }
  transport.buffer = static_cast<char*>(transport.workspace) + workspace_bytes;

  cudaError_t cuda = cudaMemset(transport.workspace, 0, workspace_bytes);
  if (cuda == cudaSuccess) {
    cuda = cudaHostAlloc(&transport.host_workspace, workspace_bytes, cudaHostAllocMapped);
  }
  if (cuda == cudaSuccess) {
    cuda = cudaHostGetDevicePointer(&transport.mapped_host_workspace, transport.host_workspace, 0);
  }
  if (cuda != cudaSuccess) {
    return cuda_refusal(error, "the workspace", cuda);
  }
  std::memset(transport.host_workspace, 0, workspace_bytes);

  for (const Allocation& allocation : allocations(transport)) {
    cuda = cudaMalloc(allocation.where, allocation.bytes);
    if (cuda != cudaSuccess) {
      return cuda_refusal(error, "the transport's storage", cuda);
    }
    cuda = cudaMemset(*allocation.where, 0, allocation.bytes);
    if (cuda != cudaSuccess) {
      return cuda_refusal(error, "clearing the transport's storage", cuda);
    }
  }
  cuda = cudaDeviceSynchronize();
  if (cuda != cudaSuccess) {
    return cuda_refusal(error, "the transport's setup", cuda);
  }
  // No rank's kernels touch a peer's workspace before every rank cleared its own.
  if (group->barrier(group->handle) != 0) {
    return refuse(error, "expert_exchange's ranks did not meet after clearing their storage");
  }
  cuda = launch(UNISERVE_BARRIER, 1, kBarrierThreads, 0, 1, true, false, cudaStream_t{nullptr},
                transport.dev_comm, transport.window, transport.workspace, 0, transport.rank);
  if (cuda == cudaSuccess) {
    cuda = cudaDeviceSynchronize();
  }
  if (cuda != cudaSuccess) {
    return cuda_refusal(error, "the first barrier", cuda);
  }
  return 0;
}

/// Refuses a lifecycle call of another group or shape than the module's; zero for its own.
int32_t check_shape(uniserve_exchange_lifecycle* call) {
  if (call->ranks != kRanks || call->attention_ranks != kAttentionRanks ||
      call->hidden != kHidden || call->top != kTop || call->experts != kExperts ||
      call->max_tokens != kMaxTokens || call->dtype != UNISERVE_DTYPE_BF16) {
    return refuse(call->error,
                  "expert_exchange's deepep module was compiled for another group or another "
                  "shape: its descriptor fixes them");
  }
  return 0;
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
  if (check_shape(call) != 0) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(std::calloc(1, sizeof(Transport)));
  if (transport == nullptr) {
    return refuse(call->error, "expert_exchange could not allocate its transport record");
  }
  const int32_t opened = open(*transport, call, call->error);
  if (opened != 0) {
    // A failed creation leaves the peers without this rank, so the process ends and
    // the driver takes the rest back.
    std::free(transport);
    return opened;
  }
  call->transport = transport;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_footprint(uniserve_exchange_lifecycle* call,
                                                      int64_t* bytes) {
  if (call == nullptr || bytes == nullptr || check_shape(call) != 0) {
    return 1;
  }
  // The sizes of the allocations do not depend on the record they fill.
  Transport sized{};
  size_t allocated = window_bytes();
  for (const Allocation& allocation : allocations(sized)) {
    allocated += allocation.bytes;
  }
  *bytes = static_cast<int64_t>(allocated) + kCommunicatorBytes;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_step(uniserve_exchange_lifecycle* call, void* stream) {
  (void)stream;
  // DeepEP packs what arrives and counts it on the device, so no capacity reaches it.
  return call == nullptr || call->transport == nullptr ? 1 : 0;
}

extern "C" int32_t uniserve_expert_exchange_suspend(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  return call == nullptr ? 1
                         : refuse(call->error,
                                  "the deepep exchange's windows cannot keep their addresses, so "
                                  "its transport does not suspend");
}

extern "C" int32_t uniserve_expert_exchange_restore(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  return call == nullptr ? 1
                         : refuse(call->error, "the deepep exchange does not suspend");
}

extern "C" int32_t uniserve_expert_exchange_release(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  // A local drain does not retire a peer's remote reads, so every rank reaches this
  // point before any rank's window goes.
  cudaDeviceSynchronize();
  if (call->group != nullptr) {
    call->group->barrier(call->group->handle);
  }
  ncclCommWindowDeregister(transport->comm, transport->window);
  ncclMemFree(transport->raw_window);
  ncclDevCommDestroy(transport->comm, &transport->dev_comm);
  ncclCommAbort(transport->comm);
  cudaFreeHost(transport->host_workspace);
  void* owned[] = {transport->routes,      transport->received, transport->slots,
                   transport->rank_sums,   transport->expert_sums, transport->metadata,
                   transport->states,      transport->ids,      transport->weights,
                   transport->combined,    transport->live};
  for (void* allocation : owned) {
    cudaFree(allocation);
  }
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
  call->states = transport->states;
  call->ids = transport->ids;
  call->weights = transport->weights;
  call->combined = transport->combined;
  call->live = transport->live;
  call->ranks = kRanks;
  call->rank = transport->rank;
  call->capacity = kMaxTokens;
  call->hidden = kHidden;
  call->top = kTop;
  call->dtype = UNISERVE_DTYPE_BF16;
  call->device = transport->device;
  call->fused = 0;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_launch(uniserve_exchange_call* call, void* stream) {
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  auto* raw = reinterpret_cast<cudaStream_t>(stream);
  const auto tokens = static_cast<int>(call->tokens);
  if (tokens > kMaxTokens) {
    return refuse(call->error, "expert_exchange's rows exceed the transport's capacity");
  }
  int rank = transport->rank;
  cudaError_t status = cudaSuccess;
  if (call->mode == UNISERVE_EXCHANGE_DISPATCH) {
    if (tokens > 0) {
      route_kernel<<<std::min(64, (tokens * kTop + 255) / 256), 256, 0, raw>>>(
          transport->routes, call->ids, call->live, tokens, kAttentionRanks * kLocalExperts);
    }
    void* no_scales = nullptr;
    status = launch(
        UNISERVE_DISPATCH, kSMs, kDispatchThreads, kSharedBytes, kCluster, true, false, raw,
        const_cast<void*>(call->states), static_cast<deep_ep::sf_pack_t*>(no_scales),
        transport->routes, const_cast<float*>(call->weights), static_cast<topk_idx_t*>(nullptr),
        static_cast<int*>(nullptr), transport->rank_sums, transport->expert_sums, transport->slots,
        tokens, 0, 0, transport->dev_comm, transport->window, transport->buffer,
        transport->workspace, transport->mapped_host_workspace, rank);
    if (status == cudaSuccess) {
      status = launch(UNISERVE_COPY, kFullSMs, kCopyWarps * 32, kSharedBytes, 1, false, true, raw,
                      transport->buffer, transport->workspace, transport->rank_sums,
                      transport->expert_sums + 1, transport->states,
                      static_cast<deep_ep::sf_pack_t*>(no_scales), transport->received,
                      transport->weights, transport->metadata, static_cast<int*>(nullptr),
                      kReceived, 0, 0, 0, rank);
    }
    received_kernel<<<64, 256, 0, raw>>>(transport->ids, transport->received,
                                         transport->rank_sums, transport->live,
                                         (rank - kAttentionRanks) * kLocalExperts, kExperts);
  } else if (call->mode == UNISERVE_EXCHANGE_COMBINE) {
    status = launch(UNISERVE_COMBINE, kSMs, kCombineWarps * 32, kSharedBytes, kCluster, true,
                    false, raw, static_cast<nv_bfloat16*>(transport->combined),
                    static_cast<float*>(nullptr), transport->metadata, transport->rank_sums,
                    transport->dev_comm, transport->window, transport->buffer,
                    transport->workspace, rank, kReceived);
    if (status == cudaSuccess) {
      void* no_bias = nullptr;
      status = launch(UNISERVE_REDUCE, kFullSMs, kReduceWarps * 32, kSharedBytes, 1, false, true,
                      raw, static_cast<nv_bfloat16*>(call->out), static_cast<float*>(nullptr),
                      transport->routes, transport->buffer, no_bias, no_bias, tokens, 0, rank);
    }
  } else {
    return refuse(call->error, "expert_exchange's deepep backend runs no fused launch");
  }
  if (status == cudaSuccess) {
    status = cudaGetLastError();
  }
  return status == cudaSuccess ? 0 : cuda_refusal(call->error, "the exchange's launch", status);
}
