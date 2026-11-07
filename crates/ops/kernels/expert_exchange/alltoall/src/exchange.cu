// SPDX-License-Identifier: Apache-2.0
//
// The alltoall backend of expert_exchange: every rank maps every rank's workspace
// and moves rows with plain stores and loads over NVLink.
//
// Each rank owns one workspace, mapped through the CUDA virtual-memory API with a
// handle every peer imports (a fabric handle on a multi-node NVLink domain, an
// in-process file descriptor otherwise). A dispatch writes each token once into
// the workspace of every rank that holds one of its experts, at
// `source rank * step capacity + slot`. A combine reads each token's partial sum
// back from those ranks and adds them in FP32, rounding once. This is the data
// flow of FlashInfer's `MoeAlltoAll`, with one change: the step's capacity is a
// device word the executor writes before a replay, so one captured graph serves
// every capacity the ranks agree on, and the agreement rounds nothing up.
//
// The ranks of a group run the same sequence of launches, so a phase number they
// all count is the value a rank waits to see in its flags: a flag is written with
// a release store of system scope and read with an acquire load of system scope.
//
// Row safety across launches needs no further flag. A rank starts layer `L + 1`'s
// dispatch only after its layer `L` combine saw every peer's flag, and a peer sets
// that flag only after its experts finished reading the rows it received, so a
// dispatch never overwrites rows a peer is reading. The same argument holds for
// `combined`: a rank writes it for layer `L + 1` only after that layer's dispatch
// saw every peer, and a peer dispatches only after it finished reading layer `L`.

#include "../include/uniserve_expert_exchange.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kMaxRanks = 64;
constexpr int kMaxTop = 32;
constexpr int kThreads = 256;
/// The most blocks a launch takes: every one of them fits on the device at once,
/// so a block that waits for a flag never keeps out the block that sets it.
constexpr int kMaxBlocks = 256;
/// How long a block waits for a peer before it stops the device, in nanoseconds.
/// A rank can lag by a module load or a first-use plan during warmup, so the bound
/// is minutes; an unbounded wait would turn a lost peer into a silent hang.
constexpr unsigned long long kPeerTimeoutNs = 600ull * 1000 * 1000 * 1000;

/// The bytes of one row of `hidden` elements of `dtype`, or -1 for a type that
/// is not an activation.
__host__ __device__ int64_t row_bytes(int64_t dtype, int64_t hidden) {
  switch (dtype) {
    case UNISERVE_DTYPE_F16:
    case UNISERVE_DTYPE_BF16:
      return 2 * hidden;
    case UNISERVE_DTYPE_F32:
      return 4 * hidden;
    default:
      return -1;
  }
}

size_t aligned(size_t bytes, size_t to = 256) { return (bytes + to - 1) / to * to; }

/// Offsets of one rank's workspace, the same on every rank. What peers write comes
/// first; the rest only this rank touches, but one mapping holds all of it.
struct Layout {
  size_t dispatch_flags;  // [ranks] u32: the phase each source's dispatch reached
  size_t combine_flags;   // [ranks] u32: the phase each peer's combined rows reached
  size_t received;        // [ranks] i32: rows each source sent here this step
  size_t states;          // [ranks * capacity, row]
  size_t ids;             // [ranks * capacity, top] i32
  size_t weights;         // [ranks * capacity, top] f32
  size_t combined;        // [ranks * capacity, row]
  size_t step;            // i32: the rows per rank this step
  size_t live;            // i32: ranks * step, the leading rows the views hold
  size_t phase;           // u32: the phase the last combine finished
  size_t sent;            // [ranks] i32: rows this rank sent to each peer this step
  size_t done;            // u32: blocks of a launch that finished
  size_t targets;         // [capacity, top] i32: the rank each route's token went to, or -1
  size_t slots;           // [capacity, top] i32: the slot it landed in
  size_t bytes;
};

Layout layout_of(int64_t ranks, int64_t capacity, int64_t row, int64_t top) {
  Layout layout{};
  size_t at = 0;
  auto take = [&at](size_t bytes) {
    const size_t here = at;
    at = aligned(at + bytes);
    return here;
  };
  const size_t rows = static_cast<size_t>(ranks) * capacity;
  layout.dispatch_flags = take(ranks * 4);
  layout.combine_flags = take(ranks * 4);
  layout.received = take(ranks * 4);
  layout.states = take(rows * row);
  layout.ids = take(rows * top * 4);
  layout.weights = take(rows * top * 4);
  layout.combined = take(rows * row);
  layout.step = take(4);
  layout.live = take(4);
  layout.phase = take(4);
  layout.sent = take(ranks * 4);
  layout.done = take(4);
  layout.targets = take(capacity * top * 4);
  layout.slots = take(capacity * top * 4);
  layout.bytes = at;
  return layout;
}

/// What a launch reads: the workspace addresses of every rank, as this rank maps
/// them, and the extents.
struct Peers {
  char* base[kMaxRanks];
};

/// A created transport.
struct Transport {
  int32_t rank;
  int32_t ranks;
  int32_t capacity;
  int32_t hidden;
  int32_t top;
  int32_t experts;
  int64_t dtype;
  int64_t device;
  int64_t row;
  Layout layout;
  CUdeviceptr base;    // `ranks` windows of `window` bytes, this rank's at `rank * window`
  size_t window;
  bool mapped;
  CUmemAllocationHandleType kind;
  CUmemGenericAllocationHandle own;
  CUmemGenericAllocationHandle imported[kMaxRanks];
};

char* own_base(const Transport& transport) {
  return reinterpret_cast<char*>(transport.base) + transport.rank * transport.window;
}

Peers peers_of(const Transport& transport) {
  Peers peers{};
  for (int rank = 0; rank < transport.ranks; ++rank) {
    peers.base[rank] = reinterpret_cast<char*>(transport.base) + rank * transport.window;
  }
  return peers;
}

/// Writes `message` into an entry's error buffer and answers its status.
int32_t refuse(char* error, const char* message) {
  std::snprintf(error, 512, "%s", message);
  return 1;
}

/// Writes a driver call's failure into an entry's error buffer.
int32_t driver_refusal(char* error, const char* what, CUresult status) {
  const char* name = nullptr;
  cuGetErrorName(status, &name);
  std::snprintf(error, 512, "expert_exchange: %s failed: %s", what, name ? name : "unknown");
  return 1;
}

CUmemAllocationProp allocation_prop(int device, CUmemAllocationHandleType kind) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = kind;
  return prop;
}

/// A shareable handle as it crosses the group, whichever kind it is.
struct Shared {
  unsigned char bytes[64];
};

/// Creates this rank's storage and maps every rank's at the transport's addresses.
///
/// A fabric handle is the NVLink domain's own: it needs no process relationship,
/// so it serves ranks on any host of the domain. Where the driver has no fabric, a
/// file descriptor serves the ranks of one process, which is what a main process
/// is. The ranks must all take the same kind, so each states what it could do.
int32_t map_storage(Transport& transport, const UniserveGroup* group, char* error) {
  const int ranks = transport.ranks;
  CUresult status = CUDA_SUCCESS;

  size_t granularity = 0;
  CUmemAllocationProp prop = allocation_prop(static_cast<int>(transport.device),
                                             CU_MEM_HANDLE_TYPE_FABRIC);
  status = cuMemGetAllocationGranularity(&granularity, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuMemGetAllocationGranularity", status);
  }
  transport.window = aligned(transport.layout.bytes, granularity);

  // Fabric first; a driver or a container without it answers at the allocation.
  CUmemAllocationHandleType kind = CU_MEM_HANDLE_TYPE_FABRIC;
  CUmemGenericAllocationHandle own{};
  status = cuMemCreate(&own, transport.window, &prop, 0);
  Shared mine{};
  if (status == CUDA_SUCCESS) {
    status = cuMemExportToShareableHandle(mine.bytes, own, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (status != CUDA_SUCCESS) {
      cuMemRelease(own);
    }
  }
  if (status != CUDA_SUCCESS) {
    kind = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop = allocation_prop(static_cast<int>(transport.device), kind);
    status = cuMemCreate(&own, transport.window, &prop, 0);
    if (status != CUDA_SUCCESS) {
      return driver_refusal(error, "cuMemCreate", status);
    }
    int descriptor = -1;
    status = cuMemExportToShareableHandle(&descriptor, own, kind, 0);
    if (status != CUDA_SUCCESS) {
      cuMemRelease(own);
      return driver_refusal(error, "cuMemExportToShareableHandle", status);
    }
    std::memcpy(mine.bytes, &descriptor, sizeof(descriptor));
  }
  transport.kind = kind;
  transport.own = own;

  // The kind travels with the handle: ranks that disagree cannot import each other.
  struct Record {
    Shared handle;
    int64_t kind;
  };
  Record record{mine, static_cast<int64_t>(kind)};
  auto* records = static_cast<Record*>(std::calloc(ranks, sizeof(Record)));
  if (records == nullptr) {
    return refuse(error, "expert_exchange could not hold its peers' handles");
  }
  if (group->all_gather(group->handle, &record, records, sizeof(Record)) != 0) {
    std::free(records);
    return refuse(error, "expert_exchange could not gather its peers' handles");
  }

  if (transport.base == 0) {
    status = cuMemAddressReserve(&transport.base, transport.window * ranks, granularity, 0, 0);
    if (status != CUDA_SUCCESS) {
      std::free(records);
      return driver_refusal(error, "cuMemAddressReserve", status);
    }
  }
  for (int rank = 0; rank < ranks; ++rank) {
    if (records[rank].kind != static_cast<int64_t>(kind)) {
      std::free(records);
      return refuse(error, "expert_exchange's ranks share no handle kind");
    }
    CUmemGenericAllocationHandle mapped = own;
    if (rank != transport.rank) {
      void* shared = nullptr;
      int descriptor = -1;
      if (kind == CU_MEM_HANDLE_TYPE_FABRIC) {
        shared = records[rank].handle.bytes;
      } else {
        std::memcpy(&descriptor, records[rank].handle.bytes, sizeof(descriptor));
        shared = reinterpret_cast<void*>(static_cast<intptr_t>(descriptor));
      }
      status = cuMemImportFromShareableHandle(&mapped, shared, kind);
      if (status != CUDA_SUCCESS) {
        std::free(records);
        return driver_refusal(error, "cuMemImportFromShareableHandle", status);
      }
      transport.imported[rank] = mapped;
    }
    status = cuMemMap(transport.base + rank * transport.window, transport.window, 0, mapped, 0);
    if (status != CUDA_SUCCESS) {
      std::free(records);
      return driver_refusal(error, "cuMemMap", status);
    }
  }
  std::free(records);
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(transport.device);
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  status = cuMemSetAccess(transport.base, transport.window * ranks, &access, 1);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuMemSetAccess", status);
  }
  transport.mapped = true;

  // A peer's first flag must land in zeroed storage, so no rank writes before
  // every rank cleared its own.
  status = cuMemsetD8(reinterpret_cast<CUdeviceptr>(own_base(transport)), 0, transport.window);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuMemsetD8", status);
  }
  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuCtxSynchronize", status);
  }
  if (group->barrier(group->handle) != 0) {
    return refuse(error, "expert_exchange's ranks did not meet after clearing their storage");
  }
  return 0;
}

/// Unmaps every rank's storage and gives this rank's back, keeping the addresses.
void unmap_storage(Transport& transport) {
  if (!transport.mapped) {
    return;
  }
  for (int rank = 0; rank < transport.ranks; ++rank) {
    cuMemUnmap(transport.base + rank * transport.window, transport.window);
    if (rank != transport.rank) {
      cuMemRelease(transport.imported[rank]);
    }
  }
  cuMemRelease(transport.own);
  transport.mapped = false;
}

__device__ __forceinline__ void store_release(uint32_t* flag, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(flag), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire(const uint32_t* flag) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(flag) : "memory");
  return value;
}

/// Waits until every rank's flag in `flags` reached `phase`, or stops the device
/// when a peer never does: a trap is a failure the executor sees, a hang is not.
__device__ void wait_for_peers(const uint32_t* flags, int ranks, uint32_t phase) {
  const unsigned long long start = [] {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
  }();
  for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x) {
    // Phases only grow, and the difference stays correct across a wrap.
    while (static_cast<int32_t>(load_acquire(flags + rank) - phase) < 0) {
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (now - start > kPeerTimeoutNs) {
        __trap();
      }
    }
  }
}

/// Copies `bytes` bytes, a multiple of 16, from `from` with every thread of a block.
__device__ __forceinline__ void copy_row(char* to, const char* from, int64_t bytes) {
  for (int64_t at = static_cast<int64_t>(threadIdx.x) * 16; at < bytes;
       at += static_cast<int64_t>(blockDim.x) * 16) {
    *reinterpret_cast<int4*>(to + at) = *reinterpret_cast<const int4*>(from + at);
  }
}

struct DispatchParams {
  Peers peers;
  Layout layout;
  const char* states;
  const int32_t* ids;
  const float* weights;
  const int32_t* live;
  int32_t rank;
  int32_t ranks;
  int32_t tokens;
  int32_t top;
  int32_t experts;
  int64_t row;
};

/// Sends every live token once to each rank that holds one of its experts.
///
/// A block takes tokens in turn. Its first threads name the ranks a token goes to
/// and claim a slot in each; then the whole block stores the row into every target.
/// The last block to finish publishes the counts and waits for every peer's.
__global__ void __launch_bounds__(kThreads) dispatch_kernel(const DispatchParams p) {
  __shared__ int32_t target_slot[kMaxRanks];
  __shared__ bool is_last;
  char* own = p.peers.base[p.rank];
  const int32_t step = *reinterpret_cast<const int32_t*>(own + p.layout.step);
  const uint32_t phase = *reinterpret_cast<const uint32_t*>(own + p.layout.phase) + 1;
  auto* sent = reinterpret_cast<int32_t*>(own + p.layout.sent);
  auto* targets = reinterpret_cast<int32_t*>(own + p.layout.targets);
  auto* slots = reinterpret_cast<int32_t*>(own + p.layout.slots);
  const int32_t tokens = p.live != nullptr ? min(*p.live, p.tokens) : p.tokens;
  const int per_rank = p.experts / p.ranks;

  for (int32_t token = blockIdx.x; token < tokens; token += gridDim.x) {
    for (int rank = threadIdx.x; rank < p.ranks; rank += blockDim.x) {
      target_slot[rank] = -1;
    }
    __syncthreads();
    // One route per thread: the first route of each target claims the target's slot.
    if (threadIdx.x < p.top) {
      const int route = threadIdx.x;
      const int target = p.ids[static_cast<int64_t>(token) * p.top + route] / per_rank;
      bool first = true;
      for (int earlier = 0; earlier < route; ++earlier) {
        first = first &&
                p.ids[static_cast<int64_t>(token) * p.top + earlier] / per_rank != target;
      }
      int32_t slot = -1;
      if (first) {
        slot = atomicAdd(sent + target, 1);
        target_slot[target] = slot;
      }
      targets[static_cast<int64_t>(token) * p.top + route] = first ? target : -1;
      slots[static_cast<int64_t>(token) * p.top + route] = slot;
    }
    __syncthreads();
    for (int target = 0; target < p.ranks; ++target) {
      const int32_t slot = target_slot[target];
      if (slot < 0) {
        continue;
      }
      const int64_t at = static_cast<int64_t>(p.rank) * step + slot;
      char* peer = p.peers.base[target];
      copy_row(peer + p.layout.states + at * p.row, p.states + token * p.row, p.row);
      if (threadIdx.x < p.top) {
        const int64_t route = at * p.top + threadIdx.x;
        reinterpret_cast<int32_t*>(peer + p.layout.ids)[route] =
            p.ids[static_cast<int64_t>(token) * p.top + threadIdx.x];
        reinterpret_cast<float*>(peer + p.layout.weights)[route] =
            p.weights[static_cast<int64_t>(token) * p.top + threadIdx.x];
      }
    }
    __syncthreads();
  }

  // Every store of this block reaches its peer before the block counts as finished.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    auto* done = reinterpret_cast<uint32_t*>(own + p.layout.done);
    is_last = atomicAdd(done, 1) + 1 == gridDim.x;
  }
  __syncthreads();
  if (!is_last) {
    return;
  }
  __threadfence();
  for (int target = threadIdx.x; target < p.ranks; target += blockDim.x) {
    char* peer = p.peers.base[target];
    reinterpret_cast<int32_t*>(peer + p.layout.received)[p.rank] = sent[target];
    __threadfence_system();
    store_release(reinterpret_cast<uint32_t*>(peer + p.layout.dispatch_flags) + p.rank, phase);
    sent[target] = 0;
  }
  if (threadIdx.x == 0) {
    *reinterpret_cast<uint32_t*>(own + p.layout.done) = 0;
  }
  wait_for_peers(reinterpret_cast<const uint32_t*>(own + p.layout.dispatch_flags), p.ranks, phase);
}

/// Marks the rows no source filled: a row past a source's count holds what an
/// earlier step left there, and a route of it would run an expert for nothing.
__global__ void __launch_bounds__(kThreads) vacate_kernel(const Peers peers, const Layout layout,
                                                          int32_t rank, int32_t ranks,
                                                          int32_t top, int32_t invalid) {
  char* own = peers.base[rank];
  const int32_t step = *reinterpret_cast<const int32_t*>(own + layout.step);
  const auto* received = reinterpret_cast<const int32_t*>(own + layout.received);
  auto* ids = reinterpret_cast<int32_t*>(own + layout.ids);
  const int64_t routes = static_cast<int64_t>(ranks) * step * top;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; at < routes;
       at += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = at / top;
    if (row % step >= received[row / step]) {
      ids[at] = invalid;
    }
  }
}

struct CombineParams {
  Peers peers;
  Layout layout;
  void* out;
  const int32_t* live;
  int32_t rank;
  int32_t ranks;
  int32_t tokens;
  int32_t top;
  int64_t dtype;
  int64_t row;
  int32_t hidden;
};

template <typename T>
__device__ __forceinline__ float widen(T value);
template <>
__device__ __forceinline__ float widen(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <>
__device__ __forceinline__ float widen(__half value) {
  return __half2float(value);
}
template <>
__device__ __forceinline__ float widen(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T narrow(float value);
template <>
__device__ __forceinline__ __nv_bfloat16 narrow(float value) {
  return __float2bfloat16(value);
}
template <>
__device__ __forceinline__ __half narrow(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ float narrow(float value) {
  return value;
}

/// Sums each token's partial rows from the ranks that computed them, in FP32, and
/// rounds once. Eight elements per thread of BF16 or FP16 is one 16-byte load.
template <typename T>
__global__ void __launch_bounds__(kThreads) combine_kernel(const CombineParams p) {
  constexpr int kVector = 16 / sizeof(T);
  __shared__ const char* sources[kMaxTop];
  __shared__ int count;
  __shared__ bool is_last;
  char* own = p.peers.base[p.rank];
  const int32_t step = *reinterpret_cast<const int32_t*>(own + p.layout.step);
  const uint32_t phase = *reinterpret_cast<const uint32_t*>(own + p.layout.phase) + 1;
  const auto* targets = reinterpret_cast<const int32_t*>(own + p.layout.targets);
  const auto* slots = reinterpret_cast<const int32_t*>(own + p.layout.slots);
  const int32_t tokens = p.live != nullptr ? min(*p.live, p.tokens) : p.tokens;

  // The expert kernels ended before this launch began, so this rank's combined rows
  // are complete: its first block tells every peer so.
  if (blockIdx.x == 0) {
    for (int target = threadIdx.x; target < p.ranks; target += blockDim.x) {
      store_release(reinterpret_cast<uint32_t*>(p.peers.base[target] + p.layout.combine_flags) +
                        p.rank,
                    phase);
    }
  }
  wait_for_peers(reinterpret_cast<const uint32_t*>(own + p.layout.combine_flags), p.ranks, phase);
  __syncthreads();

  for (int32_t token = blockIdx.x; token < p.tokens; token += gridDim.x) {
    T* out = static_cast<T*>(p.out) + static_cast<int64_t>(token) * p.hidden;
    if (threadIdx.x == 0) {
      int found = 0;
      if (token < tokens) {
        for (int route = 0; route < p.top; ++route) {
          const int target = targets[static_cast<int64_t>(token) * p.top + route];
          if (target >= 0) {
            const int64_t at = static_cast<int64_t>(p.rank) * step +
                               slots[static_cast<int64_t>(token) * p.top + route];
            sources[found++] = p.peers.base[target] + p.layout.combined + at * p.row;
          }
        }
      }
      count = found;
    }
    __syncthreads();
    for (int at = threadIdx.x * kVector; at < p.hidden; at += blockDim.x * kVector) {
      float sum[kVector];
#pragma unroll
      for (int lane = 0; lane < kVector; ++lane) {
        sum[lane] = 0.0f;
      }
      for (int source = 0; source < count; ++source) {
        const int4 packed = *reinterpret_cast<const int4*>(
            sources[source] + static_cast<int64_t>(at) * sizeof(T));
        const T* values = reinterpret_cast<const T*>(&packed);
#pragma unroll
        for (int lane = 0; lane < kVector; ++lane) {
          sum[lane] += widen<T>(values[lane]);
        }
      }
      int4 packed;
      T* values = reinterpret_cast<T*>(&packed);
#pragma unroll
      for (int lane = 0; lane < kVector; ++lane) {
        values[lane] = narrow<T>(sum[lane]);
      }
      *reinterpret_cast<int4*>(out + at) = packed;
    }
    __syncthreads();
  }

  // The last block moves the phase on, once no block reads it any more.
  __syncthreads();
  if (threadIdx.x == 0) {
    auto* done = reinterpret_cast<uint32_t*>(own + p.layout.done);
    is_last = atomicAdd(done, 1) + 1 == gridDim.x;
    if (is_last) {
      *done = 0;
      *reinterpret_cast<uint32_t*>(own + p.layout.phase) = phase;
    }
  }
}

/// The grid of one launch: a block per row, up to what the device holds at once.
int blocks_of(int64_t rows) {
  if (rows <= 0) {
    return 1;
  }
  return static_cast<int>(rows < kMaxBlocks ? rows : kMaxBlocks);
}

}  // namespace

extern "C" int32_t uniserve_expert_exchange_create(uniserve_exchange_lifecycle* call,
                                                   void* stream) {
  (void)stream;
  if (call == nullptr) {
    return 1;
  }
  call->transport = nullptr;
  const int64_t row = row_bytes(call->dtype, call->hidden);
  if (row <= 0 || row % 16 != 0 || call->hidden <= 0 || call->top <= 0 || call->max_tokens <= 0) {
    return refuse(call->error,
                  "expert_exchange needs positive hidden, top and max_tokens, a row of whole "
                  "16-byte vectors, and an activation dtype");
  }
  if (call->ranks < 1 || call->ranks > kMaxRanks || call->top > kMaxTop ||
      call->experts % call->ranks != 0) {
    return refuse(call->error,
                  "expert_exchange serves at most 64 ranks and 32 routes per token, over a "
                  "vocabulary the ranks divide evenly");
  }
  if (call->group == nullptr) {
    return refuse(call->error, "expert_exchange's create needs the group's channel");
  }
  auto* transport = static_cast<Transport*>(std::calloc(1, sizeof(Transport)));
  if (transport == nullptr) {
    return refuse(call->error, "expert_exchange could not allocate its transport record");
  }
  transport->rank = static_cast<int32_t>(call->rank);
  transport->ranks = static_cast<int32_t>(call->ranks);
  transport->capacity = static_cast<int32_t>(call->max_tokens);
  transport->hidden = static_cast<int32_t>(call->hidden);
  transport->top = static_cast<int32_t>(call->top);
  transport->experts = static_cast<int32_t>(call->experts);
  transport->dtype = call->dtype;
  transport->device = call->device;
  transport->row = row;
  transport->layout = layout_of(call->ranks, call->max_tokens, row, call->top);
  const int32_t mapped = map_storage(*transport, call->group, call->error);
  if (mapped != 0) {
    unmap_storage(*transport);
    if (transport->base != 0) {
      cuMemAddressFree(transport->base, transport->window * transport->ranks);
    }
    std::free(transport);
    return mapped;
  }
  call->transport = transport;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_step(uniserve_exchange_lifecycle* call, void* stream) {
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  const auto* transport = static_cast<const Transport*>(call->transport);
  if (call->capacity < 0 || call->capacity > transport->capacity) {
    return refuse(call->error, "the step's capacity exceeds the transport's");
  }
  char* own = own_base(*transport);
  const auto capacity = static_cast<unsigned int>(call->capacity);
  auto* raw = reinterpret_cast<CUstream>(stream);
  CUresult status = cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(own + transport->layout.step),
                                     capacity, 1, raw);
  if (status == CUDA_SUCCESS) {
    status = cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(own + transport->layout.live),
                              capacity * static_cast<unsigned int>(transport->ranks), 1, raw);
  }
  if (status != CUDA_SUCCESS) {
    return driver_refusal(call->error, "the step's capacity write", status);
  }
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_suspend(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  if (call->group == nullptr || call->group->barrier(call->group->handle) != 0) {
    return refuse(call->error, "expert_exchange's ranks did not meet before suspending");
  }
  // The storage goes; the addresses stay reserved, so a captured graph keeps
  // addressing them and `restore` maps new storage where the graph expects it.
  unmap_storage(*transport);
  return call->group->barrier(call->group->handle) == 0
             ? 0
             : refuse(call->error, "expert_exchange's ranks did not meet after suspending");
}

extern "C" int32_t uniserve_expert_exchange_restore(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr || call->group == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  if (transport->mapped) {
    return 0;
  }
  return map_storage(*transport, call->group, call->error);
}

extern "C" int32_t uniserve_expert_exchange_release(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  // A local stream drain does not retire a peer's remote reads, so every rank
  // reaches this point before any rank's storage goes.
  if (call->group != nullptr) {
    call->group->barrier(call->group->handle);
  }
  unmap_storage(*transport);
  if (transport->base != 0) {
    cuMemAddressFree(transport->base, transport->window * transport->ranks);
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
  char* own = own_base(*transport);
  call->states = own + transport->layout.states;
  call->ids = own + transport->layout.ids;
  call->weights = own + transport->layout.weights;
  call->combined = own + transport->layout.combined;
  call->live = own + transport->layout.live;
  call->ranks = transport->ranks;
  call->rank = transport->rank;
  call->capacity = transport->capacity;
  call->hidden = transport->hidden;
  call->top = transport->top;
  call->dtype = transport->dtype;
  call->device = transport->device;
  call->fused = 0;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_launch(uniserve_exchange_call* call, void* stream) {
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  const auto* transport = static_cast<const Transport*>(call->transport);
  if (call->ranks != transport->ranks) {
    return refuse(call->error,
                  "expert_exchange's launch names another group size than its transport");
  }
  if (call->tokens > transport->capacity) {
    return refuse(call->error, "expert_exchange's rows exceed the transport's capacity");
  }
  auto* raw = reinterpret_cast<cudaStream_t>(stream);
  const Peers peers = peers_of(*transport);
  const int blocks = blocks_of(call->tokens);
  if (call->mode == UNISERVE_EXCHANGE_DISPATCH) {
    const DispatchParams params{peers,
                                transport->layout,
                                static_cast<const char*>(call->states),
                                call->ids,
                                call->weights,
                                call->live,
                                transport->rank,
                                transport->ranks,
                                static_cast<int32_t>(call->tokens),
                                transport->top,
                                transport->experts,
                                transport->row};
    dispatch_kernel<<<blocks, kThreads, 0, raw>>>(params);
    vacate_kernel<<<64, kThreads, 0, raw>>>(peers, transport->layout, transport->rank,
                                            transport->ranks, transport->top, transport->experts);
  } else if (call->mode == UNISERVE_EXCHANGE_COMBINE) {
    const CombineParams params{peers,
                               transport->layout,
                               call->out,
                               call->live,
                               transport->rank,
                               transport->ranks,
                               static_cast<int32_t>(call->tokens),
                               transport->top,
                               transport->dtype,
                               transport->row,
                               transport->hidden};
    switch (transport->dtype) {
      case UNISERVE_DTYPE_BF16:
        combine_kernel<__nv_bfloat16><<<blocks, kThreads, 0, raw>>>(params);
        break;
      case UNISERVE_DTYPE_F16:
        combine_kernel<__half><<<blocks, kThreads, 0, raw>>>(params);
        break;
      default:
        combine_kernel<float><<<blocks, kThreads, 0, raw>>>(params);
        break;
    }
  } else {
    return refuse(call->error, "expert_exchange's alltoall backend runs no fused launch");
  }
  if (cudaGetLastError() != cudaSuccess) {
    return refuse(call->error, "expert_exchange could not start its kernels");
  }
  return 0;
}
