// SPDX-License-Identifier: Apache-2.0
//
// The expert_exchange module for one expert group: the transport's lifecycle and
// the two directions of the exchange.
//
// The transport is a device workspace, allocated through the CUDA virtual-memory
// API so `suspend` can return its storage while keeping its virtual addresses
// reserved — a graph captured over it stays valid, and `restore` maps fresh
// storage at the same addresses (notes/programs/attention_moe.md, "Transport
// lifecycle").
//
// This module serves a group of one rank: the exchange of an expert-parallel
// deployment whose experts all live on the launching rank (EP=1). Dispatch packs
// the rank's own rows into its slot of the workspace with the route ids and
// weights its experts read; combine sums the one partial each row gets back. A
// group of several ranks needs the peer-mapping transport (MNNVL/nvlink) whose
// handshake is the package's host code; until that exists the entries refuse a
// group they cannot drive rather than pretend.

#include "../include/uniserve_expert_exchange.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

/// The bytes of one row of `hidden` elements of `dtype`, or -1 for a type that
/// is not an activation.
__host__ __device__ int64_t row_bytes(int64_t dtype, int64_t hidden) {
  switch (dtype) {
    case UNISERVE_DTYPE_F16:
    case UNISERVE_DTYPE_BF16:
      return 2 * hidden;
    case UNISERVE_DTYPE_F32:
      return 4 * hidden;
    case UNISERVE_DTYPE_F8E4M3:
      return hidden;
    case UNISERVE_DTYPE_F4E2M1:
      return (hidden + 1) / 2;
    default:
      return -1;
  }
}

/// Rounds a size up to the 256-byte alignment the VMM views are laid out by.
size_t aligned(int64_t bytes) {
  return static_cast<size_t>((bytes + 255) & ~static_cast<int64_t>(255));
}

/// A created transport: the workspace's reserved addresses, its mapped storage
/// and the extents its views are laid out by.
struct Transport {
  int32_t rank;
  int32_t ranks;
  int32_t capacity;
  int32_t hidden;
  int32_t top;
  int64_t dtype;
  CUdeviceptr base;
  size_t bytes;
  size_t states_bytes;
  size_t ids_bytes;
  size_t weights_bytes;
  bool mapped;
  CUmemGenericAllocationHandle storage;
};

/// The views of a transport's workspace: the rows peers send, their route ids
/// and weights, and the per-peer row counts.
struct Views {
  void* states;
  void* ids;
  void* weights;
  void* counts;
};

Views views_of(const Transport& transport) {
  char* base = reinterpret_cast<char*>(transport.base);
  char* ids = base + transport.states_bytes;
  char* weights = ids + transport.ids_bytes;
  char* counts = weights + transport.weights_bytes;
  return Views{base, ids, weights, counts};
}

/// A device-resident allocation's properties, for this device.
CUmemAllocationProp allocation_prop() {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  return prop;
}

/// Reserves the transport's addresses and maps fresh storage at them.
int32_t reserve_and_map(Transport& transport) {
  const CUmemAllocationProp prop = allocation_prop();
  size_t granularity = 0;
  if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
      CUDA_SUCCESS) {
    return 4;
  }
  transport.bytes =
      ((transport.bytes + granularity - 1) / granularity) * granularity;
  if (cuMemAddressReserve(&transport.base, transport.bytes, 0, 0, 0) != CUDA_SUCCESS) {
    return 5;
  }
  CUmemGenericAllocationHandle storage{};
  if (cuMemCreate(&storage, transport.bytes, &prop, 0) != CUDA_SUCCESS) {
    cuMemAddressFree(transport.base, transport.bytes);
    return 6;
  }
  if (cuMemMap(transport.base, transport.bytes, 0, storage, 0) != CUDA_SUCCESS) {
    cuMemRelease(storage);
    cuMemAddressFree(transport.base, transport.bytes);
    return 7;
  }
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (cuMemSetAccess(transport.base, transport.bytes, &access, 1) != CUDA_SUCCESS) {
    cuMemUnmap(transport.base, transport.bytes);
    cuMemRelease(storage);
    cuMemAddressFree(transport.base, transport.bytes);
    return 8;
  }
  transport.storage = storage;
  transport.mapped = true;
  return 0;
}

/// Writes `message` into an entry's error buffer and answers its status.
int32_t refuse(char* error, const char* message) {
  std::snprintf(error, 512, "%s", message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_expert_exchange_create(uniserve_exchange_lifecycle* call,
                                                   void* stream) {
  (void)stream;
  if (call == nullptr) {
    return 1;
  }
  call->transport = nullptr;
  if (call->ranks != 1) {
    return refuse(call->error,
                  "expert_exchange serves one rank: a group of several ranks needs the "
                  "peer-mapping transport");
  }
  const int64_t row = row_bytes(call->dtype, call->hidden);
  if (row <= 0 || call->hidden <= 0 || call->top <= 0 || call->max_tokens <= 0) {
    return refuse(call->error,
                  "expert_exchange needs positive hidden, top and max_tokens, and an "
                  "activation dtype");
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
  transport->dtype = call->dtype;
  const int64_t rows = static_cast<int64_t>(transport->ranks) * transport->capacity;
  transport->states_bytes = aligned(rows * row);
  transport->ids_bytes = aligned(rows * transport->top * 4);
  transport->weights_bytes = transport->ids_bytes;
  transport->bytes = transport->states_bytes + transport->ids_bytes +
                     transport->weights_bytes + aligned(transport->ranks * 4);

  const int32_t mapped = reserve_and_map(*transport);
  if (mapped != 0) {
    std::free(transport);
    const char* message = "expert_exchange could not map its transport's storage";
    if (mapped == 4) {
      message = "expert_exchange could not read the device's allocation granularity";
    } else if (mapped == 5) {
      message = "expert_exchange could not reserve its transport's addresses";
    }
    return refuse(call->error, message);
  }
  call->transport = transport;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_suspend(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  if (!transport->mapped) {
    return 0;
  }
  // The storage goes; the addresses stay reserved, so a captured graph keeps
  // addressing them and `restore` maps new storage where the graph expects it.
  cuMemUnmap(transport->base, transport->bytes);
  cuMemRelease(transport->storage);
  transport->storage = CUmemGenericAllocationHandle{};
  transport->mapped = false;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_restore(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  if (transport->mapped) {
    return 0;
  }
  const CUmemAllocationProp prop = allocation_prop();
  CUmemGenericAllocationHandle storage{};
  if (cuMemCreate(&storage, transport->bytes, &prop, 0) != CUDA_SUCCESS) {
    return refuse(call->error, "expert_exchange could not create the transport's new storage");
  }
  if (cuMemMap(transport->base, transport->bytes, 0, storage, 0) != CUDA_SUCCESS) {
    cuMemRelease(storage);
    return refuse(call->error,
                  "expert_exchange could not map the transport's new storage at its addresses");
  }
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(transport->base, transport->bytes, &access, 1);
  transport->storage = storage;
  transport->mapped = true;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_release(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  if (transport->mapped) {
    cuMemUnmap(transport->base, transport->bytes);
    cuMemRelease(transport->storage);
  }
  cuMemAddressFree(transport->base, transport->bytes);
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
  const Views views = views_of(*transport);
  call->states = views.states;
  call->ids = views.ids;
  call->weights = views.weights;
  call->counts = views.counts;
  call->ranks = transport->ranks;
  call->rank = transport->rank;
  call->capacity = transport->capacity;
  call->hidden = transport->hidden;
  call->top = transport->top;
  call->dtype = transport->dtype;
  return 0;
}

namespace {

/// Moves one row of `bytes` bytes with whole 16-byte vectors where it can and
/// bytes otherwise.
__device__ __forceinline__ void copy_row(void* to, const void* from, int64_t bytes) {
  auto* destination = static_cast<char*>(to);
  const auto* source = static_cast<const char*>(from);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * 16;
  for (int64_t at = static_cast<int64_t>(threadIdx.x) * 16; at < bytes; at += stride) {
    if (at + 16 <= bytes) {
      *reinterpret_cast<int4*>(destination + at) = *reinterpret_cast<const int4*>(source + at);
    } else {
      for (int64_t one = at; one < bytes; ++one) {
        destination[one] = source[one];
      }
    }
  }
}

/// Dispatch of one rank: its rows into its own slot, with the route ids and
/// weights its experts read and the counts of what each peer sent.
__global__ void dispatch_kernel(const uniserve_exchange_dispatch_call call,
                                const Transport* transport, Views views) {
  const int64_t slot = static_cast<int64_t>(transport->rank) * transport->capacity;
  const int64_t row = row_bytes(call.dtype, call.hidden);
  for (int64_t token = blockIdx.x; token < call.tokens; token += gridDim.x) {
    copy_row(static_cast<char*>(views.states) + (slot + token) * row,
             static_cast<const char*>(call.states) + token * row, row);
    for (int64_t route = threadIdx.x; route < call.top; route += blockDim.x) {
      static_cast<int32_t*>(views.ids)[(slot + token) * call.top + route] =
          call.ids[token * call.top + route];
      static_cast<float*>(views.weights)[(slot + token) * call.top + route] =
          call.weights[token * call.top + route];
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int64_t rank = 0; rank < call.ranks; ++rank) {
      call.counts[rank] = (rank == transport->rank) ? static_cast<int32_t>(call.tokens) : 0;
    }
  }
}

/// Combine of one rank: the rows it received from itself are its own rows in
/// dispatch order, so each output row is the one partial it got back.
__global__ void combine_kernel(const uniserve_exchange_combine_call call) {
  const int64_t row = row_bytes(call.dtype, call.hidden);
  for (int64_t token = blockIdx.x; token < call.tokens; token += gridDim.x) {
    copy_row(static_cast<char*>(call.out) + token * row,
             static_cast<const char*>(call.partials) + token * row, row);
  }
}

/// The grid of one launch: one block per row, up to a bound a lap covers.
int blocks_of(const int64_t rows) {
  return static_cast<int>(rows < 1024 ? (rows > 0 ? rows : 0) : 1024);
}

}  // namespace

extern "C" int32_t uniserve_expert_exchange_dispatch(const uniserve_exchange_dispatch_call* call,
                                                     void* stream) {
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  const auto* transport = static_cast<const Transport*>(call->transport);
  if (call->ranks != transport->ranks) {
    return refuse(const_cast<char*>(call->error),
                  "expert_exchange's launch names another group size than its transport");
  }
  if (call->tokens > transport->capacity) {
    return refuse(const_cast<char*>(call->error),
                  "expert_exchange's rows exceed the transport's capacity");
  }
  if (call->states == nullptr || call->ids == nullptr || call->weights == nullptr ||
      call->counts == nullptr) {
    return refuse(const_cast<char*>(call->error),
                  "expert_exchange's dispatch needs states, ids, weights and counts");
  }
  if (row_bytes(call->dtype, call->hidden) <= 0) {
    return refuse(const_cast<char*>(call->error),
                  "expert_exchange's dispatch dtype is not an activation type");
  }
  const int blocks = blocks_of(call->tokens);
  if (blocks > 0) {
    dispatch_kernel<<<blocks, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        *call, transport, views_of(*transport));
    if (cudaGetLastError() != cudaSuccess) {
      return refuse(const_cast<char*>(call->error),
                    "expert_exchange could not start its dispatch kernel");
    }
  }
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_combine(const uniserve_exchange_combine_call* call,
                                                    void* stream) {
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  const auto* transport = static_cast<const Transport*>(call->transport);
  if (call->ranks != transport->ranks) {
    return refuse(const_cast<char*>(call->error),
                  "expert_exchange's launch names another group size than its transport");
  }
  if (call->partials == nullptr || call->out == nullptr) {
    return refuse(const_cast<char*>(call->error),
                  "expert_exchange's combine needs partials and out");
  }
  if (row_bytes(call->dtype, call->hidden) <= 0) {
    return refuse(const_cast<char*>(call->error),
                  "expert_exchange's combine dtype is not an activation type");
  }
  const int blocks = blocks_of(call->tokens);
  if (blocks > 0) {
    combine_kernel<<<blocks, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(*call);
    if (cudaGetLastError() != cudaSuccess) {
      return refuse(const_cast<char*>(call->error),
                    "expert_exchange could not start its combine kernel");
    }
  }
  return 0;
}
