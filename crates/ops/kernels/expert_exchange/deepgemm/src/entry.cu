// SPDX-License-Identifier: Apache-2.0
//
// The deepgemm backend of expert_exchange: one launch per layer exchanges the routes, runs the
// MXFP4 experts and the shared experts, and combines (`ops::moe::exchange`'s module documentation
// states the formula).
//
// The kernel is DeepGEMM's MegaMoE (`sm100_fp8_fp4_mega_moe_impl`, the `deepgemm` pin), one
// instance per tile height (`kernel.cu`). This unit owns what DeepGEMM's Python extension and
// vLLM did around it (`csrc/apis/mega_moe.hpp`, `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`,
// vLLM's `prepare_megamoe_inputs`):
//
// - the transport. DeepGEMM's buffer is one symmetric allocation a rank reaches its peers' copies
//   of by adding each peer's offset to a local address; this module creates one window a rank
//   (fabric storage, or a file descriptor between the ranks of one process), maps every rank's
//   window side by side in one reservation, and states the offsets between them. The window's
//   layout is DeepGEMM's `MegaMoEBuffer` for the descriptor (`config.cuh`), cleared once at
//   creation: the kernel's counters are tagged by each launch's grid index and need no reset.
// - the staging (`stage.cu`): the rank's BF16 rows encoded into MXFP8 and its routes written into
//   its own window, rows past the live count masked;
// - the tensor maps: the window's rows and rings, cached per tile height at creation, and the
//   layer's weights, encoded at each launch from the layer's addresses;
// - the launch: the instance DeepGEMM's rule picks for the step's capacity, which every rank of a
//   step states alike, so every rank runs one tile height.
//
// A rank with no rows of its own still launches, with none: its experts serve its peers' rows.

#include <uniserve_expert_exchange.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <unistd.h>

#include "config.cuh"
#include "launch.h"
// The tile heights and the runs of steps that take them, written by the build (`src/build.py`).
#include "uniserve_mega_runs.h"

using namespace uniserve::mega;

// Each instance's launch (`kernel.cu`), by its tile height.
#define X(height) \
  extern "C" int32_t uniserve_mega_moe_##height(const Launch* launch, void* stream);
UNISERVE_INSTANCES
#undef X

extern "C" int32_t uniserve_deepgemm_stage(const void* states, const int32_t* ids,
                                           const float* weights, const int32_t* live,
                                           void* codes, void* scales, int64_t* routes,
                                           float* route_weights, void* shared_scales,
                                           int32_t tokens, int32_t hidden, int32_t top,
                                           int32_t experts, int32_t block_m, int64_t shared_rows,
                                           void* stream);

namespace {

// One run of steps DeepGEMM's rule gives one tile height: the run's largest step of rows a rank,
// the height's launch and its tile.
struct Run {
  int largest;
  int32_t (*launch)(const Launch*, void*);
  Tile tile;
};

constexpr Run kRuns[] = {
#define X(height, largest) {largest, &uniserve_mega_moe_##height, tile(height)},
    UNISERVE_RUNS
#undef X
};
constexpr int kRunCount = sizeof(kRuns) / sizeof(kRuns[0]);

// The run of a step of `capacity` rows a rank: the first whose steps reach it.
int run_of(int64_t capacity) {
  for (int index = 0; index < kRunCount; ++index) {
    if (capacity <= kRuns[index].largest) {
      return index;
    }
  }
  return kRunCount - 1;
}

size_t aligned(size_t bytes, size_t to) { return (bytes + to - 1) / to * to; }

// The offsets of one rank's window, DeepGEMM's buffer at address zero.
struct Offsets {
  size_t input;          // [max tokens, hidden] E4M3
  size_t input_scales;   // [max tokens, hidden / 128] u32, row-major
  size_t routes;         // [max tokens, top] i64
  size_t route_weights;  // [max tokens, top] f32
  size_t shared_l1_scales, shared_l2, shared_l2_scales;
  size_t l1, l1_scales, l2, l2_scales;
  size_t bytes;
};

size_t offset_of(const void* address) { return reinterpret_cast<uintptr_t>(address); }

Offsets offsets_of() {
  const auto buffer = buffer_layout();
  Offsets offsets{};
  offsets.input = offset_of(buffer.input_token_buffer.base);
  offsets.input_scales = offset_of(buffer.input_sf_buffer.base);
  offsets.routes = offset_of(buffer.input_topk_idx_buffer.base);
  offsets.route_weights = offset_of(buffer.input_topk_weights_buffer.base);
  offsets.shared_l1_scales = offset_of(buffer.shared_l1_sf_buffer.base);
  offsets.shared_l2 = offset_of(buffer.shared_l2_token_buffer.base);
  offsets.shared_l2_scales = offset_of(buffer.shared_l2_sf_buffer.base);
  offsets.l1 = offset_of(buffer.l1_token_buffer.base);
  offsets.l1_scales = offset_of(buffer.l1_sf_buffer.base);
  offsets.l2 = offset_of(buffer.l2_token_buffer.base);
  offsets.l2_scales = offset_of(buffer.l2_sf_buffer.base);
  offsets.bytes = static_cast<size_t>(buffer.get_num_bytes());
  return offsets;
}

struct Transport {
  int32_t rank;
  int64_t device;
  Offsets offsets;
  CUdeviceptr base;  // `ranks` windows of `window` bytes, this rank's at `rank * window`
  size_t window;
  CUmemGenericAllocationHandle own;
  CUmemGenericAllocationHandle imported[kRanks];
  bool mapped;
  uint8_t* domains;    // [multiprocessors] the locality domain of each multiprocessor
  void* idle_output;   // [1, hidden] BF16: the output of a launch with no rows
  // The tensor maps of the window, per run: the routed and shared experts' rows, scales and
  // gated rows at the run's tile. The weights' maps are the layer's, encoded at each launch.
  CUtensorMap window_maps[kRunCount][kMaps];
};

char* own_base(const Transport& transport) {
  return reinterpret_cast<char*>(transport.base) + transport.rank * transport.window;
}

int32_t refuse(char* error, const char* message) {
  std::snprintf(error, UNISERVE_MESSAGE, "%s", message);
  return 1;
}

int32_t driver_refusal(char* error, const char* what, CUresult status) {
  const char* name = nullptr;
  cuGetErrorName(status, &name);
  std::snprintf(error, UNISERVE_MESSAGE, "expert_exchange: %s failed: %s", what,
                name ? name : "unknown");
  return 1;
}

int32_t runtime_refusal(char* error, const char* what, int32_t status) {
  std::snprintf(error, UNISERVE_MESSAGE, "expert_exchange: %s failed: %s", what,
                cudaGetErrorString(static_cast<cudaError_t>(status)));
  return 1;
}

// ---- Tensor maps (DeepGEMM's `runtime_utils.hpp`) -------------------------------------------

// A tensor map of `rank` dimensions, innermost first: `extents` elements, `strides` bytes between
// the outer dimensions' steps, `box` elements loaded at a time.
bool encode(CUtensorMap* map, CUtensorMapDataType type, const void* base, uint32_t rank,
            const cuuint64_t* extents, const cuuint64_t* strides, const cuuint32_t* box,
            CUtensorMapSwizzle swizzle) {
  const cuuint32_t steps[5] = {1, 1, 1, 1, 1};
  return cuTensorMapEncodeTiled(map, type, rank, const_cast<void*>(base), extents, strides, box,
                                steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

CUtensorMapSwizzle swizzle_of(int bytes) {
  return bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                      : bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_NONE;
}

// `[outer, inner]` E4M3 rows `inner` bytes apart, loaded `swizzle` bytes by `box_rows` rows under
// the swizzle (`make_tma_2d_desc`).
bool encode_rows(CUtensorMap* map, const void* base, uint64_t inner, uint64_t outer,
                 uint32_t box_rows, int swizzle) {
  const cuuint64_t extents[2] = {inner, outer};
  const cuuint64_t strides[1] = {inner};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(swizzle), box_rows};
  return encode(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, base, 2, extents, strides, box,
                swizzle_of(swizzle));
}

// A scale table of `columns / 128` columns of `rows` u32 words, MN-major, columns `stride` words
// apart, `groups` tables stacked, loaded one column of `box_rows` rows at a time
// (`make_tma_sf_desc` with a 32-element scale and one 128-element step a stage).
bool encode_scales(CUtensorMap* map, const void* base, uint64_t rows, uint64_t columns,
                   uint64_t stride, uint64_t groups, uint32_t box_rows) {
  const cuuint64_t extents[2] = {aligned(rows, 4), columns / 128 * groups};
  const cuuint64_t strides[1] = {stride * 4};
  const cuuint32_t box[2] = {box_rows, 1};
  return encode(map, CU_TENSOR_MAP_DATA_TYPE_INT32, base, 2, extents, strides, box,
                CU_TENSOR_MAP_SWIZZLE_NONE);
}

// `batches` weights of `[n, k]` elements (`fp4`: packed E2M1, two a byte; else E4M3), K-major, as
// DeepGEMM reads them: each weight's rows split into the device's locality domains, loaded one
// 128-element step of 128 rows at a time under the 128-byte swizzle (`make_tma_weights_3d_desc`).
bool encode_weights(CUtensorMap* map, const void* base, uint64_t n, uint64_t k, uint64_t batches,
                    bool fp4) {
  const uint64_t domains = deep_gemm::kNumDeviceLocalityDomains;
  const uint64_t row_bytes = fp4 ? k / 2 : k;
  const uint64_t per_domain = n / domains;
  const cuuint64_t extents[4] = {k, per_domain, domains, batches};
  const cuuint64_t strides[3] = {row_bytes, per_domain * row_bytes, n * row_bytes};
  const cuuint32_t box[4] = {static_cast<cuuint32_t>(kSwizzleOperands),
                             static_cast<cuuint32_t>(kBlockN), 1, 1};
  return encode(map,
                fp4 ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_UINT8, base,
                4, extents, strides, box, CU_TENSOR_MAP_SWIZZLE_128B);
}

// The window's maps of one tile, as `sm100_fp8_fp4_mega_moe` makes them. Without shared experts
// their maps state the routed ones, as DeepGEMM's do.
bool encode_window(CUtensorMap* maps, char* own, const Offsets& at, const Tile& tile) {
  const int load = tile.load_block_m;
  bool ok = encode_rows(&maps[kL1Acts], own + at.l1, kHidden, kRingTokens, load,
                        kSwizzleOperands) &&
            encode_scales(&maps[kL1ActsSf], own + at.l1_scales, kSfRingTokens, kHidden,
                          kSfRingTokens, 1, tile.sf_block_m) &&
            encode_rows(&maps[kL1Output], own + at.l2, kIntermediate, kRingTokens,
                        tile.store_block_m_l1, kSwizzleGated) &&
            encode_rows(&maps[kL2Acts], own + at.l2, kIntermediate, kRingTokens, load,
                        kSwizzleOperands) &&
            encode_scales(&maps[kL2ActsSf], own + at.l2_scales, kSfRingTokens, kIntermediate,
                          kSfRingTokens, 1, tile.sf_block_m);
  if (kShared == 0) {
    maps[kSharedL1Acts] = maps[kL1Acts];
    maps[kSharedL1ActsSf] = maps[kL1ActsSf];
    maps[kSharedL1Output] = maps[kL1Output];
    maps[kSharedL2Acts] = maps[kL2Acts];
    maps[kSharedL2ActsSf] = maps[kL2ActsSf];
    return ok;
  }
  return ok &&
         encode_rows(&maps[kSharedL1Acts], own + at.input, kHidden, kMaxTokens, load,
                     kSwizzleOperands) &&
         encode_scales(&maps[kSharedL1ActsSf], own + at.shared_l1_scales, kSharedSfTokens,
                       kHidden, kSharedSfTokens, 1, tile.sf_block_m) &&
         encode_rows(&maps[kSharedL1Output], own + at.shared_l2, kSharedIntermediate,
                     kMaxTokens, tile.store_block_m_l1, kSwizzleGated) &&
         encode_rows(&maps[kSharedL2Acts], own + at.shared_l2, kSharedIntermediate, kMaxTokens,
                     load, kSwizzleOperands) &&
         encode_scales(&maps[kSharedL2ActsSf], own + at.shared_l2_scales, kSharedSfTokens,
                       kSharedIntermediate, kSharedSfTokens, 1, tile.sf_block_m);
}

// The layer's maps: the routed stacks and, with shared experts, their matrices.
bool encode_layer(CUtensorMap* maps, const uniserve_fused_layer& layer, const Tile& tile) {
  // Each weight's scale table is `[columns / 128, rows]` words a matrix, MN-major
  // (`ops::moe::exchange::Descriptor::MXFP4_SCALES`).
  constexpr uint64_t kUpRows = 2 * static_cast<uint64_t>(kIntermediate);
  bool ok = encode_weights(&maps[kL1Weights], layer.up_gate, kUpRows, kHidden, kLocalExperts,
                           true) &&
            encode_scales(&maps[kL1WeightsSf], layer.up_gate_scales, kUpRows, kHidden, kUpRows,
                          kLocalExperts, tile.sf_block_n) &&
            encode_weights(&maps[kL2Weights], layer.down, kHidden, kIntermediate, kLocalExperts,
                           true) &&
            encode_scales(&maps[kL2WeightsSf], layer.down_scales, kHidden, kIntermediate,
                          kHidden, kLocalExperts, tile.sf_block_n);
  if (kShared == 0) {
    maps[kSharedL1Weights] = maps[kL1Weights];
    maps[kSharedL1WeightsSf] = maps[kL1WeightsSf];
    maps[kSharedL2Weights] = maps[kL2Weights];
    maps[kSharedL2WeightsSf] = maps[kL2WeightsSf];
    return ok;
  }
  constexpr uint64_t kSharedUpRows = 2 * static_cast<uint64_t>(kSharedIntermediate);
  return ok &&
         encode_weights(&maps[kSharedL1Weights], layer.shared_up_gate, kSharedUpRows, kHidden, 1,
                        false) &&
         encode_scales(&maps[kSharedL1WeightsSf], layer.shared_up_gate_scales, kSharedUpRows,
                       kHidden, kSharedUpRows, 1, tile.sf_block_n) &&
         encode_weights(&maps[kSharedL2Weights], layer.shared_down, kHidden, kSharedIntermediate,
                        1, false) &&
         encode_scales(&maps[kSharedL2WeightsSf], layer.shared_down_scales, kHidden,
                       kSharedIntermediate, kHidden, 1, tile.sf_block_n);
}

// ---- The windows -----------------------------------------------------------------------------

struct Shared {
  unsigned char bytes[64];
};

// What a rank states of its window: the handle, its kind, and the rank's process, since a file
// descriptor names a window only in the process that exported it.
struct Record {
  Shared handle;
  int64_t kind;
  int64_t process;
};

// This process's word among the ranks of a group: drawn once, so two processes on any hosts state
// different words.
int64_t process_word() {
  static const int64_t word = [] {
    std::random_device draw;
    const uint64_t high = static_cast<uint64_t>(draw()) << 32;
    return static_cast<int64_t>((high | draw()) ^ static_cast<uint64_t>(getpid()));
  }();
  return word;
}

CUmemAllocationProp window_prop(int64_t device, CUmemAllocationHandleType kind) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device);
  prop.requestedHandleTypes = kind;
  return prop;
}

// The bytes of one rank's window on `device`: whole recommended granules of its storage, which
// are also the alignment of the addresses the windows are mapped at.
CUresult window_bytes(int64_t device, size_t* window, size_t* granularity) {
  const CUmemAllocationProp prop = window_prop(device, CU_MEM_HANDLE_TYPE_FABRIC);
  const CUresult status =
      cuMemGetAllocationGranularity(granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  if (status == CUDA_SUCCESS) {
    *window = aligned(offsets_of().bytes, *granularity);
  }
  return status;
}

// Creates this rank's window and maps every rank's: a fabric handle where the driver has one,
// which serves ranks of any process on any host of the NVLink domain, and a file descriptor
// between the ranks of one process otherwise; a descriptor of another process is refused, since
// its number would name a file of this one.
int32_t map_windows(Transport& transport, const UniserveGroup* group, char* error) {
  size_t granularity = 0;
  CUresult status = window_bytes(transport.device, &transport.window, &granularity);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(error, "cuMemGetAllocationGranularity", status);
  }
  CUmemAllocationHandleType kind = CU_MEM_HANDLE_TYPE_FABRIC;
  CUmemAllocationProp prop = window_prop(transport.device, kind);
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
    prop = window_prop(transport.device, kind);
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
  Record record{mine, static_cast<int64_t>(kind), process_word()};
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
    if (kind != CU_MEM_HANDLE_TYPE_FABRIC && records[rank].process != record.process) {
      return refuse(error,
                    "expert_exchange's ranks span processes, which map each other's windows "
                    "only through fabric handles, and the driver grants none");
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
  // DeepGEMM clears the buffer once: the kernel tags its counters by each launch's grid index.
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

// The locality domain the kernel schedules each multiprocessor's blocks for. Without memory
// localized to a domain any mapping is correct (DeepGEMM's `get_even_sm_locality_domains`): the
// two multiprocessors of a texture processing cluster share one, and the clusters alternate.
std::vector<uint8_t> even_domains() {
  std::vector<uint8_t> domains(kMultiprocessors);
  for (int sm = 0; sm < kMultiprocessors; ++sm) {
    domains[sm] = static_cast<uint8_t>((sm / 2) % deep_gemm::kNumDeviceLocalityDomains);
  }
  return domains;
}

/// Refuses a lifecycle call of another shape than the module's; zero for its own.
int32_t check_shape(uniserve_exchange_lifecycle* call) {
  if (call->ranks != kRanks || call->attention_ranks != 0 || call->hidden != kHidden ||
      call->top != kTop || call->experts != kExperts || call->max_tokens != kStatedTokens ||
      call->intermediate != kIntermediate || call->dtype != UNISERVE_DTYPE_BF16 ||
      call->weights != UNISERVE_EXCHANGE_WEIGHTS_MXFP4 || call->shared != kShared ||
      std::bit_cast<uint32_t>(call->limit) != static_cast<uint32_t>(UNISERVE_LIMIT_BITS)) {
    return refuse(call->error,
                  "expert_exchange's deepgemm module was compiled for another group or another "
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
  int multiprocessors = 0;
  cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                         static_cast<int>(call->device));
  if (multiprocessors != kMultiprocessors) {
    char message[UNISERVE_MESSAGE];
    std::snprintf(message, sizeof(message),
                  "expert_exchange's deepgemm module runs one block on each of %d "
                  "multiprocessors, and device %lld has %d",
                  kMultiprocessors, static_cast<long long>(call->device), multiprocessors);
    return refuse(call->error, message);
  }

  auto* transport = static_cast<Transport*>(std::calloc(1, sizeof(Transport)));
  if (transport == nullptr) {
    return refuse(call->error, "expert_exchange could not allocate its transport record");
  }
  transport->rank = static_cast<int32_t>(call->rank);
  transport->device = call->device;
  transport->offsets = offsets_of();
  int32_t status = map_windows(*transport, call->group, call->error);
  if (status == 0) {
    const std::vector<uint8_t> domains = even_domains();
    cudaError_t cuda = cudaMalloc(reinterpret_cast<void**>(&transport->domains), domains.size());
    if (cuda == cudaSuccess) {
      cuda = cudaMemcpy(transport->domains, domains.data(), domains.size(),
                        cudaMemcpyHostToDevice);
    }
    if (cuda == cudaSuccess) {
      cuda = cudaMalloc(&transport->idle_output, static_cast<size_t>(kHidden) * 2);
    }
    if (cuda != cudaSuccess) {
      status = runtime_refusal(call->error, "creating the module's device state", cuda);
    }
  }
  for (int index = 0; status == 0 && index < kRunCount; ++index) {
    if (!encode_window(transport->window_maps[index], own_base(*transport), transport->offsets,
                       kRuns[index].tile)) {
      status = refuse(call->error, "expert_exchange could not encode its window's tensor maps");
    }
  }
  if (status != 0) {
    std::free(transport);
    return status;
  }
  call->transport = transport;
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_footprint(uniserve_exchange_lifecycle* call,
                                                      int64_t* bytes) {
  if (call == nullptr || bytes == nullptr || check_shape(call) != 0) {
    return 1;
  }
  // A rank creates its own window alone: it maps its peers' windows, which they create.
  size_t window = 0;
  size_t granularity = 0;
  const CUresult status = window_bytes(call->device, &window, &granularity);
  if (status != CUDA_SUCCESS) {
    return driver_refusal(call->error, "cuMemGetAllocationGranularity", status);
  }
  *bytes = static_cast<int64_t>(window + kMultiprocessors + static_cast<size_t>(kHidden) * 2);
  return 0;
}

extern "C" int32_t uniserve_expert_exchange_step(uniserve_exchange_lifecycle* call, void* stream) {
  (void)stream;
  // A launch takes its tile from its call's capacity, so a step changes nothing here.
  return call == nullptr || call->transport == nullptr ? 1 : 0;
}

extern "C" int32_t uniserve_expert_exchange_suspend(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  return call == nullptr ? 1
                         : refuse(call->error, "the deepgemm exchange does not suspend yet");
}

extern "C" int32_t uniserve_expert_exchange_restore(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  return call == nullptr ? 1
                         : refuse(call->error, "the deepgemm exchange does not suspend yet");
}

extern "C" int32_t uniserve_expert_exchange_release(uniserve_exchange_lifecycle* call,
                                                    void* stream) {
  (void)stream;
  if (call == nullptr || call->transport == nullptr) {
    return 1;
  }
  auto* transport = static_cast<Transport*>(call->transport);
  cudaDeviceSynchronize();
  // No rank unmaps its window while a peer may still reach it.
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
  cudaFree(transport->domains);
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
  // A fused launch reads the layer's rows itself and computes in its own window, so no view holds
  // anything a layer reads; each names the window, so that its extents can still be stated.
  call->states = own;
  call->ids = own;
  call->weights = own;
  call->combined = own;
  call->live = own;
  call->packed = own;
  call->packed_scales = own;
  call->ranks = kRanks;
  call->rank = transport->rank;
  call->capacity = kStatedTokens;
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
    return refuse(call->error, "expert_exchange's deepgemm backend runs only a fused launch");
  }
  const auto tokens = call->tokens;
  if (tokens < 0 || tokens > kStatedTokens || call->capacity < tokens ||
      call->capacity > kStatedTokens) {
    return refuse(call->error,
                  "expert_exchange's rows exceed their step's capacity or the transport's");
  }
  const uniserve_fused_layer& layer = *call->layer;
  if ((layer.shared_up_gate != nullptr) != (kShared > 0)) {
    return refuse(call->error, "the layer's shared experts are not the module's");
  }
  auto* transport = static_cast<Transport*>(call->transport);
  char* own = own_base(*transport);
  const Offsets& at = transport->offsets;

  const int index = run_of(call->capacity);
  const Tile& tile = kRuns[index].tile;
  int32_t status = uniserve_deepgemm_stage(
      call->states, call->ids, call->weights, call->live, own + at.input, own + at.input_scales,
      reinterpret_cast<int64_t*>(own + at.routes), reinterpret_cast<float*>(own + at.route_weights),
      kShared > 0 ? own + at.shared_l1_scales : nullptr, static_cast<int32_t>(tokens), kHidden,
      kTop, kExperts, tile.block_m, kSharedSfTokens, stream);
  if (status != 0) {
    return runtime_refusal(call->error, "staging the rows", status);
  }

  Launch launch{};
  launch.y = tokens > 0 ? call->out : transport->idle_output;
  launch.tokens = static_cast<uint32_t>(tokens);
  std::vector<int64_t> windows(kRanks);
  for (int rank = 0; rank < kRanks; ++rank) {
    windows[rank] = static_cast<int64_t>(transport->base + rank * transport->window);
  }
  launch.windows = deep_gemm::layout::SymBuffer<kRanks>(windows, transport->rank);
  std::memcpy(launch.maps, transport->window_maps[index], sizeof(launch.maps));
  if (!encode_layer(launch.maps, layer, tile)) {
    return refuse(call->error, "expert_exchange could not encode the layer's tensor maps");
  }
  launch.domains = transport->domains;
  status = kRuns[index].launch(&launch, stream);
  return status == 0 ? 0 : runtime_refusal(call->error, "the fused kernel's launch", status);
}
