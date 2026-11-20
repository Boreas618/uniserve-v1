// SPDX-License-Identifier: Apache-2.0

// The index_scores module's DeepGEMM backend: the operator's dense, weighted scoring of MXFP4
// queries against MXFP4 index keys in the page layout, in DeepSeek's BF16 arithmetic
// (`ops::attention::IndexScores`), launched as DeepGEMM's SM100 paged MQA logits.
//
//   s[t, g] = bf16(Σ_h bf16(relu(bf16(q[t, h] · key[g])) · bf16(w[t, h])))   g < visible[t]
//   s[t, g] = −inf                                                           otherwise
//
// The kernel is DeepGEMM's own (`sm100_paged_mqa_logits`, the `deepgemm` pin) with the
// package's patch (`patches/paged-logits.patch`): its epilogue rounds where DeepSeek's indexer
// rounds and sums the heads in FP32 (SM100's mixed-precision `add.f32.bf16` reads each rounded
// BF16 term in place, two instructions a head where DeepGEMM's BF16 chains take one), a
// request's tokens read the block-table row their index names, a request's end is read from its
// tokens' ends rather than scanned, and the schedule is balanced over the launched
// multiprocessors. It owns the products: TMA loads of the packed E2M1
// queries and keys and their UE8M0 scale words, the tcgen05 block-scaled MMA with the scales
// applied inside it, FP32 products in tensor memory, and the weighted sum over the heads of
// every visible key a row sees.
//
// This unit owns what DeepGEMM's extension did around it (its `fp8_fp4_paged_mqa_logits`,
// `csrc/apis/attention.hpp` and `csrc/jit_kernels/impls/sm100_mqa_logits.hpp`), and what the
// operator's call does not hand the kernel in its own form, in the call's workspace:
//
// 1. `prepare`: each head's query rounded to MXFP4 blocks of 32 by the operator's rule (the
//    record encoding's power-of-two scale and nearest-even E2M1 codes) and packed as the kernel
//    reads it (E2M1 pairs, then one word of the four UE8M0 scale bytes), the head weights
//    rounded to BF16, the row's table row and its visible count (at most the columns) copied as
//    the kernel reads them, the segment the row is scheduled in, and the row's columns past its
//    visible groups stored as −inf, which the kernel never writes;
// 2. DeepGEMM's schedule (`sm100_paged_mqa_logits_metadata`, one block): the rows' segments,
//    their costs balanced over the multiprocessors, and each row's segment end. DeepGEMM's
//    scheduler spreads one request over the multiprocessors by its key splits alone, which
//    leaves a prefill of thousands of rows to the few multiprocessors its splits give; so a
//    request is scheduled as segments of at most `segment` consecutive rows, each the kernel's
//    request: its rows read their request's table row, and its keys end at its last row's
//    visible groups;
// 3. the kernel: one block per multiprocessor, with the five tensor maps DeepGEMM's runtime
//    encodes (queries, query scales, the key plane's values as `[pages, groups, width]` E2M1
//    and its scale words as `[pages, groups]`, and the BF16 weights).
//
// The kernel's tensor maps hold the operands' addresses by value, and nothing synchronizes with
// the host, so a CUDA graph captures the three launches.

#include <uniserve_index_scores.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <deep_gemm/impls/sm100_mqa_logits.cuh>

#if !defined(UNISERVE_HEADS) || !defined(UNISERVE_WIDTH) || !defined(UNISERVE_GROUPS)
#error "the build defines the descriptor's heads, head width and groups of a page (src/build.py)"
#endif

namespace {

// The descriptor's extents.
constexpr uint32_t kHeads = UNISERVE_HEADS;
constexpr uint32_t kWidth = UNISERVE_WIDTH;
constexpr uint32_t kGroups = UNISERVE_GROUPS;

// DeepGEMM's geometry for MXFP4 operands (`MQALogitsConfig`): a query tile of 128 (row, head)
// pairs, splits of 384 keys, one per math warpgroup's 128-key MMA, three tensor-memory stages,
// two query and eight key stages, eight splits a chunk whose keys a request's query blocks share.
constexpr uint32_t kBlockQ = 128 / kHeads;
constexpr uint32_t kUmmaN = (kBlockQ * kHeads + 7) / 8 * 8;
constexpr uint32_t kSplitKV = 384;
constexpr uint32_t kTmemStages = 3;
constexpr uint32_t kQStages = 2;
constexpr uint32_t kKVStages = 8;
constexpr uint32_t kSplitsPerChunk = 8;
constexpr uint32_t kSpecializedThreads = 128;
constexpr uint32_t kMathThreads = kSplitKV / 128 * 128;

static_assert(kHeads % 4 == 0 && 128 % kHeads == 0, "a query tile holds whole rows of heads");
static_assert(kWidth == 128, "a head's MXFP4 query is one word of four scales");
static_assert(kGroups == 32 || kGroups == 64 || kGroups == 128, "DeepGEMM's page sizes");
static_assert(kSplitKV % kGroups == 0, "a split is whole pages");

// Bytes of one E2M1 record's values and of a page; a head's four scale bytes are one word.
constexpr uint32_t kValueBytes = kWidth / 2;
constexpr uint32_t kRecordBytes = kValueBytes + kWidth / 32;
constexpr uint64_t kPageBytes = uint64_t{kGroups} * kRecordBytes;

// A weight row: the heads' BF16 weights, padded to 16 bytes for its tensor map.
constexpr uint32_t kWeightRowBytes = (kHeads * 2 + 15) / 16 * 16;

// The multiprocessors the workspace holds a schedule for (`ops::attention::indexed`).
constexpr int64_t kScheduledMultiprocessors = 1024;

using Storage = deep_gemm::layout::MQALogitsSharedStorage<kHeads, kWidth, kBlockQ, kSplitKV,
                                                          kUmmaN, kQStages, kKVStages,
                                                          kTmemStages, cutlass::float_e2m1_t>;
constexpr uint32_t kSharedBytes = sizeof(Storage);
static_assert(kSharedBytes <= 232448, "the stages fit a block's shared memory");

constexpr auto kKernel = &deep_gemm::sm100_paged_mqa_logits<
    kHeads, kWidth, kGroups, kBlockQ, kUmmaN, kQStages, kKVStages, kTmemStages, kSplitKV,
    kSplitsPerChunk, kSpecializedThreads, kMathThreads, cutlass::float_e2m1_t>;

// The schedule's block: its thread count by the rows, as DeepGEMM's runtime chooses it.
template <uint32_t kThreads>
constexpr auto kSchedule =
    &deep_gemm::sched::sm100_paged_mqa_logits_metadata<kSplitKV, kBlockQ, kThreads>;

// The workspace's regions (`ops::attention::IndexScores::workspace_bytes`).
struct Workspace {
  uint8_t* query;       // [tokens · heads, width / 2] E2M1 pairs
  uint32_t* scales;     // [tokens, heads] words of four UE8M0 bytes, the first block lowest
  uint8_t* weights;     // [tokens, kWeightRowBytes] BF16 weights
  uint32_t* indices;    // [tokens] each row's table row
  uint32_t* visible;    // [tokens] each row's visible groups, at most the columns
  uint32_t* segments;   // [tokens] the first row of each row's segment
  uint32_t* ends;       // [tokens] the end of each row's segment
  uint32_t* schedule;   // [multiprocessors + 1] pairs of (row, split) starts
  uint64_t bytes;       // what the regions take, with each one's alignment
};

constexpr uint64_t align16(uint64_t bytes) { return (bytes + 15) / 16 * 16; }

Workspace carve(void* base, int64_t tokens, int64_t multiprocessors) {
  const auto rows = static_cast<uint64_t>(tokens);
  auto* cursor = static_cast<uint8_t*>(base);
  uint64_t taken = 0;
  const auto take = [&](uint64_t bytes) {
    uint8_t* region = cursor + taken;
    taken += align16(bytes);
    return region;
  };
  Workspace workspace{};
  // The base is 16-byte aligned (a tensor allocation), so every region is.
  workspace.query = take(rows * kHeads * kValueBytes);
  workspace.scales = reinterpret_cast<uint32_t*>(take(rows * kHeads * 4));
  workspace.weights = take(rows * kWeightRowBytes);
  workspace.indices = reinterpret_cast<uint32_t*>(take(rows * 4));
  workspace.visible = reinterpret_cast<uint32_t*>(take(rows * 4));
  workspace.segments = reinterpret_cast<uint32_t*>(take(rows * 4));
  workspace.ends = reinterpret_cast<uint32_t*>(take(rows * 4));
  workspace.schedule =
      reinterpret_cast<uint32_t*>(take(8 * static_cast<uint64_t>(multiprocessors + 1)));
  workspace.bytes = taken;
  return workspace;
}

// The E2M1 code of `x`, within ±6, by nearest even: the record encoding's rule
// (`ops::record::encode`), whose boundaries are the midpoints between neighbouring codes.
__device__ __forceinline__ uint32_t e2m1_code(float x) {
  const float magnitude = fabsf(x);
  uint32_t code = magnitude > 0.25f ? 1u : 0u;
  code = magnitude >= 0.75f ? 2u : code;
  code = magnitude > 1.25f ? 3u : code;
  code = magnitude >= 1.75f ? 4u : code;
  code = magnitude > 2.5f ? 5u : code;
  code = magnitude >= 3.5f ? 6u : code;
  code = magnitude > 5.0f ? 7u : code;
  return code | (x < 0.0f && code != 0 ? 8u : 0u);
}

// The threads of a `prepare` block.
constexpr int kPrepareThreads = 256;
constexpr int kPrepareWarps = kPrepareThreads / 32;

// Rounds the query of one (row, head) pair to MXFP4 into the workspace, with all 32 lanes of a
// warp: lane `l` holds the head's elements `4l .. 4l + 4`, so the eight lanes `8b .. 8b + 8`
// hold scale block `b`.
__device__ __forceinline__ void round_query(const __nv_bfloat16* __restrict__ query, uint64_t pair,
                                            uint32_t lane, const Workspace& workspace) {
  // The head's four values of this lane, read as one 8-byte word.
  const uint2 packed = *reinterpret_cast<const uint2*>(query + pair * kWidth + 4 * lane);
  const float values[4] = {
      __uint_as_float(packed.x << 16), __uint_as_float(packed.x & 0xFFFF0000u),
      __uint_as_float(packed.y << 16), __uint_as_float(packed.y & 0xFFFF0000u)};

  // The block's largest magnitude, floored at 6 · 2^-126, and its power-of-two scale
  // `2^ceil(log2(amax · f32(1/6)))`, read off the bits as the record encoding computes it.
  float amax = fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])),
                     fmaxf(fabsf(values[2]), fabsf(values[3])));
#pragma unroll
  for (int offset = 1; offset < 8; offset *= 2) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
  }
  amax = fmaxf(amax, 6.0f * 1.1754943508222875e-38f);
  const uint32_t bits = __float_as_uint(amax * (1.0f / 6.0f));
  const int32_t exponent =
      static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
  // The scale's reciprocal is a normal power of two (the exponent lies in -126 .. 126), so the
  // quotient is exact.
  const float reciprocal = __uint_as_float(static_cast<uint32_t>(127 - exponent) << 23);

  uint32_t codes = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float quotient = fminf(fmaxf(values[j] * reciprocal, -6.0f), 6.0f);
    codes |= e2m1_code(quotient) << (4 * j);
  }
  // Two bytes of E2M1 pairs: the even element in a byte's low nibble.
  reinterpret_cast<uint16_t*>(workspace.query + pair * kValueBytes)[lane] =
      static_cast<uint16_t>(codes);

  // The four blocks' scale bytes, the first block's lowest, from the first lane of each block.
  const uint32_t scale = static_cast<uint32_t>(exponent + 127);
  uint32_t word = 0;
#pragma unroll
  for (int block = 0; block < 4; ++block) {
    word |= __shfl_sync(0xFFFFFFFFu, scale, block * 8) << (8 * block);
  }
  if (lane == 0) {
    workspace.scales[pair] = word;
  }
}

// What the kernel reads, and the −inf of the columns it never writes. The first
// `query_blocks` blocks round the queries, each warp of them the (row, head) pairs a stride of
// the grid's query warps apart; each later block takes one row: its weights, its words and
// its columns past its visible groups.
__global__ void __launch_bounds__(kPrepareThreads) prepare(
    const __nv_bfloat16* __restrict__ query, const float* __restrict__ weights,
    const uint32_t* __restrict__ rows, const uint32_t* __restrict__ visible,
    __nv_bfloat16* __restrict__ out, int64_t tokens, int64_t columns, int64_t segment,
    int64_t query_blocks, Workspace workspace) {
  if (blockIdx.x < query_blocks) {
    const uint64_t pairs = static_cast<uint64_t>(tokens) * kHeads;
    const uint64_t warps = static_cast<uint64_t>(query_blocks) * kPrepareWarps;
    for (uint64_t pair = blockIdx.x * kPrepareWarps + threadIdx.x / 32; pair < pairs;
         pair += warps) {
      round_query(query, pair, threadIdx.x % 32, workspace);
    }
    return;
  }
  const int64_t token = blockIdx.x - query_blocks;

  // The head weights rounded to BF16, nearest even, as the scores' formula reads them.
  if (threadIdx.x < kHeads) {
    reinterpret_cast<__nv_bfloat16*>(workspace.weights + token * kWeightRowBytes)[threadIdx.x] =
        __float2bfloat16_rn(weights[token * kHeads + threadIdx.x]);
  }

  // The row's table row and visible count, which the schedule and the kernel read, and the
  // first row of its segment: the later of its request's first row and the first row of its
  // `segment`-row block. A request's rows are consecutive and no other request reads its table
  // row, so the rows of the block that read it are a suffix of the block up to this one, whose
  // first is found by bisection.
  const int64_t seen = visible[token] < columns ? visible[token] : columns;
  if (threadIdx.x == 0) {
    const uint32_t row = rows[token];
    int64_t low = token / segment * segment;
    int64_t high = token;
    while (low < high) {
      const int64_t middle = (low + high) / 2;
      if (rows[middle] == row) {
        high = middle;
      } else {
        low = middle + 1;
      }
    }
    workspace.indices[token] = row;
    workspace.visible[token] = static_cast<uint32_t>(seen);
    workspace.segments[token] = static_cast<uint32_t>(low);
  }

  // The columns past the visible groups score −inf (BF16 0xFF80): single stores up to the
  // first column on a 16-byte boundary, eight at a time from there, single stores after the
  // last whole eight. The scores are at least 2-byte aligned, so the boundary is a column.
  constexpr uint16_t kNegativeInfinity = 0xFF80u;
  auto* line = reinterpret_cast<uint16_t*>(out) + token * columns;
  const auto offset = static_cast<int64_t>((reinterpret_cast<uintptr_t>(line) / 2) % 8);
  const int64_t vectored = min(columns, (seen + offset + 7) / 8 * 8 - offset);
  for (int64_t c = seen + threadIdx.x; c < vectored; c += blockDim.x) {
    line[c] = kNegativeInfinity;
  }
  const int64_t vectors = (columns - vectored) / 8;
  const uint4 negative = make_uint4(0xFF80FF80u, 0xFF80FF80u, 0xFF80FF80u, 0xFF80FF80u);
  for (int64_t v = threadIdx.x; v < vectors; v += blockDim.x) {
    *reinterpret_cast<uint4*>(line + vectored + 8 * v) = negative;
  }
  for (int64_t c = vectored + 8 * vectors + threadIdx.x; c < columns; c += blockDim.x) {
    line[c] = kNegativeInfinity;
  }
}

int refuse(uniserve_index_scores_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

bool aligned(const void* address) {
  return address != nullptr && reinterpret_cast<uintptr_t>(address) % 16 == 0;
}

// A tensor map of `rank` dimensions over `base`: `extents` elements of `type`, strides between
// consecutive indices of each outer dimension in bytes, loaded in boxes of `box` under
// `swizzle`. Out-of-bounds elements of a box load as zeros.
bool encode(CUtensorMap* map, CUtensorMapDataType type, uint32_t rank, const void* base,
            const cuuint64_t* extents, const cuuint64_t* strides, const cuuint32_t* box,
            CUtensorMapSwizzle swizzle) {
  const cuuint32_t steps[3] = {1, 1, 1};
  return cuTensorMapEncodeTiled(map, type, rank, const_cast<void*>(base), extents, strides, box,
                                steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// The schedule's launch for `tokens` rows: DeepGEMM's thread count for them, and the shared
// memory of their request starts, their costs and the warp sums.
cudaError_t schedule(const Workspace& workspace, uint32_t tokens, uint32_t multiprocessors,
                     cudaStream_t stream) {
  const uint32_t threads = tokens <= 512 ? 256 : (tokens <= 2048 ? 512 : 1024);
  const size_t shared = (2 * static_cast<size_t>(tokens) + threads / 32 + 1) * sizeof(uint32_t);
  const auto launch = [&](auto kernel) {
    cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(shared));
    if (status != cudaSuccess) {
      return status;
    }
    kernel<<<1, threads, shared, stream>>>(multiprocessors, tokens, workspace.visible,
                                           workspace.segments, workspace.schedule,
                                           workspace.ends);
    return cudaGetLastError();
  };
  switch (threads) {
    case 256:
      return launch(kSchedule<256>);
    case 512:
      return launch(kSchedule<512>);
    default:
      return launch(kSchedule<1024>);
  }
}

int launch(uniserve_index_scores_call* call, cudaStream_t stream) {
  const int64_t tokens = call->tokens;
  const auto multiprocessors = static_cast<uint32_t>(call->multiprocessors);
  const Workspace workspace = carve(call->workspace, tokens, multiprocessors);
  if (workspace.bytes > static_cast<uint64_t>(call->workspace_bytes)) {
    return refuse(call, "the workspace does not hold the rounded queries, weights and schedule");
  }

  // 1. The kernel's operands of each row, and the columns it never writes. A segment spans
  // the rows that give the schedule about four (segment, split) units a multiprocessor when the
  // rows are one causal request over the columns, `tokens · columns / 2` (row, key) pairs in
  // units of the split's 384 keys: a power of two of whole query blocks, at most 1024 rows.
  const auto pairs = static_cast<uint64_t>(tokens) * static_cast<uint64_t>(call->columns);
  const uint64_t wanted = pairs / (uint64_t{2} * kSplitKV * 4 * multiprocessors);
  int64_t segment = kBlockQ;
  while (segment < 1024 && static_cast<uint64_t>(segment) * 2 <= wanted) {
    segment *= 2;
  }
  // The queries are rounded by up to eight blocks a multiprocessor, a warp a (row, head) pair
  // at a time; every row has a block of its own after them.
  const int64_t query_pairs = tokens * kHeads;
  const int64_t query_blocks =
      min((query_pairs + kPrepareWarps - 1) / kPrepareWarps, int64_t{8} * multiprocessors);
  prepare<<<static_cast<unsigned>(query_blocks + tokens), kPrepareThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(call->query), static_cast<const float*>(call->weights),
      static_cast<const uint32_t*>(call->rows), static_cast<const uint32_t*>(call->visible),
      static_cast<__nv_bfloat16*>(call->out), tokens, call->columns, segment, query_blocks,
      workspace);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }

  // 2. The rows' requests and their schedule over the multiprocessors.
  status = schedule(workspace, static_cast<uint32_t>(tokens), multiprocessors, stream);
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }

  // 3. The tensor maps DeepGEMM's runtime encodes for packed MXFP4 operands: E2M1 values packed
  // two to a byte in shared memory under the 64-byte swizzle of a 128-element head, scale words
  // and weights unswizzled.
  CUtensorMap query, query_scales, keys, key_scales, weights;
  const auto rows = static_cast<cuuint64_t>(tokens);
  const auto pages = static_cast<cuuint64_t>(call->pages);
  {
    const cuuint64_t extents[2] = {kWidth, rows * kHeads};
    const cuuint64_t strides[1] = {kValueBytes};
    const cuuint32_t box[2] = {kWidth, kBlockQ * kHeads};
    if (!encode(&query, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, workspace.query, extents,
                strides, box, CU_TENSOR_MAP_SWIZZLE_64B)) {
      return refuse(call, "the queries' tensor map cannot be encoded");
    }
  }
  {
    const cuuint64_t extents[2] = {kHeads, rows};
    const cuuint64_t strides[1] = {kHeads * sizeof(uint32_t)};
    const cuuint32_t box[2] = {kHeads, kBlockQ};
    if (!encode(&query_scales, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, workspace.scales, extents,
                strides, box, CU_TENSOR_MAP_SWIZZLE_NONE)) {
      return refuse(call, "the query scales' tensor map cannot be encoded");
    }
  }
  {
    // A page's values: `[groups, width]` E2M1, one record's 64 bytes after another.
    const cuuint64_t extents[3] = {kWidth, kGroups, pages};
    const cuuint64_t strides[2] = {kValueBytes, kPageBytes};
    const cuuint32_t box[3] = {kWidth, kGroups, 1};
    if (!encode(&keys, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, call->keys, extents, strides,
                box, CU_TENSOR_MAP_SWIZZLE_64B)) {
      return refuse(call, "the keys' tensor map cannot be encoded");
    }
  }
  {
    // A page's scale words follow all its values: one row of `groups` words a page.
    const cuuint64_t extents[2] = {kGroups, pages};
    const cuuint64_t strides[1] = {kPageBytes};
    const cuuint32_t box[2] = {kGroups, 1};
    const auto* base = static_cast<const uint8_t*>(call->keys) + uint64_t{kGroups} * kValueBytes;
    if (!encode(&key_scales, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, base, extents, strides, box,
                CU_TENSOR_MAP_SWIZZLE_NONE)) {
      return refuse(call, "the key scales' tensor map cannot be encoded");
    }
  }
  {
    const cuuint64_t extents[2] = {kHeads, rows};
    const cuuint64_t strides[1] = {kWeightRowBytes};
    const cuuint32_t box[2] = {kWeightRowBytes / 2, kBlockQ};
    if (!encode(&weights, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, workspace.weights, extents,
                strides, box, CU_TENSOR_MAP_SWIZZLE_NONE)) {
      return refuse(call, "the weights' tensor map cannot be encoded");
    }
  }

  status = cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(kSharedBytes));
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(multiprocessors, 1, 1);
  config.blockDim = dim3(kSpecializedThreads + kMathThreads, 1, 1);
  config.dynamicSmemBytes = kSharedBytes;
  config.stream = stream;
  status = cudaLaunchKernelEx(
      &config, kKernel, static_cast<uint32_t>(tokens), static_cast<uint32_t>(call->columns),
      static_cast<uint32_t>(call->max_pages), static_cast<const uint32_t*>(workspace.visible),
      static_cast<nv_bfloat16*>(call->out), static_cast<const uint32_t*>(call->page_table),
      static_cast<const uint32_t*>(workspace.indices),
      static_cast<const uint32_t*>(workspace.ends),
      static_cast<const uint32_t*>(workspace.schedule), query, query_scales, keys, key_scales,
      weights);
  if (status != cudaSuccess) {
    return refuse(call, cudaGetErrorString(status));
  }
  return 0;
}

}  // namespace

extern "C" int uniserve_index_scores(uniserve_index_scores_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  std::memset(call->error, 0, sizeof(call->error));

  if (call->heads != kHeads || call->width != kWidth || call->groups != kGroups ||
      call->candidate_block != 0 || call->scores_dtype != UNISERVE_DTYPE_BF16) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "this module is compiled for %u heads of %u over pages of %u keys, dense, "
                  "with BF16 scores, and was handed %lld, %lld, %lld, candidate blocks of %lld "
                  "and dtype %lld",
                  kHeads, kWidth, kGroups, static_cast<long long>(call->heads),
                  static_cast<long long>(call->width), static_cast<long long>(call->groups),
                  static_cast<long long>(call->candidate_block),
                  static_cast<long long>(call->scores_dtype));
    return refuse(call, message);
  }
  if (call->weights == nullptr || call->candidates != nullptr) {
    return refuse(call, "the module scores weighted heads over every visible group");
  }
  if (call->tokens < 0 || call->tokens > INT32_MAX || call->columns <= 0 ||
      call->columns > UINT32_MAX || call->pages <= 0 || call->max_pages <= 0 ||
      call->columns != call->max_pages * kGroups) {
    return refuse(call, "a row's columns are the groups of its table's pages, in 32 bits");
  }
  if (call->tokens == 0) {
    return 0;
  }
  // DeepGEMM's schedule keeps two words of every row in one block's shared memory.
  if ((2 * call->tokens + 1024 / 32 + 1) * 4 > 232448) {
    return refuse(call, "the schedule holds at most 29000 rows a call");
  }
  if (call->multiprocessors <= 0 || call->multiprocessors > kScheduledMultiprocessors) {
    return refuse(call, "the schedule is planned for 1 to 1024 multiprocessors");
  }
  if (!aligned(call->query) || !aligned(call->keys) || !aligned(call->workspace) ||
      call->weights == nullptr || call->page_table == nullptr || call->rows == nullptr ||
      call->visible == nullptr || call->out == nullptr) {
    return refuse(call, "the operands are device addresses, the queries, keys and workspace "
                        "on 16-byte boundaries");
  }

  return launch(call, reinterpret_cast<cudaStream_t>(stream));
}
