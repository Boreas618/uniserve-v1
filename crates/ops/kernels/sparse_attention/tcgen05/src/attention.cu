// SPDX-License-Identifier: Apache-2.0

// The sparse_attention module's tcgen05 backend: the operator's call launched as
// FastVideo's block-sparse attention forward pass for data-center Blackwell.
//
// The kernel is FastVideo's own source, the `fastvideo` pin of the Build Server's pinned
// sources, with the package's patch for an odd number of query tiles
// (`patches/unpaired-tile.patch`). It owns the hard part — warp-specialized TMA loads,
// tcgen05 MMA, an online softmax with TMEM accumulators, correction warps,
// cluster-launch-control scheduling. This unit owns the host side of the launch, which
// FastVideo's own launch states only for its two fixed layouts:
//
// * the tensor maps address rows and heads through the call's element strides, each of Q,
//   O, K and V its own, so every stride must be a multiple of 8 elements and every base
//   address 16-byte aligned (cuTensorMapEncodeTiled rejects the rest);
// * the keys have an extent of their own: the K and V maps span the call's key rows, while
//   the kernel's `seqlen` is the query rows (with one sample, the kernel's key offset of a
//   sample is zero);
// * a head's query tiles may be odd in number: the trailing tile is a work item whose
//   partner tile is absent, and the Q and O maps end at the query rows, so the partner's
//   rows load as zeros and are not stored;
// * channels are contiguous, one head 128 bf16 elements wide — the kernel's own HEAD_DIM —
//   and the block size is a compile-time choice of the module (`VSA_BLK128`), so the
//   call's rows must divide the block this module was built for.
//
// The kernel writes the base-2 logarithm of the softmax normalizer; the operator's call
// promises the natural one, so when the call asks for it a second, tiny kernel converts it
// in place.

#include <block_sparse_kernel_blackwell.cuh>
#include <uniserve_sparse_attention.h>

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

// The kernel's names live in a namespace its header names after the block size.
namespace vsa = VSA_NAMESPACE;

//: The kernel's own HEAD_DIM; a call with another one is a compilation this module lacks.
constexpr int64_t kHeadDim = vsa::HEAD_DIM;

//: The kernel's block size, chosen by the build (VSA_BLK128).
constexpr int64_t kBlock = vsa::BLOCK;

//: The natural logarithm's base change from the kernel's base-2 normalizer.
constexpr double kLn2 = 0.69314718055994530941723212125898;

//: Bytes of one bf16 element: a tensor map's strides are in bytes.
constexpr uint64_t kElementBytes = sizeof(__nv_bfloat16);

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_sparse_attention_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_sparse_attention_call*>(call)->error, sizeof(call->error), "%s",
                message);
  return 1;
}

// A TMA tensor map needs 16-byte aligned bases and strides that are whole 16-byte units in
// bytes; for bf16 that is 8 elements.
bool aligned(const void* base, int64_t row_stride, int64_t head_stride) {
  return base != nullptr && reinterpret_cast<uintptr_t>(base) % 16 == 0 && row_stride > 0 &&
         head_stride > 0 && row_stride % 8 == 0 && head_stride % 8 == 0 &&
         row_stride <= std::numeric_limits<int32_t>::max() &&
         head_stride <= std::numeric_limits<int32_t>::max();
}

// Encodes a four-dimensional bf16 tensor map with the kernel's 128-byte swizzle (a box row is
// at most 128 bytes, so a head's 128 channels are addressed as subtiles of SUB_COLS_BF16).
// Extents and box are listed innermost first; strides are in bytes, for dimensions 1 to 3.
bool encode(CUtensorMap* map, const void* base, const cuuint64_t (&extents)[4],
            const cuuint64_t (&strides)[3], const cuuint32_t (&box)[4]) {
  const cuuint32_t steps[4] = {1, 1, 1, 1};
  return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base),
                                extents, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// The normalizer the operator promises, from the one the kernel writes: the kernel's is
// base 2, the operator's is natural, and the two differ by a constant factor.
__global__ void to_natural_log(float* lse, int64_t count, double ln2) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    lse[index] = static_cast<float>(lse[index] * ln2);
  }
}

// Launches the kernel over the call's one sequence: `heads` heads of `query_tiles` query tiles
// against `key_rows` key rows, each tile attending the `selected` key blocks its row of the
// selection names. The caller has checked the geometry, the strides and the pointers; a
// tensor map the driver cannot encode, or a launch the runtime refuses, is the returned
// status.
cudaError_t launch(const uniserve_sparse_attention_call* call, int heads, int query_tiles,
                   int key_rows, int selected, cudaStream_t stream) {
  const int query_rows = query_tiles * static_cast<int>(kBlock);
  // A work item is a pair of adjacent query tiles of one head; an odd count leaves the
  // trailing tile of each head a work item of its own.
  const int pairs = (query_tiles + 1) / 2;
  const int work_items = heads * pairs;

  // Q and O: [channel in subtile, head, row, subtile]. One box is one head's M_TILE rows with
  // all subtiles.
  CUtensorMap query_map, output_map, key_map, value_map;
  {
    const cuuint64_t extents[4] = {vsa::SUB_COLS_BF16, static_cast<cuuint64_t>(heads),
                                   static_cast<cuuint64_t>(query_rows), vsa::Q_SUBTILES};
    const cuuint32_t box[4] = {vsa::SUB_COLS_BF16, 1, vsa::M_TILE, vsa::Q_SUBTILES};
    const cuuint64_t query_strides[3] = {call->q_head_stride * kElementBytes,
                                         call->q_row_stride * kElementBytes,
                                         vsa::SUB_COLS_BF16 * kElementBytes};
    const cuuint64_t output_strides[3] = {call->o_head_stride * kElementBytes,
                                          call->o_row_stride * kElementBytes,
                                          vsa::SUB_COLS_BF16 * kElementBytes};
    if (!encode(&query_map, call->query, extents, query_strides, box) ||
        !encode(&output_map, call->out, extents, output_strides, box)) {
      return cudaErrorInvalidValue;
    }
  }
  // K and V: [channel in subtile, key row, subtile, head]. One box is one BLOCK-row key
  // block of one subtile (all subtiles in the 128-token build).
  {
    const cuuint64_t extents[4] = {vsa::SUB_COLS_BF16, static_cast<cuuint64_t>(key_rows),
                                   vsa::HEAD_DIM / vsa::SUB_COLS_BF16,
                                   static_cast<cuuint64_t>(heads)};
    const cuuint32_t box[4] = {vsa::SUB_COLS_BF16, vsa::BLOCK,
                               vsa::BLK128 ? static_cast<cuuint32_t>(vsa::K_SUBTILES) : 1u, 1};
    const cuuint64_t key_strides[3] = {call->k_row_stride * kElementBytes,
                                       vsa::SUB_COLS_BF16 * kElementBytes,
                                       call->k_head_stride * kElementBytes};
    const cuuint64_t value_strides[3] = {call->v_row_stride * kElementBytes,
                                         vsa::SUB_COLS_BF16 * kElementBytes,
                                         call->v_head_stride * kElementBytes};
    if (!encode(&key_map, call->key, extents, key_strides, box) ||
        !encode(&value_map, call->value, extents, value_strides, box)) {
      return cudaErrorInvalidValue;
    }
  }

  // The dynamic shared memory of the kernel's carve-up: two Q tiles, the K/V ring, two BF16 O
  // buffer tiles, the K/V ring mbarriers plus 22 pipeline mbarriers, the CLC barriers and
  // responses with 16 bytes of response alignment, the TMEM address slot, the softmax
  // statistics and 256 spare bytes, as FastVideo's launch sizes it.
  const size_t shared = static_cast<size_t>(2) * vsa::Q_TILE_BYTES +
                        vsa::NUM_KV_STAGES * vsa::KV_RING_SLOT_BYTES +
                        static_cast<size_t>(2) * vsa::M_TILE * vsa::HEAD_DIM * kElementBytes +
                        (2 * vsa::NUM_KV_STAGES + 22) * 8 +
                        static_cast<size_t>(vsa::CLC_STAGES) * (2 * 8 + 16) + 16 + 8 +
                        static_cast<size_t>(2) * vsa::STAT_REGIONS * vsa::STATS * sizeof(float) +
                        256;

  // FastVideo's configuration of the kernel: scores loaded 32 TMEM columns at a time, the
  // softmax-to-correction signals on mbarriers, emulated exp2 for a share of the scores, P
  // published in two parts, no throttling of the correction warps, cluster-launch-control
  // scheduling, query-tile-major rasterization, a rescale threshold of 8 in scaled log2
  // units, and tensor maps addressed by head and row (BHSD).
  const auto kernel = &vsa::fmha_context_bf16_gen_kernel<
      /*S_LD_COLS=*/32, /*FULL_NAMED_BAR=*/false, /*EX2_EMU=*/true, /*SPLIT_P=*/true,
      /*SOFTMAX_THROTTLE=*/false, /*USE_CLC=*/true, /*Q_RASTER=*/true, /*MHA=*/true,
      /*RESCALE_THRESHOLD=*/8, /*BHSD=*/true>;
  const cudaError_t limited = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (limited != cudaSuccess) {
    return limited;
  }

  // Packed divisors (make_magic) of the per-work-item divisions the kernel makes.
  const unsigned long long per_sample = make_magic(static_cast<unsigned>(heads * pairs));
  const unsigned long long per_head = make_magic(static_cast<unsigned>(heads));
  const unsigned long long per_pair = make_magic(static_cast<unsigned>(pairs));
  // The kernel evaluates exp2, so the softmax scale is folded with log2(e).
  const float scale_log2 = call->scale * static_cast<float>(M_LOG2E);

  // With cluster-launch control the grid holds one CTA per work item, and running CTAs cancel
  // unlaunched ones to take over their work items.
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(work_items), 1, 1);
  config.blockDim = dim3(vsa::N_WARPS * 32, 1, 1);
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  cudaLaunchAttribute cluster[1];
  cluster[0].id = cudaLaunchAttributeClusterDimension;
  cluster[0].val.clusterDim.x = 1;
  cluster[0].val.clusterDim.y = 1;
  cluster[0].val.clusterDim.z = 1;
  config.attrs = cluster;
  config.numAttrs = 1;
  // The kernel's `tmap_v_t` parameter is unread by its body; it is handed the value map. The
  // call is one sample.
  return cudaLaunchKernelEx(&config, kernel, query_map, key_map, value_map, value_map, output_map,
                            query_rows, heads, scale_log2, /*num_samples=*/1, query_tiles, pairs,
                            selected, per_sample, per_head, per_pair,
                            static_cast<const int*>(call->indices),
                            static_cast<const int*>(call->counts),
                            static_cast<const int*>(call->valid), static_cast<float*>(call->lse));
}

}  // namespace

extern "C" int32_t uniserve_sparse_attention(const uniserve_sparse_attention_call* call,
                                             void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->dtype != UNISERVE_DTYPE_BF16 || call->output_dtype != UNISERVE_DTYPE_BF16) {
    return refuse(call, "the tcgen05 module serves bf16 queries, keys, values and output");
  }
  if (call->head_dim != kHeadDim) {
    return refuse(call, "the tcgen05 module's kernel is compiled for 128 elements per head");
  }
  if (call->heads <= 0 || call->query_rows <= 0 || call->key_rows <= 0) {
    return refuse(call, "sparse_attention needs at least one head and one query tile and key block");
  }
  if (call->query_rows % kBlock != 0 || call->key_rows % kBlock != 0) {
    return refuse(call,
                  "query and key rows must divide this module's sparse block size; a partial tile "
                  "is the caller's padding, addressed by the valid-size array");
  }
  if (call->selected <= 0) {
    return refuse(call, "sparse_attention needs a positive selection width");
  }
  if (!(call->scale > 0.0f) || !std::isfinite(call->scale)) {
    return refuse(call, "sparse_attention needs a positive, finite softmax scale");
  }
  if (!aligned(call->query, call->q_row_stride, call->q_head_stride) ||
      !aligned(call->out, call->o_row_stride, call->o_head_stride) ||
      !aligned(call->key, call->k_row_stride, call->k_head_stride) ||
      !aligned(call->value, call->v_row_stride, call->v_head_stride)) {
    return refuse(call,
                  "query, key, value and out need 16-byte aligned bases and row and head strides "
                  "that are multiples of 8 elements");
  }
  if (call->indices == nullptr || call->counts == nullptr || call->valid == nullptr) {
    return refuse(call, "sparse_attention reads the selection from the device; it needs indices, "
                        "counts and the valid-size array");
  }
  // The metadata arrays are read by the kernel only, never on the host, so this module
  // checks their presence and nothing of their contents.
  const int64_t query_tiles = call->query_rows / kBlock;
  const int64_t key_blocks = call->key_rows / kBlock;
  if (call->heads > std::numeric_limits<int32_t>::max() ||
      call->query_rows > std::numeric_limits<int32_t>::max() ||
      call->key_rows > std::numeric_limits<int32_t>::max() ||
      call->selected > std::numeric_limits<int32_t>::max()) {
    return refuse(call, "the tcgen05 kernel's geometry is 32-bit signed; this call does not fit");
  }
  if (query_tiles * call->selected > std::numeric_limits<int32_t>::max() ||
      key_blocks > std::numeric_limits<int32_t>::max()) {
    return refuse(call, "the tcgen05 kernel's metadata addressing is 32-bit signed; this call "
                        "does not fit");
  }

  const cudaStream_t queue = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t status =
      launch(call, static_cast<int>(call->heads), static_cast<int>(query_tiles),
             static_cast<int>(call->key_rows), static_cast<int>(call->selected), queue);
  if (status != cudaSuccess) {
    std::snprintf(const_cast<uniserve_sparse_attention_call*>(call)->error, sizeof(call->error),
                  "the VSA kernel refused this launch: %s", cudaGetErrorString(status));
    return 1;
  }
  if (call->lse != nullptr) {
    // The operator promises the natural logarithm of the softmax normalizer.
    const int64_t rows = call->heads * call->query_rows;
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    to_natural_log<<<blocks, threads, 0, queue>>>(static_cast<float*>(call->lse), rows, kLn2);
    const cudaError_t converted = cudaGetLastError();
    if (converted != cudaSuccess) {
      std::snprintf(const_cast<uniserve_sparse_attention_call*>(call)->error, sizeof(call->error),
                    "the normalizer conversion failed: %s", cudaGetErrorString(converted));
      return 1;
    }
  }
  return 0;
}
