// SPDX-License-Identifier: Apache-2.0

// The sparse_attention module's sm100 backend: the operator's call turned into the
// VSA block-sparse SM100 forward pass.
//
// The kernel is FastVideo's block-sparse attention (see ../LICENSE), carried verbatim in
// this package's sources. It owns the hard part — warp-specialized TMA loads, tcgen05 MMA,
// an online softmax with TMEM accumulators, correction warps, cluster-launch-control
// scheduling — and this unit owns the operator's side of the trade: which of the call's
// fields the pass reads, and what the call must say for the compiled configuration to be
// the right one.
//
// Three things the reference binding does and this entry must too, because the kernel
// assumes them and cannot check them:
//
// * the tensor maps address rows and heads through explicit element strides, so every
//   stride must be a multiple of 8 elements and every base address 16-byte aligned (a
//   TMA tensor map is built from them, and cuTensorMapEncodeTiled rejects the rest);
// * channels are contiguous, one head 128 bf16 elements wide — the kernel's own HEAD_DIM;
// * the block size is a compile-time choice of the configuration (VSA_BLK128), not a
//   runtime one, so the call's rows must divide the block this module was built for.
//
// The kernel writes the base-2 logarithm of the softmax normalizer; the operator's call
// promises the natural one, so when the call asks for it a second, tiny kernel converts it
// in place.

// The operator's call carries the tensors' element strides and what the reference binding
// calls BHSD: rows and heads addressed through them, rather than one fixed contiguous
// layout. It must precede the launch header, which defaults it to false.
#define VSA_BHSD true
#include "block_sparse_launch_sm100a.cuh"

#include "uniserve_sparse_attention.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

//: The kernel's own HEAD_DIM; a call with another one is a compilation this module lacks.
constexpr int64_t kHeadDim = 128;

//: The kernel's block size, chosen by the build (VSA_BLK128).
constexpr int64_t kBlock = BLK128 ? 128 : 64;

//: The natural logarithm's base change from the kernel's base-2 normalizer.
constexpr double kLn2 = 0.69314718055994530941723212125898;

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_sparse_attention_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_sparse_attention_call*>(call)->error, sizeof(call->error), "%s",
                message);
  return 1;
}

// A TMA tensor map needs 16-byte aligned bases and strides that are whole 16-byte units in
// bytes; for bf16 that is 8 elements.
bool aligned(const uniserve_sparse_attention_call* call, const void* base, int64_t row_stride,
             int64_t head_stride) {
  return base != nullptr && reinterpret_cast<uintptr_t>(base) % 16 == 0 && row_stride > 0 &&
         head_stride > 0 && row_stride % 8 == 0 && head_stride % 8 == 0 &&
         row_stride <= std::numeric_limits<int32_t>::max() &&
         head_stride <= std::numeric_limits<int32_t>::max();
}

// The normalizer the operator promises, from the one the kernel writes: the kernel's is
// base 2, the operator's is natural, and the two differ by a constant factor.
__global__ void to_natural_log(float* lse, int64_t count, double ln2) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    lse[index] = static_cast<float>(lse[index] * ln2);
  }
}

}  // namespace

extern "C" int32_t uniserve_sparse_attention(const uniserve_sparse_attention_call* call,
                                             void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->dtype != UNISERVE_SPARSE_BF16 || call->output_dtype != UNISERVE_SPARSE_BF16) {
    return refuse(call, "the sm100 module serves bf16 queries, keys, values and output");
  }
  if (call->head_dim != kHeadDim) {
    return refuse(call, "the sm100 module's kernel is compiled for 128 elements per head");
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
  if (!aligned(call, call->query, call->q_row_stride, call->q_head_stride) ||
      !aligned(call, call->out, call->o_row_stride, call->o_head_stride) ||
      !aligned(call, call->key, call->k_row_stride, call->k_head_stride) ||
      !aligned(call, call->value, call->v_row_stride, call->v_head_stride)) {
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
    return refuse(call, "the sm100 kernel's geometry is 32-bit signed; this call does not fit");
  }
  if (query_tiles * call->selected > std::numeric_limits<int32_t>::max() ||
      key_blocks > std::numeric_limits<int32_t>::max()) {
    return refuse(call, "the sm100 kernel's metadata addressing is 32-bit signed; this call does "
                        "not fit");
  }

  BlockSparseVsaArgs args{};
  args.q = static_cast<const __nv_bfloat16*>(call->query);
  args.k = static_cast<const __nv_bfloat16*>(call->key);
  args.v = static_cast<const __nv_bfloat16*>(call->value);
  args.o = static_cast<__nv_bfloat16*>(call->out);
  args.lse = static_cast<float*>(call->lse);
  args.q2k_idx = static_cast<const int*>(call->indices);
  args.q2k_num = static_cast<const int*>(call->counts);
  args.variable_block_sizes = static_cast<const int*>(call->valid);
  // The operator's call is one sequence: its rows are that sequence's rows.
  args.batch = 1;
  args.num_heads = static_cast<int>(call->heads);
  args.seqlen = static_cast<int>(call->query_rows);
  args.key_seqlen = static_cast<int>(call->key_rows);
  args.query_row_stride = call->q_row_stride;
  args.query_head_stride = call->q_head_stride;
  args.output_row_stride = call->o_row_stride;
  args.output_head_stride = call->o_head_stride;
  args.key_row_stride = call->k_row_stride;
  args.key_head_stride = call->k_head_stride;
  args.value_row_stride = call->v_row_stride;
  args.value_head_stride = call->v_head_stride;
  args.head_dim = static_cast<int>(call->head_dim);
  args.num_blocks = static_cast<int>(query_tiles);
  args.max_kv = static_cast<int>(call->selected);
  args.sm_scale = call->scale;

  const cudaStream_t queue = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t status = launch_block_sparse_sm100a(args, queue);
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
