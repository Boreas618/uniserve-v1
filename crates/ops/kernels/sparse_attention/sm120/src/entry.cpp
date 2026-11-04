// SPDX-License-Identifier: Apache-2.0

// The sparse_attention module's sm120 backend: the operator's call turned into FlashInfer's SM120
// block-sparse attention kernel.
//
// The kernel is FlashInfer's, compiled at build time and exported to C (see `src/bsa_sm120.py`);
// this unit owns the operator's side of the trade — which of the call's fields the kernel reads,
// and what the call must say for the compiled configuration to be the right one.
//
// The kernel's layouts are the call's layouts with batch 1: a packed `[rows, heads, 128]` view with
// explicit row and head strides is its `(rows, 128, heads, 1)`, and the selection's
// `[heads, query_tiles, selected]` is its `(selected, query_tiles, heads, 1)`. Nothing is copied and
// no host-side permutation exists; what the kernel gains is the strides, which it takes at run time.

#include "flash_attn_sparse_sm120.h"

#include "entry.h"

#include <cstdio>
#include <mutex>

namespace {

//: The kernel's tile in both the query and the key domain, and the width of its head.
constexpr int64_t kBlock = 64;
constexpr int64_t kHeadDim = 128;

//: The kernels' assumed alignment of every tensor the TMA engine reads or writes, in bytes; the
//: normalizer and the selection arrays carry the vendor's weaker assumptions.
constexpr uintptr_t kTensorAlignment = 128;
constexpr uintptr_t kScalarAlignment = 4;

// The exported module holds the kernel's cubin and its host launcher.
flash_attn_sparse_sm120_Kernel_Module_t& module() {
  static flash_attn_sparse_sm120_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { flash_attn_sparse_sm120_Kernel_Module_Load(&handle); });
  return handle;
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_sparse_attention_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_sparse_attention_call*>(call)->error, sizeof(call->error), "%s",
                message);
  return 1;
}

// The vendor's assumed alignment: the kernel may load a tensor's row in 128-byte vectors.
bool aligned(const void* base, uintptr_t alignment) {
  return base != nullptr && reinterpret_cast<uintptr_t>(base) % alignment == 0;
}

}  // namespace

extern "C" int32_t uniserve_sparse_attention(const uniserve_sparse_attention_call* call,
                                             void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->dtype != UNISERVE_SPARSE_BF16 && call->dtype != UNISERVE_SPARSE_F16) {
    return refuse(call, "the sm120 module serves bf16 and f16 queries, keys, values and output");
  }
  if (call->dtype != call->output_dtype) {
    return refuse(call, "the sm120 kernel writes the output in the queries' element type");
  }
  if (call->head_dim != kHeadDim) {
    return refuse(call, "the sm120 kernel is compiled for 128 elements per head");
  }
  if (call->lse == nullptr) {
    return refuse(call,
                  "the sm120 kernel always writes the softmax normalizer, so the call must carry "
                  "its buffer");
  }
  if (call->heads <= 0 || call->query_rows <= 0 || call->key_rows <= 0) {
    return refuse(call, "sparse_attention needs at least one head and one query tile and key block");
  }
  if (call->query_rows % kBlock != 0 || call->key_rows % kBlock != 0) {
    return refuse(call,
                  "query and key rows must be whole 64-row tiles; a partial tile is the caller's "
                  "padding, addressed by the valid-size array");
  }
  if (call->selected <= 0) {
    return refuse(call, "sparse_attention needs a positive selection width");
  }
  if (!(call->scale > 0.0f)) {
    return refuse(call, "sparse_attention needs a positive softmax scale");
  }
  if (!aligned(call->query, kTensorAlignment) || !aligned(call->key, kTensorAlignment) ||
      !aligned(call->value, kTensorAlignment) || !aligned(call->out, kTensorAlignment)) {
    return refuse(call,
                  "the sm120 kernel reads and writes 128-byte aligned rows; query, key, value and "
                  "out must be aligned to 128 bytes");
  }
  if (!aligned(call->lse, kScalarAlignment) || !aligned(call->indices, kScalarAlignment) ||
      !aligned(call->counts, kScalarAlignment) || !aligned(call->valid, kScalarAlignment)) {
    return refuse(call,
                  "the selection arrays and the normalizer must be aligned to 4 bytes");
  }
  if (call->q_head_stride <= 0 || call->k_head_stride <= 0 || call->v_head_stride <= 0 ||
      call->o_head_stride <= 0 || call->q_row_stride <= 0 || call->k_row_stride <= 0 ||
      call->v_row_stride <= 0 || call->o_row_stride <= 0) {
    return refuse(call, "every row and head stride of the four tensors must be positive");
  }

  const int64_t query_tiles = call->query_rows / kBlock;
  const int64_t key_blocks = call->key_rows / kBlock;
  const int64_t heads = call->heads;

  // Q, K and O: (rows, 128, heads, 1) with the head dimension contiguous.
  flash_attn_sparse_sm120_Tensor_mQ_t query{};
  query.data = const_cast<void*>(call->query);
  query.dynamic_shapes[0] = static_cast<int32_t>(call->query_rows);
  query.dynamic_shapes[1] = static_cast<int32_t>(kHeadDim);
  query.dynamic_shapes[2] = static_cast<int32_t>(heads);
  query.dynamic_shapes[3] = 1;
  query.dynamic_strides[0] = 1;
  query.dynamic_strides[1] = call->q_head_stride;
  query.dynamic_strides[2] = call->query_rows * call->q_row_stride;

  flash_attn_sparse_sm120_Tensor_mK_t key{};
  key.data = const_cast<void*>(call->key);
  key.dynamic_shapes[0] = static_cast<int32_t>(call->key_rows);
  key.dynamic_shapes[1] = static_cast<int32_t>(kHeadDim);
  key.dynamic_shapes[2] = static_cast<int32_t>(heads);
  key.dynamic_shapes[3] = 1;
  key.dynamic_strides[0] = 1;
  key.dynamic_strides[1] = call->k_head_stride;
  key.dynamic_strides[2] = call->key_rows * call->k_row_stride;

  // V: (128, key_rows, heads, 1), addressed the same way.
  flash_attn_sparse_sm120_Tensor_mV_t value{};
  value.data = const_cast<void*>(call->value);
  value.dynamic_shapes[0] = static_cast<int32_t>(kHeadDim);
  value.dynamic_shapes[1] = static_cast<int32_t>(call->key_rows);
  value.dynamic_shapes[2] = static_cast<int32_t>(heads);
  value.dynamic_shapes[3] = 1;
  value.dynamic_strides[0] = call->v_row_stride;
  value.dynamic_strides[1] = call->v_head_stride;
  value.dynamic_strides[2] = call->key_rows * call->v_row_stride;

  flash_attn_sparse_sm120_Tensor_mO_t out{};
  out.data = call->out;
  out.dynamic_shapes[0] = static_cast<int32_t>(call->query_rows);
  out.dynamic_shapes[1] = static_cast<int32_t>(kHeadDim);
  out.dynamic_shapes[2] = static_cast<int32_t>(heads);
  out.dynamic_shapes[3] = 1;
  out.dynamic_strides[0] = 1;
  out.dynamic_strides[1] = call->o_head_stride;
  out.dynamic_strides[2] = call->query_rows * call->o_row_stride;

  // The normalizer: (rows, heads, 1), the row contiguous, which is the operator's [heads, rows].
  flash_attn_sparse_sm120_Tensor_mLSE_t lse{};
  lse.data = call->lse;
  lse.dynamic_shapes[0] = static_cast<int32_t>(call->query_rows);
  lse.dynamic_shapes[1] = static_cast<int32_t>(heads);
  lse.dynamic_shapes[2] = 1;
  lse.dynamic_strides[0] = call->query_rows;
  lse.dynamic_strides[1] = 1;

  // The selection: (selected, query_tiles, heads, 1) and (query_tiles, heads, 1).
  flash_attn_sparse_sm120_Tensor_blocksparse_indices_q2k_t indices{};
  indices.data = const_cast<void*>(call->indices);
  indices.dynamic_shapes[0] = static_cast<int32_t>(call->selected);
  indices.dynamic_shapes[1] = static_cast<int32_t>(query_tiles);
  indices.dynamic_shapes[2] = static_cast<int32_t>(heads);
  indices.dynamic_shapes[3] = 1;
  indices.dynamic_strides[0] = call->selected;
  indices.dynamic_strides[1] = query_tiles * call->selected;
  indices.dynamic_strides[2] = heads * query_tiles * call->selected;

  flash_attn_sparse_sm120_Tensor_blocksparse_num_blocks_q2k_t counts{};
  counts.data = const_cast<void*>(call->counts);
  counts.dynamic_shapes[0] = static_cast<int32_t>(query_tiles);
  counts.dynamic_shapes[1] = static_cast<int32_t>(heads);
  counts.dynamic_shapes[2] = 1;
  counts.dynamic_strides[0] = query_tiles;
  counts.dynamic_strides[1] = heads * query_tiles;

  // The valid rows of each key block: one array, one entry per block.
  flash_attn_sparse_sm120_Tensor_blocksparse_varblk_t valid{};
  valid.data = const_cast<void*>(call->valid);
  valid.dynamic_shapes[0] = static_cast<int32_t>(key_blocks);

  const int32_t status = cute_dsl_flash_attn_sparse_sm120_wrapper(
      &module(), &query, &key, &value, &out, &lse, &indices, &counts,
      static_cast<int32_t>(call->selected), &valid, call->scale,
      reinterpret_cast<cudaStream_t>(stream));
  if (status != 0) {
    return refuse(call, "FlashInfer's SM120 block-sparse launch reported an error");
  }
  return 0;
}
