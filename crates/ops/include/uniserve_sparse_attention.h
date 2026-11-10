// SPDX-License-Identifier: Apache-2.0

// The C entry of the sparse_attention module's sm100 backend: the operator's
// `attention::sparse::SparseCall` (crates/ops/src/attention/sparse.rs,
// SPARSE_CALL_LAYOUT), field for field, and the entry the operator's Rust side calls.

#pragma once

#include <cstddef>
#include <cstdint>

// One block-sparse attention launch over selected key blocks.
//
// `query` and `out` are [query_rows, heads, head_dim] and `key`, `value`
// [key_rows, heads, head_dim], rows addressed by the strides the call gives. Rows are whole
// tiles of `block`: the descriptor's block is the module's compilation, so a call whose rows
// do not divide it is refused. `indices` is [heads, query_tiles, selected] i32 of global
// key-block ids, `counts` [heads, query_tiles] i32 of how many of each row are read, and
// `valid` [key_blocks] i32 of the valid rows of each block. `lse`, when not null, is
// [heads, query_rows] f32.
struct uniserve_sparse_attention_call {
  const void* query;
  const void* key;
  const void* value;
  void* out;
  void* lse;
  const void* indices;
  const void* counts;
  const void* valid;
  int64_t heads;
  int64_t query_rows;
  int64_t key_rows;
  int64_t q_row_stride;
  int64_t q_head_stride;
  int64_t o_row_stride;
  int64_t o_head_stride;
  int64_t k_row_stride;
  int64_t k_head_stride;
  int64_t v_row_stride;
  int64_t v_head_stride;
  int64_t selected;
  int64_t head_dim;
  int64_t dtype;
  int64_t output_dtype;
  float scale;
  char error[512];
};

// The operator crate's `DType::as_i64()` numbering.
enum uniserve_sparse_dtype : int64_t {
  UNISERVE_SPARSE_U8 = 0,
  UNISERVE_SPARSE_I32 = 1,
  UNISERVE_SPARSE_I64 = 2,
  UNISERVE_SPARSE_F16 = 3,
  UNISERVE_SPARSE_BF16 = 4,
  UNISERVE_SPARSE_F32 = 5,
  UNISERVE_SPARSE_F8E4M3 = 6,
  UNISERVE_SPARSE_F4E2M1 = 7,
};

static_assert(offsetof(uniserve_sparse_attention_call, valid) == 56,
              "the call's valid-size array must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_sparse_attention_call, heads) == 64,
              "the call's geometry must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_sparse_attention_call, selected) == 152,
              "the call's selection width must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_sparse_attention_call, scale) == 184,
              "the call's scale must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_sparse_attention_call, error) == 188,
              "the call's error message must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_sparse_attention_call) == 704,
              "the call must be the size of the operator's layout");

// Run one block-sparse attention call. Returns 0, or a nonzero status with a
// NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_sparse_attention(const uniserve_sparse_attention_call* call,
                                             void* stream);
