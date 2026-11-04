// SPDX-License-Identifier: Apache-2.0

// The C entry of the dense_attention module's fa4 backend: the operator's
// `attention::dense::Call` (crates/ops/src/attention/dense.rs,
// DENSE_ATTENTION_CALL_LAYOUT), field for field, and the entry the operator's
// Rust side calls.

#pragma once

#include <cstddef>
#include <cstdint>

// One dense attention launch over packed sequences: no cache and no page table, each
// sequence's rows a contiguous run of the packed tensors. `stream` is the CUstream of
// the device the executor drives.
//
// `query` is [q_tokens, q_heads, head_dim], `key` and `value`
// [kv_tokens, kv_heads, head_dim], `out` [q_tokens, q_heads, head_dim], and the two
// tables [batch + 1] i32 giving each sequence's first row and then the token count.
struct uniserve_dense_attention_call {
  void* out;
  const void* query;
  const void* key;
  const void* value;
  const int32_t* cu_seq_lens_q;
  const int32_t* cu_seq_lens_k;
  int64_t batch;
  int64_t q_tokens;
  int64_t kv_tokens;
  int64_t max_q_len;
  int64_t max_kv_len;
  int64_t q_heads;
  int64_t kv_heads;
  int64_t head_dim;
  int64_t q_stride_tokens;
  int64_t q_stride_heads;
  int64_t kv_stride_tokens;
  int64_t kv_stride_heads;
  int64_t o_stride_tokens;
  int64_t o_stride_heads;
  double scale;
  int64_t window_left;
  int64_t causal;
  int64_t dtype_query;
  int64_t dtype_key_value;
  int64_t dtype_output;
  char error[512];
};

// The operator crate's `DType::as_i64()` numbering.
enum uniserve_dtype : int64_t {
  UNISERVE_DTYPE_U8 = 0,
  UNISERVE_DTYPE_I32 = 1,
  UNISERVE_DTYPE_I64 = 2,
  UNISERVE_DTYPE_F16 = 3,
  UNISERVE_DTYPE_BF16 = 4,
  UNISERVE_DTYPE_F32 = 5,
  UNISERVE_DTYPE_F8E4M3 = 6,
  UNISERVE_DTYPE_F4E2M1 = 7,
};

static_assert(offsetof(uniserve_dense_attention_call, batch) == 48,
              "the call's batch must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_dense_attention_call, scale) == 160,
              "the call's scale must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_dense_attention_call, error) == 208,
              "the call's error message must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_dense_attention_call) == 720,
              "the call must be the size of the operator's layout");

// Run one dense attention call. Returns 0, or a nonzero status with a
// NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_dense_attention(const uniserve_dense_attention_call* call, void* stream);
