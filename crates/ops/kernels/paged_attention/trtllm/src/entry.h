// SPDX-License-Identifier: Apache-2.0

// The C entry of the paged_attention module's trtllm backend: the operator's
// `attention::paged::Call` (crates/ops/src/attention/mod.rs,
// PAGED_ATTENTION_CALL_LAYOUT), field for field, and the entry the operator's
// Rust side calls.

#pragma once

#include <cstddef>
#include <cstdint>

// One paged attention launch: prefill or decode over the pages of the cache.
// The call struct is the operator's, byte for byte; `stream` is the CUstream of
// the device the executor drives.
//
// `query` and `out` are [tokens, q_heads, head_dim]; `key` and `value`
// [pages, page_size, kv_heads, head_dim] in the layout `kv_write` wrote;
// `key_scale` and `value_scale` are [pages, 1, 1, 1] f32 and null for an
// unquantized cache; `page_table` is [batch, max_pages] i32, `seq_lens` [batch]
// i32 and `cum_seq_lens_q` [batch + 1] i32 (null at decode). `workspace` is the
// program's transient arena of `workspace_bytes`.
struct uniserve_paged_attention_call {
  void* out;
  const void* query;
  const void* key;
  const void* value;
  const void* key_scale;
  const void* value_scale;
  void* workspace;
  const int32_t* page_table;
  const int32_t* seq_lens;
  const int32_t* cum_seq_lens_q;
  int64_t batch;
  int64_t tokens;
  int64_t max_q_len;
  int64_t max_kv_len;
  int64_t num_pages;
  int64_t page_size;
  int64_t q_heads;
  int64_t kv_heads;
  int64_t head_dim;
  int64_t q_stride_tokens;
  int64_t q_stride_heads;
  int64_t kv_stride_page;
  int64_t kv_stride_tokens;
  int64_t kv_stride_heads;
  int64_t o_stride_tokens;
  int64_t o_stride_heads;
  double scale;
  int64_t window_left;
  int64_t causal;
  int64_t workspace_bytes;
  int64_t dtype_query;
  int64_t dtype_key_value;
  int64_t dtype_output;
  int64_t phase;
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

// The phase of the call, as the operator numbers it.
enum uniserve_phase : int64_t {
  UNISERVE_PHASE_PREFILL = 0,
  UNISERVE_PHASE_DECODE = 1,
};

// Every field is asserted, not one: two fields of the same size can be swapped between this
// header and the operator without a size assert noticing, and the operator reads them by
// name. The Rust side reads these literals (crates/ops/src/attention/mod.rs,
// `the_package_header_declares_this_call_layout`), so a layout that moves on one side fails
// on the other rather than at run time.
static_assert(offsetof(uniserve_paged_attention_call, out) == 0,
              "the call's out must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, query) == 8,
              "the call's query must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, key) == 16,
              "the call's key must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, value) == 24,
              "the call's value must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, key_scale) == 32,
              "the call's key_scale must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, value_scale) == 40,
              "the call's value_scale must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, workspace) == 48,
              "the call's workspace must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, page_table) == 56,
              "the call's page_table must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, seq_lens) == 64,
              "the call's seq_lens must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, cum_seq_lens_q) == 72,
              "the call's cum_seq_lens_q must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, batch) == 80,
              "the call's batch must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, tokens) == 88,
              "the call's tokens must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, max_q_len) == 96,
              "the call's max_q_len must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, max_kv_len) == 104,
              "the call's max_kv_len must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, num_pages) == 112,
              "the call's num_pages must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, page_size) == 120,
              "the call's page_size must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, q_heads) == 128,
              "the call's q_heads must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, kv_heads) == 136,
              "the call's kv_heads must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, head_dim) == 144,
              "the call's head_dim must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, q_stride_tokens) == 152,
              "the call's q_stride_tokens must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, q_stride_heads) == 160,
              "the call's q_stride_heads must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, kv_stride_page) == 168,
              "the call's kv_stride_page must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, kv_stride_tokens) == 176,
              "the call's kv_stride_tokens must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, kv_stride_heads) == 184,
              "the call's kv_stride_heads must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, o_stride_tokens) == 192,
              "the call's o_stride_tokens must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, o_stride_heads) == 200,
              "the call's o_stride_heads must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, scale) == 208,
              "the call's scale must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, window_left) == 216,
              "the call's window_left must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, causal) == 224,
              "the call's causal must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, workspace_bytes) == 232,
              "the call's workspace_bytes must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, dtype_query) == 240,
              "the call's dtype_query must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, dtype_key_value) == 248,
              "the call's dtype_key_value must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, dtype_output) == 256,
              "the call's dtype_output must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, phase) == 264,
              "the call's phase must sit where the operator's layout puts it");
static_assert(offsetof(uniserve_paged_attention_call, error) == 272,
              "the call's error must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_paged_attention_call) == 784,
              "the call must be the size of the operator's layout");

// Run one paged attention call. Returns 0, or a nonzero status with a
// NUL-terminated message in `call->error`.
extern "C" int32_t uniserve_paged_attention(const uniserve_paged_attention_call* call, void* stream);
