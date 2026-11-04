// SPDX-License-Identifier: Apache-2.0
//
// The C header of the `dense_attention` operator's `f32` backend: the call
// structure `crates/ops/src/attention/dense.rs` hands the module, mirrored field
// for field and in its order.
//
// The Rust side pins the layout (`DENSE_ATTENTION_CALL_LAYOUT`) and asserts every
// offset of it; the assertions below fail this module's build if the two drift
// apart, so a field that moves on one side cannot reach a device.

#ifndef UNISERVE_DENSE_ATTENTION_H
#define UNISERVE_DENSE_ATTENTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// One dense attention call: the operator's `DenseCall`.
///
/// Every pointer is a device address. `cu_seq_lens_q` and `cu_seq_lens_k` are the
/// two offset tables the scheduler writes (`[batch + 1]` each, the last entry the
/// packed token count); they are read on the device, so this module never copies
/// them to the host and a captured call stays capturable.
typedef struct {
  /// `[q_tokens, q_heads, head_dim]` output.
  void* out;
  /// `[q_tokens, q_heads, head_dim]` queries.
  const void* query;
  /// `[kv_tokens, kv_heads, head_dim]` keys.
  const void* key;
  /// `[kv_tokens, kv_heads, head_dim]` values.
  const void* value;
  /// `[batch + 1]` i32 query offsets.
  const int32_t* cu_seq_lens_q;
  /// `[batch + 1]` i32 key offsets.
  const int32_t* cu_seq_lens_k;
  /// Sequences in the call.
  int64_t batch;
  /// Packed query rows.
  int64_t q_tokens;
  /// Packed key rows.
  int64_t kv_tokens;
  /// Rows of the longest query sequence: the module pads to at least this.
  int64_t max_q_len;
  /// Rows of the longest key sequence, which the descriptor bounds.
  int64_t max_kv_len;
  /// Query heads.
  int64_t q_heads;
  /// Key/value heads.
  int64_t kv_heads;
  /// Elements per head.
  int64_t head_dim;
  /// Elements between consecutive query rows.
  int64_t q_stride_tokens;
  /// Elements between consecutive query heads.
  int64_t q_stride_heads;
  /// Elements between consecutive key rows.
  int64_t kv_stride_tokens;
  /// Elements between consecutive key heads.
  int64_t kv_stride_heads;
  /// Elements between consecutive output rows.
  int64_t o_stride_tokens;
  /// Elements between consecutive output heads.
  int64_t o_stride_heads;
  /// The softmax scale.
  double scale;
  /// Inclusive history window; -1 for none.
  int64_t window_left;
  /// 1 when the causal mask applies.
  int64_t causal;
  /// `DType`'s ABI number of the queries.
  int64_t dtype_query;
  /// `DType`'s ABI number of the keys and values.
  int64_t dtype_key_value;
  /// `DType`'s ABI number of the output.
  int64_t dtype_output;
  /// This entry's message on a nonzero return.
  uint8_t error[512];
} UniserveDenseAttentionCall;

// The size and offsets of the operator's `DenseCall`, from
// `attention::dense::DENSE_ATTENTION_CALL_LAYOUT`.
static_assert(sizeof(UniserveDenseAttentionCall) == 720,
              "the dense attention call structure is 720 bytes");
static_assert(offsetof(UniserveDenseAttentionCall, out) == 0,
              "the dense attention call's `out` moved");
static_assert(offsetof(UniserveDenseAttentionCall, query) == 8,
              "the dense attention call's `query` moved");
static_assert(offsetof(UniserveDenseAttentionCall, key) == 16,
              "the dense attention call's `key` moved");
static_assert(offsetof(UniserveDenseAttentionCall, value) == 24,
              "the dense attention call's `value` moved");
static_assert(offsetof(UniserveDenseAttentionCall, cu_seq_lens_q) == 32,
              "the dense attention call's `cu_seq_lens_q` moved");
static_assert(offsetof(UniserveDenseAttentionCall, cu_seq_lens_k) == 40,
              "the dense attention call's `cu_seq_lens_k` moved");
static_assert(offsetof(UniserveDenseAttentionCall, batch) == 48,
              "the dense attention call's `batch` moved");
static_assert(offsetof(UniserveDenseAttentionCall, max_kv_len) == 80,
              "the dense attention call's `max_kv_len` moved");
static_assert(offsetof(UniserveDenseAttentionCall, head_dim) == 104,
              "the dense attention call's `head_dim` moved");
static_assert(offsetof(UniserveDenseAttentionCall, q_stride_tokens) == 112,
              "the dense attention call's query strides moved");
static_assert(offsetof(UniserveDenseAttentionCall, scale) == 160,
              "the dense attention call's `scale` moved");
static_assert(offsetof(UniserveDenseAttentionCall, dtype_query) == 184,
              "the dense attention call's `dtype_query` moved");
static_assert(offsetof(UniserveDenseAttentionCall, error) == 208,
              "the dense attention call's `error` moved");

/// The entry one compiled module exports.
///
/// It returns 0 when the attention ran, and 1 with this entry's message in the
/// call's `error` otherwise. `stream` is the caller's `cudaStream_t`, as
/// `void*`.
typedef int32_t (*UniserveDenseAttentionEntry)(
    UniserveDenseAttentionCall* call, void* stream);

#ifdef __cplusplus
}
#endif

#endif  // UNISERVE_DENSE_ATTENTION_H
