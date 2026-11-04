// SPDX-License-Identifier: Apache-2.0

// The dense_attention module's fa4 backend: the operator's call turned into
// FlashAttention 4's dense forward pass.
//
// The kernel is FlashAttention's: `build.py` compiled the configuration the vendor's own
// selection chose for this descriptor's geometry and exported it to C, so this unit's
// work is the operator's side of the trade — which of the call's fields the pass reads,
// and what the call must say for the compiled configuration to be the right one. The
// exported tensors are the vendor's own shapes: packed sequences
// (`[tokens, heads, head_dim]`), which the two offset arrays divide into sequences.
//
// The pass reads no cross-CTA counter and no workspace: FA4's scheduler is its own.

#include "flash_attn_dense.h"

#include "entry.h"

#include <cstdio>
#include <mutex>

namespace {

// The exported module holds the kernel's cubin and its host launcher.
flash_attn_dense_Kernel_Module_t& module() {
  static flash_attn_dense_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { flash_attn_dense_Kernel_Module_Load(&handle); });
  return handle;
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_dense_attention_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_dense_attention_call*>(call)->error, sizeof(call->error), "%s",
                message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_dense_attention(const uniserve_dense_attention_call* call,
                                            void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->dtype_query != call->dtype_key_value || call->dtype_query != call->dtype_output) {
    return refuse(call, "the fa4 module serves one element type for query, key, value and output");
  }
  if (call->dtype_query != UNISERVE_DTYPE_BF16 && call->dtype_query != UNISERVE_DTYPE_F16) {
    return refuse(call, "the fa4 module serves bf16 and f16");
  }
  if (call->query == nullptr || call->key == nullptr || call->value == nullptr ||
      call->out == nullptr) {
    return refuse(call, "dense_attention needs query, key, value and out");
  }
  if (call->cu_seq_lens_q == nullptr || call->cu_seq_lens_k == nullptr) {
    return refuse(call,
                  "the fa4 module reads the sequences' offsets; a launch without them is a "
                  "compilation this package does not carry");
  }
  if (call->window_left >= 0) {
    return refuse(call, "the fa4 module is compiled without an attention window");
  }
  if (!(call->scale > 0.0)) {
    return refuse(call, "dense_attention needs a positive softmax scale");
  }
  if (call->batch <= 0 || call->q_tokens <= 0 || call->kv_tokens <= 0) {
    return refuse(call, "dense_attention needs a positive batch and token count");
  }
  if (call->q_heads % call->kv_heads != 0) {
    return refuse(call, "dense_attention needs q_heads to be a multiple of kv_heads");
  }
  if (call->max_kv_len > 0 && call->kv_tokens > call->batch * call->max_kv_len) {
    return refuse(call, "a key sequence is longer than the length the module serves");
  }

  flash_attn_dense_Tensor_mQ_t query{};
  query.data = const_cast<void*>(call->query);
  query.dynamic_shapes[0] = static_cast<int32_t>(call->q_tokens);
  query.dynamic_shapes[1] = static_cast<int32_t>(call->q_heads);
  query.dynamic_shapes[2] = static_cast<int32_t>(call->head_dim);
  query.dynamic_strides[0] = call->q_stride_tokens;
  query.dynamic_strides[1] = call->q_stride_heads;

  flash_attn_dense_Tensor_mO_t out{};
  out.data = call->out;
  out.dynamic_shapes[0] = static_cast<int32_t>(call->q_tokens);
  out.dynamic_shapes[1] = static_cast<int32_t>(call->q_heads);
  out.dynamic_shapes[2] = static_cast<int32_t>(call->head_dim);
  out.dynamic_strides[0] = call->o_stride_tokens;
  out.dynamic_strides[1] = call->o_stride_heads;

  flash_attn_dense_Tensor_mK_t key{};
  key.data = const_cast<void*>(call->key);
  key.dynamic_shapes[0] = static_cast<int32_t>(call->kv_tokens);
  key.dynamic_shapes[1] = static_cast<int32_t>(call->kv_heads);
  key.dynamic_shapes[2] = static_cast<int32_t>(call->head_dim);
  key.dynamic_strides[0] = call->kv_stride_tokens;
  key.dynamic_strides[1] = call->kv_stride_heads;

  flash_attn_dense_Tensor_mV_t value{};
  value.data = const_cast<void*>(call->value);
  value.dynamic_shapes[0] = static_cast<int32_t>(call->kv_tokens);
  value.dynamic_shapes[1] = static_cast<int32_t>(call->kv_heads);
  value.dynamic_shapes[2] = static_cast<int32_t>(call->head_dim);
  value.dynamic_strides[0] = call->kv_stride_tokens;
  value.dynamic_strides[1] = call->kv_stride_heads;

  flash_attn_dense_Tensor_mCuSeqlensQ_t cu_seqlens_q{};
  cu_seqlens_q.data = const_cast<void*>(static_cast<const void*>(call->cu_seq_lens_q));
  cu_seqlens_q.dynamic_shapes[0] = static_cast<int32_t>(call->batch + 1);

  flash_attn_dense_Tensor_mCuSeqlensK_t cu_seqlens_k{};
  cu_seqlens_k.data = const_cast<void*>(static_cast<const void*>(call->cu_seq_lens_k));
  cu_seqlens_k.dynamic_shapes[0] = static_cast<int32_t>(call->batch + 1);

  const int32_t status = cute_dsl_flash_attn_dense_wrapper(
      &module(), &query, &key, &value, &out, static_cast<float>(call->scale), &cu_seqlens_q,
      &cu_seqlens_k, static_cast<int32_t>(call->max_q_len),
      reinterpret_cast<cudaStream_t>(stream));
  if (status != 0) {
    return refuse(call, "FlashAttention 4's launch reported an error");
  }
  return 0;
}
