// SPDX-License-Identifier: Apache-2.0

// The paged_attention module's fa4 backend: the operator's call turned into
// FlashAttention 4's paged forward pass.
//
// The kernel is FlashAttention's: `build.py` compiled the configuration the
// vendor's own selection chose for this descriptor's geometry and exported it to
// C, so this unit's work is the operator's side of the trade — which of the
// call's fields the pass reads, and what the call must say for the compiled
// configuration to be the right one. The exported tensors are the vendor's own
// shapes: whole queries per request (`[tokens, heads, head_dim]`), a paged cache,
// the page table about which requests read which pages, and the two extent arrays
// the kernel indexes by request.
//
// The pass reads no cross-CTA counter and no workspace: FA4's scheduler is its
// own, so `workspace` is unused here and a call that carries one is not refused.

#include "flash_attn_paged.h"

#include "entry.h"

#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

// The exported module holds the kernel's cubin and its host launcher.
flash_attn_paged_Kernel_Module_t& module() {
  static flash_attn_paged_Kernel_Module_t handle{};
  static std::once_flag loaded;
  std::call_once(loaded, [] { flash_attn_paged_Kernel_Module_Load(&handle); });
  return handle;
}

// The message an entry writes before it returns a nonzero status.
int32_t refuse(uniserve_paged_attention_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_paged_attention(const uniserve_paged_attention_call* call,
                                            void* stream) {
  if (call == nullptr) {
    return 1;
  }
  auto refuse_call = [&](const char* message) {
    return refuse(const_cast<uniserve_paged_attention_call*>(call), message);
  };

  // The compiled pass is one geometry: its element type, head width, page size,
  // head counts, mask and key length are the descriptor's, so a call of another
  // geometry is refused here rather than run against tiles sized for it.
  if (call->phase != UNISERVE_PHASE_PREFILL) {
    return refuse_call("the fa4 module is compiled for whole-query prefill");
  }
  if (call->dtype_query != UNISERVE_FA4_DTYPE) {
    return refuse_call("the fa4 module is compiled for another element type");
  }
  if (call->head_dim != UNISERVE_FA4_HEAD_DIM) {
    return refuse_call("the fa4 module is compiled for another head width");
  }
  if (call->page_size != UNISERVE_FA4_PAGE) {
    return refuse_call("the fa4 module is compiled for another page size");
  }
  if (call->q_heads != UNISERVE_FA4_Q_HEADS || call->kv_heads != UNISERVE_FA4_KV_HEADS) {
    return refuse_call("the fa4 module is compiled for other head counts");
  }
  if ((call->causal != 0) != (UNISERVE_FA4_CAUSAL != 0)) {
    return refuse_call("the fa4 module is compiled for the other mask");
  }
  if (call->max_kv_len > UNISERVE_FA4_MAX_KV_LEN) {
    return refuse_call("the fa4 module is compiled for a shorter key cache");
  }
  if (call->dtype_query != call->dtype_key_value || call->dtype_query != call->dtype_output) {
    return refuse_call("the fa4 module serves one element type for query, cache and output");
  }
  if (call->dtype_query != UNISERVE_DTYPE_BF16 && call->dtype_query != UNISERVE_DTYPE_F16) {
    return refuse_call("the fa4 module serves bf16 and f16");
  }
  if (call->key_scale != nullptr || call->value_scale != nullptr) {
    return refuse_call("an unquantized cache takes no scales");
  }
  if (call->query == nullptr || call->key == nullptr || call->value == nullptr ||
      call->out == nullptr || call->page_table == nullptr || call->seq_lens == nullptr ||
      call->cum_seq_lens_q == nullptr) {
    return refuse_call(
        "paged_attention needs query, key, value, out, page_table, seq_lens and cum_seq_lens_q");
  }
  if (call->q_heads % call->kv_heads != 0) {
    return refuse_call("paged_attention needs q_heads to be a multiple of kv_heads");
  }
  if (call->window_left >= 0) {
    return refuse_call("the fa4 module is compiled without an attention window");
  }
  if (!(call->scale > 0.0)) {
    return refuse_call("paged_attention needs a positive softmax scale");
  }
  if (call->tokens <= 0 || call->batch <= 0) {
    return refuse_call("paged_attention needs a positive token and request count");
  }
  // FA4's page table is the request's pages in order; its width is the longest
  // sequence the compiled configuration was sized for.
  const int64_t max_pages = (call->max_kv_len + call->page_size - 1) / call->page_size;

  flash_attn_paged_Tensor_mQ_t query{};
  query.data = const_cast<void*>(call->query);
  query.dynamic_shapes[0] = static_cast<int32_t>(call->tokens);
  query.dynamic_shapes[1] = static_cast<int32_t>(call->q_heads);
  query.dynamic_shapes[2] = static_cast<int32_t>(call->head_dim);
  query.dynamic_strides[0] = call->q_stride_tokens;
  query.dynamic_strides[1] = call->q_stride_heads;

  flash_attn_paged_Tensor_mO_t out{};
  out.data = call->out;
  out.dynamic_shapes[0] = static_cast<int32_t>(call->tokens);
  out.dynamic_shapes[1] = static_cast<int32_t>(call->q_heads);
  out.dynamic_shapes[2] = static_cast<int32_t>(call->head_dim);
  out.dynamic_strides[0] = call->o_stride_tokens;
  out.dynamic_strides[1] = call->o_stride_heads;

  // The cache is read page by page: the page's own stride, then the token and
  // head strides inside it.
  flash_attn_paged_Tensor_mK_t key{};
  key.data = const_cast<void*>(call->key);
  key.dynamic_shapes[0] = static_cast<int32_t>(call->num_pages);
  key.dynamic_shapes[1] = static_cast<int32_t>(call->page_size);
  key.dynamic_shapes[2] = static_cast<int32_t>(call->kv_heads);
  key.dynamic_shapes[3] = static_cast<int32_t>(call->head_dim);
  key.dynamic_strides[0] = call->kv_stride_page;
  key.dynamic_strides[1] = call->kv_stride_tokens;
  key.dynamic_strides[2] = call->kv_stride_heads;

  flash_attn_paged_Tensor_mV_t value{};
  value.data = const_cast<void*>(call->value);
  value.dynamic_shapes[0] = static_cast<int32_t>(call->num_pages);
  value.dynamic_shapes[1] = static_cast<int32_t>(call->page_size);
  value.dynamic_shapes[2] = static_cast<int32_t>(call->kv_heads);
  value.dynamic_shapes[3] = static_cast<int32_t>(call->head_dim);
  value.dynamic_strides[0] = call->kv_stride_page;
  value.dynamic_strides[1] = call->kv_stride_tokens;
  value.dynamic_strides[2] = call->kv_stride_heads;

  flash_attn_paged_Tensor_mCuSeqlensQ_t cu_seqlens_q{};
  cu_seqlens_q.data = const_cast<void*>(static_cast<const void*>(call->cum_seq_lens_q));
  cu_seqlens_q.dynamic_shapes[0] = static_cast<int32_t>(call->batch + 1);

  flash_attn_paged_Tensor_mSeqUsedK_t seq_lens{};
  seq_lens.data = const_cast<void*>(static_cast<const void*>(call->seq_lens));
  seq_lens.dynamic_shapes[0] = static_cast<int32_t>(call->batch);

  flash_attn_paged_Tensor_mPageTable_t page_table{};
  page_table.data = const_cast<void*>(static_cast<const void*>(call->page_table));
  page_table.dynamic_shapes[0] = static_cast<int32_t>(call->batch);
  page_table.dynamic_shapes[1] = static_cast<int32_t>(max_pages);
  page_table.dynamic_strides[0] = max_pages;

  return cute_dsl_flash_attn_paged_wrapper(&module(), &query, &key, &value, &out,
                                           static_cast<float>(call->scale), &cu_seqlens_q,
                                           &seq_lens, &page_table,
                                           static_cast<int32_t>(call->max_q_len),
                                           reinterpret_cast<cudaStream_t>(stream));
}
