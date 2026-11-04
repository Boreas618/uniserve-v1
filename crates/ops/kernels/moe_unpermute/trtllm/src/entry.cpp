// SPDX-License-Identifier: Apache-2.0

// The moe_unpermute module's C entry: dispatches the operator's call onto
// trtllm-gen's combine kernel (`moeUnpermute` in
// `csrc/nv_internal/tensorrt_llm/kernels/cuteDslKernels/moeUtils.cu`), scaled by
// the route weights in FP32 and rounded once into the output.
//
// The kernel is the trtllm-gen one; this unit only chooses the instantiation the
// call's dtypes name and refuses a call the module does not serve.

#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "entry.h"

#include <cstdio>
#include <cstring>

namespace {

namespace cute_dsl = tensorrt_llm::kernels::cute_dsl;

// The message an entry writes before it returns a nonzero status.
int32_t refuse(uniserve_unpermute_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

}  // namespace

extern "C" int32_t uniserve_moe_unpermute(const uniserve_unpermute_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  // The combine kernel rounds once into the input type, so an output of another
  // type is a call the module does not serve.
  if (call->input_dtype != call->output_dtype) {
    return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute needs one input and output dtype");
  }
  if (call->routes == nullptr || call->out == nullptr || call->expanded_to_permuted == nullptr ||
      call->scales == nullptr) {
    return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute needs routes, out, expanded_to_permuted and scales");
  }
  if (call->hidden % 8 != 0) {
    return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute needs a hidden width of whole 8-element groups");
  }

  const auto tokens = static_cast<int32_t>(call->tokens);
  const auto top = static_cast<int32_t>(call->top);
  const auto hidden = static_cast<int32_t>(call->hidden);
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // The routes are indexed by `expanded_to_permuted`, so the kernel reads them
  // as the permuted rows they are; the dependent launch is on, which is how the
  // sort, the GEMMs and this combine overlap in one captured graph.
  constexpr bool kInputIsExpanded = false;
  constexpr bool kEnablePdl = true;

  if (call->input_dtype == UNISERVE_DTYPE_BF16) {
    if (call->scales_dtype == UNISERVE_DTYPE_F32) {
      cute_dsl::moeUnpermute<__nv_bfloat16, float>(
          static_cast<__nv_bfloat16 const*>(call->routes), static_cast<__nv_bfloat16*>(call->out),
          call->expanded_to_permuted, static_cast<float const*>(call->scales), tokens, hidden, top,
          kInputIsExpanded, kEnablePdl, cuda_stream);
      return 0;
    }
    if (call->scales_dtype == UNISERVE_DTYPE_BF16) {
      cute_dsl::moeUnpermute<__nv_bfloat16, __nv_bfloat16>(
          static_cast<__nv_bfloat16 const*>(call->routes), static_cast<__nv_bfloat16*>(call->out),
          call->expanded_to_permuted, static_cast<__nv_bfloat16 const*>(call->scales), tokens, hidden,
          top, kInputIsExpanded, kEnablePdl, cuda_stream);
      return 0;
    }
    return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute serves bf16 routes with f32 or bf16 scales");
  }
  if (call->input_dtype == UNISERVE_DTYPE_F16) {
    if (call->scales_dtype == UNISERVE_DTYPE_F32) {
      cute_dsl::moeUnpermute<__half, float>(
          static_cast<__half const*>(call->routes), static_cast<__half*>(call->out),
          call->expanded_to_permuted, static_cast<float const*>(call->scales), tokens, hidden, top,
          kInputIsExpanded, kEnablePdl, cuda_stream);
      return 0;
    }
    if (call->scales_dtype == UNISERVE_DTYPE_F16) {
      cute_dsl::moeUnpermute<__half, __half>(
          static_cast<__half const*>(call->routes), static_cast<__half*>(call->out),
          call->expanded_to_permuted, static_cast<__half const*>(call->scales), tokens, hidden, top,
          kInputIsExpanded, kEnablePdl, cuda_stream);
      return 0;
    }
    return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute serves f16 routes with f32 or f16 scales");
  }
  return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute serves bf16 or f16 routes");
}
