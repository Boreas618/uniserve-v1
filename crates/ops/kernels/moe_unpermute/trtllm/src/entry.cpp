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
#include <exception>

namespace {

namespace cute_dsl = tensorrt_llm::kernels::cute_dsl;
// The message an entry writes before it returns a nonzero status.
int32_t refuse(uniserve_unpermute_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

// The dtype dispatch, instantiated once per input convention: the artifact's kernel takes the
// convention as a compile-time flag, and the call names it at run time.
template <bool kExpanded>
int32_t combine(const uniserve_unpermute_call* call, cudaStream_t stream, int32_t tokens,
                int32_t top, int32_t hidden) {
  constexpr bool kEnablePdl = true;
    if (call->input_dtype == UNISERVE_DTYPE_BF16) {
      if (call->scales_dtype == UNISERVE_DTYPE_F32) {
        cute_dsl::moeUnpermute<__nv_bfloat16, float>(
            static_cast<__nv_bfloat16 const*>(call->routes), static_cast<__nv_bfloat16*>(call->out),
            call->expanded_to_permuted, static_cast<float const*>(call->scales), tokens, hidden, top,
            kExpanded, kEnablePdl, stream);
        return 0;
      }
      if (call->scales_dtype == UNISERVE_DTYPE_BF16) {
        cute_dsl::moeUnpermute<__nv_bfloat16, __nv_bfloat16>(
            static_cast<__nv_bfloat16 const*>(call->routes), static_cast<__nv_bfloat16*>(call->out),
            call->expanded_to_permuted, static_cast<__nv_bfloat16 const*>(call->scales), tokens, hidden,
            top, kExpanded, kEnablePdl, stream);
        return 0;
      }
      return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute serves bf16 routes with f32 or bf16 scales");
    }
    if (call->input_dtype == UNISERVE_DTYPE_F16) {
      if (call->scales_dtype == UNISERVE_DTYPE_F32) {
        cute_dsl::moeUnpermute<__half, float>(
            static_cast<__half const*>(call->routes), static_cast<__half*>(call->out),
            call->expanded_to_permuted, static_cast<float const*>(call->scales), tokens, hidden, top,
            kExpanded, kEnablePdl, stream);
        return 0;
      }
      if (call->scales_dtype == UNISERVE_DTYPE_F16) {
        cute_dsl::moeUnpermute<__half, __half>(
            static_cast<__half const*>(call->routes), static_cast<__half*>(call->out),
            call->expanded_to_permuted, static_cast<__half const*>(call->scales), tokens, hidden, top,
            kExpanded, kEnablePdl, stream);
        return 0;
      }
      return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute serves f16 routes with f32 or f16 scales");
    }
  return refuse(const_cast<uniserve_unpermute_call*>(call), "moe_unpermute serves bf16 or f16 routes");
}

}  // namespace

int32_t launch(const uniserve_unpermute_call* call, void* stream) {
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

  // The input's row convention is the call's, not this unit's: the dense product writes one row per
  // expanded route and the encoded one writes permuted rows, so the module is compiled for one of
  // them (`ops::moe::combine::Layout`, a descriptor field) and the entry dispatches on it.
  const bool expanded = call->layout == 1;

  if (expanded) {
    return combine<true>(call, cuda_stream, tokens, top, hidden);
  }
  return combine<false>(call, cuda_stream, tokens, top, hidden);
}

/// The operator's entry: the launch, and an exception turned into a message.
///
/// An exception raised inside the kernel's own code would otherwise cross this C boundary,
/// where nothing may unwind: the process ends and the only message a log holds is the runtime's
/// own (`internal error: entered unreachable code`), which names neither the operator nor the
/// reason. A thrown exception states one, so it goes into the call's error buffer and the entry
/// answers a refusal like every other one it makes.
extern "C" int32_t uniserve_moe_unpermute(const uniserve_unpermute_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  try {
    return launch(const_cast<uniserve_unpermute_call*>(call), stream);
  } catch (std::exception const& thrown) {
    return refuse(const_cast<uniserve_unpermute_call*>(call), thrown.what());
  } catch (...) {
    return refuse(const_cast<uniserve_unpermute_call*>(call), "the kernel's launch failed without stating why");
  }
}
