// SPDX-License-Identifier: Apache-2.0

// The gather_gemm_nvfp4 entry: trtllm-gen's routed block-scaled first matrix, fused with the gated
// activation, behind the routed products' shared entry (`uniserve_trtllm/routed_gemm.h`). This
// unit states what is the first matrix's own: its runner, the rows it sizes its tactics by (the
// tokens it gathers), the counts a call must satisfy and the operands the runner takes.

#include <uniserve_gather_gemm_nvfp4.h>

#include <uniserve_trtllm/routed_gemm.h>

#ifndef UNISERVE_GELU_TANH
#error "UNISERVE_GELU_TANH is not defined; the build states the descriptor's activation"
#endif

namespace {

namespace routed = uniserve::trtllm::routed;

using ActivationType = tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;

// The fused gated epilogue the descriptor's `activation` names, which the build states as
// `UNISERVE_GELU_TANH`. The library calls `silu(gate) · up` Swiglu and `gelu_tanh(gate) · up`
// Geglu, and the kernels the build published are the same activation's.
constexpr ActivationType kActivation =
    UNISERVE_GELU_TANH ? ActivationType::Geglu : ActivationType::Swiglu;

struct FirstMatrix {
  using Runner = tensorrt_llm::kernels::trtllmgen_moe::PermuteGemm1::Runner;
  using Call = uniserve_gather_gemm_nvfp4_call;

  // The runner, built once per process from the compiled geometry.
  //
  // The kernels must be read in their block-swizzled form (`useShuffledMatrix`): the loader states
  // the shard's layout and refuses a plain one, so by the time a call arrives the operand is the
  // layout the kernels index.
  static Runner& runner() {
    static Runner one(
        /*dtypeAct=*/routed::btg::Dtype::E2m1,
        /*dtypeWeights=*/routed::btg::Dtype::E2m1,
        /*dtypeOutput=*/routed::btg::Dtype::Bfloat16,
        /*useDeepSeekFp8=*/false,
        /*tileTokensDim=*/routed::kTileTokens,
        /*activationType=*/kActivation,
        /*useShuffledMatrix=*/true,
        /*weight_layout=*/routed::kWeightLayout,
        // The operator has no bias, and `None` is the only bias type the library's cubins and the
        // runner can agree on: the runner's options hardcode the bias dtype to BF16 while every
        // gated BF16-output cubin in this library is `biasFp32M`, so a bias kernel can never match.
        // With `None` the runner skips the bias checks and accepts a null pointer (its own words:
        // "we can pass nullptr bias to get biasType::None for any bias kernel"), which is exactly
        // what this bias-free operator passes.
        /*biasType=*/batchedGemm::gemm::BiasType::None,
        // The library's FP4 kernels read the encoder's per-token factor beside the block scales.
        /*usePerTokenScaling=*/true,
        /*usePerChannelScaling=*/false);
    return one;
  }

  // The first matrix's tactics are sized by the tokens it gathers.
  static int32_t rows(Call const* call) { return static_cast<int32_t>(call->tokens); }

  // The module's activations and output are BF16 rows.
  static char const* refused_types(Call const* call) {
    return call->dtype == UNISERVE_DTYPE_BF16
               ? nullptr
               : "the product's output is compiled as bf16 and this module is the bf16 one";
  }

  // Every token gathers its `top` routes.
  static bool counts(Call const* call) {
    return call->tokens > 0 && call->routes >= call->tokens * call->top;
  }

  static void run(Runner& runner, Call* call, routed::Problem const& problem,
                  int32_t* padded_routes, void* workspace, int device, cudaStream_t order,
                  int32_t config) {
    runner.run(const_cast<void*>(call->hidden), const_cast<void*>(call->hidden_scale),
               const_cast<void*>(call->weights), const_cast<void*>(call->scales),
               const_cast<void*>(call->hidden_token_scale), /*perChannelScales=*/nullptr,
               const_cast<float*>(static_cast<float const*>(call->global_scale)),
               const_cast<float*>(static_cast<float const*>(call->global_scale_gate)),
               /*bias=*/nullptr, /*gatedActAlpha=*/nullptr, /*gatedActBeta=*/nullptr,
               /*clampLimit=*/nullptr, /*permutedIdxToBiasRowIdx=*/nullptr, call->out,
               /*outputScale=*/nullptr, problem.top, problem.hidden, problem.intermediate,
               problem.experts, problem.rows, const_cast<int32_t*>(call->permuted_to_token),
               const_cast<int32_t*>(call->num_non_exiting_tiles), padded_routes,
               const_cast<int32_t*>(call->tile_to_expert),
               const_cast<int32_t*>(call->tile_to_limit), workspace,
               /*useRoutingScalesOnInput=*/false, device, order, config, /*enable_pdl=*/false);
  }
};

}  // namespace

extern "C" int uniserve_gather_gemm_nvfp4(uniserve_gather_gemm_nvfp4_call* call, void* stream) {
  return routed::entry<FirstMatrix>(call, stream);
}
