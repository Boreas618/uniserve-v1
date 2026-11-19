// SPDX-License-Identifier: Apache-2.0

// The gather_gemm_mxfp4 entry: trtllm-gen's routed block-scaled first matrix over MXFP8
// activations and MXFP4 weights, fused with the clamped SwiGLU and encoding its output into MXFP8,
// behind the routed products' shared entry (`uniserve_trtllm/routed_gemm.h`). This unit states what
// is the first matrix's own: its runner, the rows it sizes its tactics by (the tokens it gathers),
// the counts a call must satisfy and the operands the runner takes.

#include <uniserve_gather_gemm_mxfp4.h>

#include <uniserve_trtllm/routed_gemm.h>

namespace {

namespace routed = uniserve::trtllm::routed;

using ActivationType = tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;

struct FirstMatrix {
  using Runner = tensorrt_llm::kernels::trtllmgen_moe::PermuteGemm1::Runner;
  using Call = uniserve_gather_gemm_mxfp4_call;

  // The runner, built once per process from the compiled geometry.
  //
  // MXFP8 activations against MXFP4 weights, the gated rows written in MXFP8 again (the runner's
  // output type is the activation's, as FlashInfer's MoE builds it). The weights are read in their
  // block-swizzled form (`useShuffledMatrix`), which the loader publishes. The block scales carry
  // the whole scale of each block, so no per-token or per-channel factor is read.
  static Runner& runner() {
    static Runner one(
        /*dtypeAct=*/routed::btg::Dtype::MxE4m3,
        /*dtypeWeights=*/routed::btg::Dtype::MxE2m1,
        /*dtypeOutput=*/routed::btg::Dtype::MxE4m3,
        /*useDeepSeekFp8=*/false,
        /*tileTokensDim=*/routed::kTileTokens,
        /*activationType=*/ActivationType::Swiglu,
        /*useShuffledMatrix=*/true,
        /*weight_layout=*/routed::kWeightLayout,
        // The operator has no bias, and `None` is what the runner takes for a null bias of a
        // bias-carrying cubin (the NVFP4 entry states why).
        /*biasType=*/batchedGemm::gemm::BiasType::None,
        /*usePerTokenScaling=*/false,
        /*usePerChannelScaling=*/false);
    return one;
  }

  // The first matrix's tactics are sized by the tokens it gathers.
  static int32_t rows(Call const* call) { return static_cast<int32_t>(call->tokens); }

  // The element types are the operator's alone: the call states none.
  static char const* refused_types(Call const*) { return nullptr; }

  // Every token gathers its `top` routes.
  static bool counts(Call const* call) {
    return call->tokens > 0 && call->routes >= call->tokens * call->top;
  }

  static void run(Runner& runner, Call* call, routed::Problem const& problem,
                  int32_t* padded_routes, void* workspace, int device, cudaStream_t order,
                  int32_t config) {
    // The SwiGLU's alpha and beta are absent: `silu(gate) · up`. The clamp table is one limit per
    // expert, applied before the activation: the gate from above, the up half on both sides.
    runner.run(const_cast<void*>(call->hidden), const_cast<void*>(call->hidden_scale),
               const_cast<void*>(call->weights), const_cast<void*>(call->scales),
               /*perTokenScales=*/nullptr, /*perChannelScales=*/nullptr,
               /*outputScalesScalar=*/nullptr, /*outputScalesGateScalar=*/nullptr,
               /*bias=*/nullptr, /*gatedActAlpha=*/nullptr, /*gatedActBeta=*/nullptr,
               const_cast<float*>(static_cast<float const*>(call->limit)),
               /*permutedIdxToBiasRowIdx=*/nullptr, call->out, call->out_scale, problem.top,
               problem.hidden, problem.intermediate, problem.experts, problem.rows,
               const_cast<int32_t*>(call->permuted_to_token),
               const_cast<int32_t*>(call->num_non_exiting_tiles), padded_routes,
               const_cast<int32_t*>(call->tile_to_expert),
               const_cast<int32_t*>(call->tile_to_limit), workspace,
               /*useRoutingScalesOnInput=*/false, device, order, config, /*enable_pdl=*/false);
  }
};

}  // namespace

extern "C" int uniserve_gather_gemm_mxfp4(uniserve_gather_gemm_mxfp4_call* call, void* stream) {
  return routed::entry<FirstMatrix>(call, stream);
}
