// SPDX-License-Identifier: Apache-2.0

// The route_gemm_mxfp4 entry: trtllm-gen's routed block-scaled second matrix over the gated MXFP8
// rows and MXFP4 down weights, into BF16, behind the routed products' shared entry
// (`uniserve_trtllm/routed_gemm.h`). This unit states what is the second matrix's own: its runner,
// the rows it sizes its tactics by (the routes it reads), the counts a call must satisfy and the
// operands the runner takes.

#include <uniserve_route_gemm_mxfp4.h>

#include <uniserve_trtllm/routed_gemm.h>

namespace {

namespace routed = uniserve::trtllm::routed;

struct SecondMatrix {
  using Runner = tensorrt_llm::kernels::trtllmgen_moe::Gemm2::Runner;
  using Call = uniserve_route_gemm_mxfp4_call;

  // The runner, built once per process from the compiled geometry: MXFP8 rows against MXFP4
  // weights in their block-swizzled form, into BF16, with no per-token or per-channel factor.
  static Runner& runner() {
    static Runner one(
        /*dtypeAct=*/routed::btg::Dtype::MxE4m3,
        /*dtypeWeights=*/routed::btg::Dtype::MxE2m1,
        /*dtypeOutput=*/routed::btg::Dtype::Bfloat16,
        /*useDeepSeekFp8=*/false,
        /*tileTokensDim=*/routed::kTileTokens,
        /*useShuffledMatrix=*/true,
        /*weight_layout=*/routed::kWeightLayout,
        /*usePerTokenScaling=*/false,
        /*usePerChannelScaling=*/false);
    return one;
  }

  // The second matrix's tactics are sized by the routes it reads.
  static int32_t rows(Call const* call) { return static_cast<int32_t>(call->routes); }

  // The element types are the operator's alone: the call states none.
  static char const* refused_types(Call const*) { return nullptr; }

  // A call reads at least one token's routes.
  static bool counts(Call const* call) { return call->routes >= call->top; }

  static void run(Runner& runner, Call* call, routed::Problem const& problem,
                  int32_t* padded_routes, void* workspace, int device, cudaStream_t order,
                  int32_t config) {
    runner.run(const_cast<void*>(call->hidden), const_cast<void*>(call->hidden_scale),
               const_cast<void*>(call->weights), const_cast<void*>(call->scales),
               /*perTokenScales=*/nullptr, /*perChannelScales=*/nullptr,
               /*outputScalesScalar=*/nullptr, /*bias=*/nullptr, call->out,
               /*outputScale=*/nullptr, problem.top, problem.hidden, problem.intermediate,
               problem.experts, problem.rows, const_cast<int32_t*>(call->num_non_exiting_tiles),
               padded_routes, const_cast<int32_t*>(call->tile_to_expert),
               const_cast<int32_t*>(call->tile_to_limit), workspace, device, order, config,
               /*enable_pdl=*/false);
  }
};

}  // namespace

extern "C" int uniserve_route_gemm_mxfp4(uniserve_route_gemm_mxfp4_call* call, void* stream) {
  return routed::entry<SecondMatrix>(call, stream);
}
