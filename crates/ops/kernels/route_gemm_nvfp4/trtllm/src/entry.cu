// SPDX-License-Identifier: Apache-2.0

// The route_gemm_nvfp4 entry: trtllm-gen's routed block-scaled second matrix behind the routed
// products' shared entry (`uniserve_trtllm/routed_gemm.h`). This unit states what is the second
// matrix's own: its runner, the rows it sizes its tactics by (the routes it reads), the counts a
// call must satisfy and the operands the runner takes.

#include <uniserve_route_gemm_nvfp4.h>

#include <uniserve_trtllm/routed_gemm.h>

namespace {

namespace routed = uniserve::trtllm::routed;

struct SecondMatrix {
  using Runner = tensorrt_llm::kernels::trtllmgen_moe::Gemm2::Runner;
  using Call = uniserve_route_gemm_nvfp4_call;

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
        /*useShuffledMatrix=*/true,
        /*weight_layout=*/routed::kWeightLayout,
        // The library's FP4 kernels read the encoder's per-token factor beside the block scales.
        /*usePerTokenScaling=*/true,
        /*usePerChannelScaling=*/false);
    return one;
  }

  // The second matrix's tactics are sized by the routes it reads.
  static int32_t rows(Call const* call) { return static_cast<int32_t>(call->routes); }

  // The module's activations and output are BF16 rows.
  static char const* refused_types(Call const* call) {
    return call->dtype == UNISERVE_DTYPE_BF16
               ? nullptr
               : "the product's output is compiled as bf16 and this module is the bf16 one";
  }

  // A call reads at least one token's routes.
  static bool counts(Call const* call) { return call->routes >= call->top; }

  static void run(Runner& runner, Call* call, routed::Problem const& problem,
                  int32_t* padded_routes, void* workspace, int device, cudaStream_t order,
                  int32_t config) {
    runner.run(const_cast<void*>(call->hidden), const_cast<void*>(call->hidden_scale),
               const_cast<void*>(call->weights), const_cast<void*>(call->scales),
               const_cast<void*>(call->hidden_token_scale), /*perChannelScales=*/nullptr,
               const_cast<float*>(static_cast<float const*>(call->global_scale)),
               /*bias=*/nullptr, call->out, /*outputScale=*/nullptr, problem.top, problem.hidden,
               problem.intermediate, problem.experts, problem.rows,
               const_cast<int32_t*>(call->num_non_exiting_tiles), padded_routes,
               const_cast<int32_t*>(call->tile_to_expert),
               const_cast<int32_t*>(call->tile_to_limit), workspace, device, order, config,
               /*enable_pdl=*/false);
  }
};

}  // namespace

extern "C" int uniserve_route_gemm_nvfp4(uniserve_route_gemm_nvfp4_call* call, void* stream) {
  return routed::entry<SecondMatrix>(call, stream);
}
